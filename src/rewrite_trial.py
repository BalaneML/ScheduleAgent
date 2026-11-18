"""
rewrite_trial.py
================
作業 D：書き直しを足すと、時刻別行動者率の誤差が床にどこまで近づくかを見る試行

出発点は llm_memory_trial の出力（既定は memory_person の 100 人）。素の生成能力と書き直しの効果を、
同じ人たちで比べられる。1 回の書き直しは次のとおり。

    1. 選ぶ    その人を抜いたときに、時刻別 MAE と日次 MAE の和（重み 1:1）が最も下がる人をコードが選ぶ。
               直近 cooldown 回に書き直しを試した人は選ばない（同じ人を何度も選び続けないため）
    2. 書かせる その人の今の 1 日と、その人を抜いた集団のメモリ（memory_day の文）と、その人の条件
               （person_conditions を「残り 1 人」として呼ぶ）を LLM に渡し、1 日を書き直させる
    3. 採る    書き直した 1 日に入れ替えたとき、和が下がれば採る。下がらなければ元の 1 日に戻す

時刻別 MAE が床以下になるか、試した回数が max_rewrites に達したら止める。採るかどうかは集計表との差だけで
決め、R1〜R4 の違反は記録だけする（違反が増えるかを見るため）。

    # サーバーなしの空回し
    .venv/bin/python src/rewrite_trial.py --dry-run --start outputs/llm_memory_trial/llm_gemma4-12b_N100_seed42_v2 \\
        --groups 5 --max-rewrites 10 --tag dry
    # 本番
    .venv/bin/python src/rewrite_trial.py --start outputs/llm_memory_trial/llm_gemma4-12b_N100_seed42_v2 \\
        --model-tag gemma4-12b --groups 5 0 --extra-json '{"top_k": 64}'

出力（outputs/rewrite_trial/{stem}/）

    group{d:02d}_rewrite.npz     X (N, 96) int8。書き直しを終えた集団
    group{d:02d}_rewrite.jsonl   1 回 1 行。選んだ人・プロンプト・応答・採ったか・前後の誤差
    curve.csv                    試した回数ごとの時刻別 MAE・日次 MAE・違反
    summary.csv                  群ごとの指標（書き直し前と後）

```mermaid
flowchart TD
    S["np.load(start)['X']<br/>X (N, 96)"] --> PICK["pick_person(X, target, part, recent)<br/>j"]
    PICK --> REST["X_rest = X から j を抜く"]
    REST --> MEM["memory_day_text(X_rest, ...)<br/>person_conditions(X_rest, ..., n_people=N)"]
    MEM --> MSG["rewrite_messages(d, X[j], memory)"]
    MSG --> LLM["client.chat(messages, seed)<br/>row (96,)"]
    LLM --> ACC{"fit_error(X で j を row に替える)<br/>の和が下がる?"}
    ACC -- yes --> UPD["X[j] = row"]
    ACC -- no --> KEEP["X はそのまま"]
    UPD --> STOP{"時刻別 MAE ≤ 床<br/>または max_rewrites 回"}
    KEEP --> STOP
    STOP -- no --> PICK
```
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import llm_memory_trial as trial  # noqa: E402
import schedule_metrics as sm  # noqa: E402
import schedule_rules as sr  # noqa: E402
import stula_targets as tg  # noqa: E402

OUT_ROOT = tg.REPO_ROOT / "outputs" / "rewrite_trial"


def fit_error(X: npt.NDArray[np.integer], target: npt.NDArray[np.float64],
              part: npt.NDArray[np.float64]) -> tuple[float, float]:
    """較正に使う 2 表との差 (時刻別 MAE, 日次 MAE)。どちらもポイント"""
    rate = float(np.abs(sm.slot_rates(X) - target).mean() * 100)
    day = float(np.abs(sm.participation(X) - part).mean() * 100)
    return rate, day


def pick_person(X: npt.NDArray[np.integer], target: npt.NDArray[np.float64], part: npt.NDArray[np.float64],
                recent: set[int]) -> int:
    """抜いたときに時刻別 MAE と日次 MAE の和が最も下がる人を選ぶ（recent に入っている人は除く）

    Args:
        X: (N, 96) いまの集団
        target: (20, 96) 時刻別行動者率の公表値
        part: (20,) 日次行動者率の公表値
        recent: 直近に書き直しを試した人の番号

    Returns:
        選んだ人の番号 j
    """
    n = len(X)
    onehot = (np.asarray(X)[:, None, :] == np.arange(tg.NUM_ACT20)[None, :, None])     # (N, 20, 96)
    present = onehot.any(axis=2)                                                        # (N, 20)
    slot_sum, day_sum = onehot.sum(axis=0), present.sum(axis=0)
    rate_wo = (slot_sum[None] - onehot) / (n - 1)                                       # (N, 20, 96)
    day_wo = (day_sum[None] - present) / (n - 1)                                        # (N, 20)
    score = (np.abs(rate_wo - target[None]).mean(axis=(1, 2)) + np.abs(day_wo - part[None]).mean(axis=1)) * 100
    order = np.argsort(score, kind="stable")
    for j in order:
        if int(j) not in recent:
            return int(j)
    return int(order[0])


def diary_text(row: npt.NDArray[np.integer]) -> str:
    """1 日を「04:00-07:00 Sleep」の行の並びにする"""
    lines, start = [], "04:00"
    for seg in trial.slots_to_segments(np.asarray(row, dtype=np.int8)):
        lines.append(f"- {start}-{seg['end']} {seg['activity']}")
        start = seg["end"]
    return "\n".join(lines)


def rewrite_messages(d: int, row: npt.NDArray[np.integer], memory: str) -> list[dict[str, str]]:
    """書き直しのプロンプトを作る

    Args:
        d: 群の番号
        row: (96,) その人の今の 1 日
        memory: その人を抜いた集団のメモリと、その人の条件

    Returns:
        system と user のメッセージ
    """
    user = ["Rewrite the diary of one person for a weekday (Monday to Friday) in October 2021.",
            f"Person: {trial.group_text(d)}.",
            "",
            "This is the person's current diary:",
            diary_text(row),
            "",
            "Rewrite the whole day as a new diary of the same person. Change what is needed so that the group",
            "as a whole moves closer to the survey statistics below, and keep the day realistic for one person."]
    if memory:
        user += ["", memory]
    return [{"role": "system", "content": trial.SYSTEM_PROMPT}, {"role": "user", "content": "\n".join(user)}]


def rewrite_group(d: int, X0: npt.NDArray[np.int8], target: npt.NDArray[np.float64], part: npt.NDArray[np.float64],
                  client: trial.Chat, seed: int, top_k: int, min_gap: float, person_lines: int, max_rewrites: int,
                  cooldown: int, max_tries: int, log_path: Path) -> tuple[npt.NDArray[np.int8], list[dict[str, float]]]:
    """1 群ぶんの書き直しを回す

    Args:
        d: 群の番号
        X0: (N, 96) 出発点の集団（書き直さない素の生成物）
        target: (20, 96) 時刻別行動者率の公表値
        part: (20,) 日次行動者率の公表値
        client: LLM への問い合わせ
        seed: 種。t 回目の try 回目は seed · 100000 + t · 10 + try。条件を引く乱数は (seed, d) から作る
        top_k: メモリに挙げる個数
        min_gap: メモリに挙げる差・条件を付ける差の下限（割合）
        person_lines: 日次の条件の数の上限
        max_rewrites: 試す回数の上限
        cooldown: 直近この回数に試した人は選ばない
        max_tries: 形式が正しくないときにやり直す回数の上限
        log_path: 1 回 1 行の記録の書き出し先（jsonl）

    Returns:
        書き直しを終えた X (N, 96) と、試した回数ごとの誤差の推移
    """
    X = np.array(X0, dtype=np.int8, copy=True)
    n_people = len(X)
    floor = sm.floor_mae_pt(target, n_people)
    rng = np.random.default_rng([seed, d, 99])
    told = np.zeros((n_people, tg.NUM_ACT20), dtype=np.int8)
    rate, day = fit_error(X, target, part)
    curve = [{"rewrites": 0.0, "accepted": 0.0, "rate_mae_pt": rate, "part_mae_pt": day,
              "violations_mean": float(sr.violations(X).mean())}]
    recent: list[int] = []
    n_accepted = 0
    with open(log_path, "w", encoding="utf-8") as log:
        for t in range(max_rewrites):
            if rate <= floor:
                break
            j = pick_person(X, target, part, set(recent[-cooldown:]))
            X_rest = np.delete(X, j, axis=0)
            u = rng.random(tg.NUM_ACT20)
            conds = trial.person_conditions(X_rest, target, part, n_people, told[:n_people - 1], u, person_lines,
                                            min_gap)
            memory = "\n\n".join(s for s in (trial.memory_day_text(X_rest, target, part, top_k, min_gap),
                                             trial.person_text(conds)) if s)
            messages = rewrite_messages(d, X[j], memory)
            row, problem, content, meta, tries = None, "", "", {}, 0
            t0 = time.time()
            while row is None:
                content, meta = client.chat(messages, seed * 100000 + t * 10 + tries)
                tries += 1
                try:
                    segments = json.loads(content).get("segments")
                except (json.JSONDecodeError, AttributeError):
                    segments = None
                row, problem = trial.segments_to_slots(segments, repair=tries >= max_tries)
                if row is None and tries >= max_tries:
                    raise RuntimeError(f"群 {d} の {t} 回目: {max_tries} 回やり直しても形式が直らない（{problem}）")
            X_new = X.copy()
            X_new[j] = row
            rate_new, day_new = fit_error(X_new, target, part)
            accepted = rate_new + day_new < rate + day
            if accepted:
                X, rate, day = X_new, rate_new, day_new
                n_accepted += 1
            recent.append(j)
            curve.append({"rewrites": float(t + 1), "accepted": float(n_accepted), "rate_mae_pt": rate,
                          "part_mae_pt": day, "violations_mean": float(sr.violations(X).mean())})
            sec = time.time() - t0
            log.write(json.dumps({"group": d, "t": t, "person": j, "accepted": bool(accepted),
                                  "before": [curve[-2]["rate_mae_pt"], curve[-2]["part_mae_pt"]],
                                  "candidate": [rate_new, day_new], "violations_old": int(sr.person_violation(X0[j])),
                                  "violations_new": int(sr.person_violation(row)), "tries": tries,
                                  "repaired": bool(problem), "seconds": round(sec, 2), "memory": memory,
                                  "conditions": conds, "content": content, **meta}, ensure_ascii=False) + "\n")
            log.flush()
            print(f"[rewrite] 群 {d:02d} {t + 1:>4}/{max_rewrites}  人 {j:>3}  {'採る ' if accepted else '戻す '}"
                  f"時刻別 {rate:.3f}  日次 {day:.3f}  {sec:5.1f} 秒", flush=True)
    return X, curve


def main() -> None:
    """CLI"""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=Path, required=True, help="出発点の llm_memory_trial の出力フォルダ")
    ap.add_argument("--start-condition", default="memory_person", help="出発点の条件")
    ap.add_argument("--groups", type=int, nargs="+", default=[5, 0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-rewrites", type=int, default=100, help="群あたり試す回数の上限")
    ap.add_argument("--cooldown", type=int, default=10, help="直近この回数に試した人は選ばない")
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--min-gap", type=float, default=0.03)
    ap.add_argument("--person-lines", type=int, default=5)
    ap.add_argument("--max-tries", type=int, default=3)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--model-tag", default="dry")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--extra-json", default="{}")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry-run", action="store_true", help="サーバーにつながず、偽の応答で回す")
    args = ap.parse_args()

    stem = f"rewrite_{args.model_tag}_from_{args.start.name}_{args.start_condition}" + (f"_{args.tag}" if args.tag else "")
    out_dir = OUT_ROOT / stem
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} に既に出力がある（取り違え防止）。--tag を変えるか、フォルダを消してから回す")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps({**{k: str(v) if isinstance(v, Path) else v
                                                      for k, v in vars(args).items()},
                                                   "system_prompt": trial.SYSTEM_PROMPT}, indent=2,
                                                  ensure_ascii=False), encoding="utf-8")

    rates20, part20 = tg.load_rates20(), tg.load_participation20()
    summary: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    for d in args.groups:
        target, part = rates20[d], part20[d]
        X0 = np.load(args.start / f"group{d:02d}_{args.start_condition}.npz")["X"]
        client: trial.Chat = trial.FakeChat(target) if args.dry_run else trial.ServerChat(
            args.base_url, args.model_tag, args.temperature, args.top_p, args.max_tokens, json.loads(args.extra_json))
        t0 = time.time()
        X, curve = rewrite_group(d, X0, target, part, client, args.seed, args.top_k, args.min_gap, args.person_lines,
                                 args.max_rewrites, args.cooldown, args.max_tries,
                                 out_dir / f"group{d:02d}_rewrite.jsonl")
        np.savez_compressed(out_dir / f"group{d:02d}_rewrite.npz", X=X)
        for when, Xs in (("before", X0), ("after", X)):
            summary.append({"stem": stem, "group": d, "when": when, "n_people": len(Xs),
                            "rewrites": curve[-1]["rewrites"] if when == "after" else 0.0,
                            "accepted": curve[-1]["accepted"] if when == "after" else 0.0,
                            "changed_people": float((Xs != X0).any(axis=1).sum()),
                            "floor_mae_pt": sm.floor_mae_pt(target, len(Xs)), **sm.evaluate(Xs, target, part),
                            **sm.rule_metrics(Xs), "minutes": (time.time() - t0) / 60 if when == "after" else 0.0})
        curves += [{"stem": stem, "group": d, **row} for row in curve]
        pd.DataFrame(summary).to_csv(out_dir / "summary.csv", index=False)
        pd.DataFrame(curves).to_csv(out_dir / "curve.csv", index=False)
    cols = ["group", "when", "rewrites", "accepted", "changed_people", "rate_mae_pt", "floor_mae_pt", "part_mae_pt",
            "switches_mean", "violations_mean", "violation_share"]
    print(pd.DataFrame(summary)[cols].round(3).to_string(index=False), flush=True)
    print(f"[rewrite] 書いた: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
