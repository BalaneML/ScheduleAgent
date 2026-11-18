"""
llm_memory_trial.py
===================
LLM が 1 人ずつ活動スケジュールを書き、生成済みの集計と集計表の差をメモリとして次の人に渡す試行（簡単な設計）

    条件 none    メモリなし。毎回同じプロンプトで 1 人ぶん書く
    条件 memory  メモリあり。生成済みの時刻別行動者率と集計表の差が大きい箇所を、
                 不足・超過それぞれ上位 top_k 個だけ文にして渡す

書いた活動スケジュールは確定で、後から直さない。メモリは 1 人足すごとに作り直す。
LLM には OpenAI 互換の API（llama.cpp の llama-server など）でつなぐ。プロンプトは英語。

    # サーバーなしの空回し（形式とループの確認）
    .venv/bin/python src/llm_memory_trial.py --dry-run --groups 5 --N 10 --tag dry
    # サーバーにつなぐ
    .venv/bin/python src/llm_memory_trial.py --model-tag gemma4-12b --groups 5 0 --N 100

出力（outputs/llm_memory_trial/{stem}/）

    group{d:02d}_{condition}.npz     X (N, 96) int8。20 分類・04:00 起点（DomainTransfer_trial の SA の best_X と同じ並び）
    group{d:02d}_{condition}.jsonl   1 人 1 行。プロンプト・応答・所要時間
    summary.csv                      群 × 条件ごとの指標
    curve.csv                        n 人目までの時刻別行動者率の誤差
    participation.csv                日次行動者率（公表値と生成物）

```mermaid
flowchart LR
    T["tg.load_rates20()[d]<br/>target (20, 96)"] --> M["memory_text(X_done, target, top_k, min_gap)"]
    M --> P["build_messages(d, memory)"]
    P --> L["client.chat(messages, seed)<br/>content"]
    L --> S["segments_to_slots(segments)<br/>row (96,)"]
    S --> X["X_done に row を足す（確定）"]
    X --> M
    X --> E["evaluate(X, target, part)"]
    PART["tg.load_participation20()[d]<br/>part (20,)"] --> E
```

★メモリに入れるのは時刻別行動者率だけ。日次行動者率（第70-3表）は LLM に見せず、評価にだけ使う。
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stula_targets as tg  # noqa: E402

OUT_ROOT = tg.REPO_ROOT / "outputs" / "llm_memory_trial"

SLOT_MIN = 15                 # 1 スロットの長さ（分）
DAY_MIN = 24 * 60
ORIGIN_MIN = 4 * 60           # スロット 0 は 04:00〜04:15（parse_timeband.py の回転と同じ）
SLOTS_PER_HOUR = 60 // SLOT_MIN
CONDITIONS = ("none", "memory")

# 社会生活基本調査 2021 調査票 A（英語版）の「Kind of activities」の表記。並びは活動コード 1〜20
ACTIVITY_NAMES: list[str] = [
    "Sleep",
    "Personal care",
    "Meals",
    "Commuting to and from school or work",
    "Work",
    "Schoolwork",
    "Housework",
    "Caring or nursing",
    "Child care",
    "Shopping",
    "Moving (excluding commuting)",
    "Watching TV, listening to the radio, reading newspapers or magazines",
    "Rest and relaxation",
    "Learning, self-education, and training (except for school work)",
    "Hobbies and amusements",
    "Sports",
    "Volunteer and social activities",
    "Social life",
    "Medical examination or treatment",
    "Other activities",
]
ACTIVITY_CODE: dict[str, int] = {name: k for k, name in enumerate(ACTIVITY_NAMES)}

# LLM の出力の型。区間の終わりの時刻だけを書かせるので、すき間と重なりは構成上できない
DIARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "minItems": 3,
            "maxItems": 48,
            "items": {
                "type": "object",
                "properties": {
                    "activity": {"type": "string", "enum": ACTIVITY_NAMES},
                    "end": {"type": "string", "pattern": "^([01][0-9]|2[0-3]):(00|15|30|45)$"},
                },
                "required": ["activity", "end"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["segments"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = "\n".join([
    "You write realistic one-day time-use diaries of people living in Japan, in the format of the",
    "Survey on Time Use and Leisure Activities (Statistics Bureau of Japan, Questionnaire A).",
    "",
    "Rules of the diary:",
    "- The diary covers 24 hours, from 04:00 to 04:00 the next morning, in 15-minute units.",
    "- At every moment the person does exactly one activity (the main one), chosen from these 20 kinds of activities:",
    *[f"  {k + 1}. {name}" for k, name in enumerate(ACTIVITY_NAMES)],
    "- 'Commuting to and from school or work' is only the trip between home and the workplace or school.",
    "  Any other trip is 'Moving (excluding commuting)'.",
    "- Travel during work and work brought home are 'Work'.",
    "- Washing, bathing, using the toilet, dressing and grooming are 'Personal care'.",
    "- Dropping off or picking up family members is 'Housework'. Shopping is its own activity.",
    "- Naps and snacks are 'Rest and relaxation'.",
    "",
    "Output format: a JSON object with a list named \"segments\".",
    "- Each segment has \"activity\" (one of the 20 names, spelled exactly) and \"end\" (the time the activity ends).",
    "- Times are HH:MM on a 24-hour clock. Times after midnight are written 00:00, 00:15, ..., 04:00.",
    "- The first segment starts at 04:00. Each segment starts when the previous one ends.",
    "- The last segment must end at 04:00.",
])

AGE_TEXT = ["aged 15-24", "aged 25-34", "aged 35-44", "aged 45-54", "aged 55-64", "aged 65-74", "aged 75 or older"]
SEX_TEXT = ["man", "woman"]
# 調査票 A の問 11「Do you usually work?」の区分に合わせた説明
EMPLOY_TEXT = [
    "who usually does not work (for example: attends school, does housework, or other)",
    "who usually works (work for pay or profit, including part-time work and helping in a family business)",
]


class Chat(Protocol):
    """LLM への問い合わせ。本物のサーバーと空回し用の偽物で同じ形にする"""

    def chat(self, messages: list[dict[str, str]], seed: int) -> tuple[str, dict[str, Any]]:
        """メッセージを送り、(応答の本文, 付随する情報) を返す"""
        ...


class ServerChat:
    """OpenAI 互換の API（/v1/chat/completions）につなぐ

    Args:
        base_url: サーバーの URL（例 http://127.0.0.1:8080）
        model: リクエストに書くモデル名。llama-server は 1 モデルしか持たないので記録用
        temperature: サンプリングの温度
        top_p: nucleus sampling のしきい値
        max_tokens: 応答の上限のトークン数
        extra: リクエストに足す項目（例 {"top_k": 64}）
        timeout: 1 回の応答を待つ秒数
    """

    def __init__(self, base_url: str, model: str, temperature: float, top_p: float, max_tokens: int,
                 extra: dict[str, Any], timeout: float = 900.0) -> None:
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.extra = extra
        self.timeout = timeout

    def chat(self, messages: list[dict[str, str]], seed: int) -> tuple[str, dict[str, Any]]:
        """1 回問い合わせる。出力は DIARY_SCHEMA の型に縛る

        Args:
            messages: system と user のメッセージ
            seed: サンプリングの種

        Returns:
            (応答の本文, {"usage": ..., "timings": ...})
        """
        body: dict[str, Any] = {
            "model": self.model, "messages": messages, "seed": seed,
            "temperature": self.temperature, "top_p": self.top_p, "max_tokens": self.max_tokens,
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "diary", "strict": True, "schema": DIARY_SCHEMA}},
            **self.extra,
        }
        req = urllib.request.Request(self.url, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as res:
            data = json.load(res)
        content = str(data["choices"][0]["message"]["content"])
        return content, {"usage": data.get("usage", {}), "timings": data.get("timings", {})}


class FakeChat:
    """空回し用。集計表の 1 時間ごとの割合から活動を引いて、1 人ぶんの応答を作る

    Args:
        target: (20, 96) 1 群ぶんの時刻別行動者率
    """

    def __init__(self, target: npt.NDArray[np.float64]) -> None:
        self.hourly = hourly_mean(target)

    def chat(self, messages: list[dict[str, str]], seed: int) -> tuple[str, dict[str, Any]]:
        """1 時間ごとに活動を 1 つ引き、同じ活動が続く区間をまとめて JSON にする"""
        rng = np.random.default_rng(seed)
        row = np.empty(tg.NUM_SLOTS, dtype=np.int8)
        for h in range(24):
            p = self.hourly[:, h] / self.hourly[:, h].sum()
            row[h * SLOTS_PER_HOUR:(h + 1) * SLOTS_PER_HOUR] = rng.choice(tg.NUM_ACT20, p=p)
        return json.dumps({"segments": slots_to_segments(row)}), {"usage": {}, "timings": {}}


def clock(minutes_from_origin: int) -> str:
    """04:00 からの経過分を HH:MM にする（1440 分は 04:00）"""
    m = (ORIGIN_MIN + minutes_from_origin) % DAY_MIN
    return f"{m // 60:02d}:{m % 60:02d}"


def slots_to_segments(row: npt.NDArray[np.int8]) -> list[dict[str, str]]:
    """96 スロットの活動を、同じ活動が続く区間の並びにする

    Args:
        row: (96,) 活動コード 0〜19。スロット 0 が 04:00〜04:15

    Returns:
        [{"activity": 活動名, "end": 区間の終わりの時刻}, ...]
    """
    segments: list[dict[str, str]] = []
    for s in range(tg.NUM_SLOTS):
        if s + 1 == tg.NUM_SLOTS or row[s + 1] != row[s]:
            segments.append({"activity": ACTIVITY_NAMES[int(row[s])], "end": clock((s + 1) * SLOT_MIN)})
    return segments


def segments_to_slots(segments: Any, repair: bool = False) -> tuple[npt.NDArray[np.int8] | None, str]:
    """LLM が書いた区間の並びを 96 スロットにする

    区間は 04:00 から始まり、前の区間の終わりから次の区間が始まる。最後の区間は 04:00 で終わる。

    Args:
        segments: [{"activity": 活動名, "end": "HH:MM"}, ...]
        repair: True のとき、時刻が進まない区間を捨て、最後の区間を 04:00 まで延ばして必ず 96 スロットを返す

    Returns:
        (row, problem)。row は (96,) int8。形式が正しくないときは (None, 理由)。
        repair=True で直したときは row と理由の両方を返す
    """
    if not isinstance(segments, list) or not segments:
        return None, "segments が空"
    row = np.full(tg.NUM_SLOTS, -1, dtype=np.int8)
    problem = ""
    start = 0
    for seg in segments:
        name, end = seg.get("activity"), str(seg.get("end", ""))
        if name not in ACTIVITY_CODE or len(end) != 5 or end[2] != ":" or not (end[:2] + end[3:]).isdigit():
            return None, f"区間の形式が違う: {seg}"
        stop = (int(end[:2]) * 60 + int(end[3:]) - ORIGIN_MIN) % DAY_MIN
        stop = DAY_MIN if stop == 0 else stop
        if stop % SLOT_MIN != 0 or stop <= start:
            problem = problem or f"時刻が進んでいない: {end}"
            if not repair:
                return None, problem
            continue
        row[start // SLOT_MIN:stop // SLOT_MIN] = ACTIVITY_CODE[name]
        start = stop
    if start != DAY_MIN:
        problem = problem or f"04:00 で終わっていない（最後は {clock(start)}）"
        if not repair or start == 0:
            return None, problem
        row[start // SLOT_MIN:] = row[start // SLOT_MIN - 1]
    return row, problem


def hourly_mean(rates: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """(20, 96) の時刻別行動者率を 1 時間ごとの平均 (20, 24) にする"""
    return rates.reshape(tg.NUM_ACT20, 24, SLOTS_PER_HOUR).mean(axis=2)


def slot_rates(X: npt.NDArray[np.int8]) -> npt.NDArray[np.float64]:
    """生成済みの活動スケジュール (n, 96) から時刻別行動者率 (20, 96) を集計する"""
    rates = np.zeros((tg.NUM_ACT20, tg.NUM_SLOTS))
    for k in range(tg.NUM_ACT20):
        rates[k] = (X == k).mean(axis=0)
    return rates


def memory_text(X_done: npt.NDArray[np.int8], target: npt.NDArray[np.float64], top_k: int, min_gap: float) -> str:
    """メモリの文章を作る。生成済みの集計と集計表の差が大きい箇所を、不足・超過それぞれ上位 top_k 個

    差は 1 時間ごとの平均で比べる（20 活動 × 24 時間）。

    Args:
        X_done: (n, 96) これまでに確定した活動スケジュール
        target: (20, 96) 集計表の時刻別行動者率
        top_k: 不足・超過それぞれで挙げる個数
        min_gap: これより差の小さい箇所は挙げない（割合。0.03 は 3 ポイント）

    Returns:
        英語の文章。挙げる箇所が 1 つも無いときは空文字
    """
    want = hourly_mean(target)
    have = hourly_mean(slot_rates(X_done))
    gap = have - want
    order = np.argsort(gap, axis=None)

    def lines(flat_indices: npt.NDArray[np.intp], sign: int) -> list[str]:
        out: list[str] = []
        for flat in flat_indices[:top_k]:
            k, h = divmod(int(flat), 24)
            if sign * gap[k, h] < min_gap:
                break
            out.append(f"- {ACTIVITY_NAMES[k]}, {clock(h * 60)}-{clock((h + 1) * 60)}: "
                       f"survey {want[k, h]:.0%}, diaries so far {have[k, h]:.0%}")
        return out

    too_few, too_many = lines(order, -1), lines(order[::-1], 1)
    if not too_few and not too_many:
        return ""
    text = [f"MEMORY. {len(X_done)} diaries have been written for this group so far. Compared with the",
            "official survey statistics (share of people doing each activity in each hour):"]
    if too_few:
        text += ["Too FEW people so far are doing:"] + too_few
    if too_many:
        text += ["Too MANY people so far are doing:"] + too_many
    text += ["Choose this person's day so that the group as a whole moves closer to the survey statistics,",
             "while the day stays realistic for one person."]
    return "\n".join(text)


def group_text(d: int) -> str:
    """群の番号 d = (性 · 7 + 年齢) · 2 + 就業 を英語の人物の説明にする"""
    g, rest = divmod(d, tg.N_A * tg.N_E)
    a, e = divmod(rest, tg.N_E)
    return f"a {SEX_TEXT[g]} {AGE_TEXT[a]} living in Japan {EMPLOY_TEXT[e]}"


def build_messages(d: int, memory: str) -> list[dict[str, str]]:
    """1 人ぶんのプロンプトを作る

    Args:
        d: 群の番号
        memory: memory_text の文章。メモリなしのときは空文字

    Returns:
        system と user のメッセージ
    """
    user = ["Write the diary of one person for a weekday (Monday to Friday) in October 2021.",
            f"Person: {group_text(d)}.",
            "Write one individual from this group."]
    if memory:
        user += ["", memory]
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "\n".join(user)}]


def generate_group(d: int, condition: str, n_people: int, target: npt.NDArray[np.float64], client: Chat,
                   seed: int, top_k: int, min_gap: float, warmup: int, max_tries: int,
                   log_path: Path) -> npt.NDArray[np.int8]:
    """1 群ぶんを 1 人ずつ生成する。書いた活動スケジュールは確定で、後から直さない

    Args:
        d: 群の番号
        condition: "none"（メモリなし）か "memory"（メモリあり）
        n_people: 生成する人数 N
        target: (20, 96) 集計表の時刻別行動者率
        client: LLM への問い合わせ
        seed: 種。i 人目の try 回目は seed · 100000 + i · 10 + try を使う
        top_k: メモリに挙げる個数（不足・超過それぞれ）
        min_gap: メモリに挙げる差の下限
        warmup: この人数までは、メモリありでもメモリを渡さない（集計が粗すぎるため）
        max_tries: 形式が正しくないときにやり直す回数の上限。超えたら直して採用する
        log_path: 1 人 1 行の記録の書き出し先（jsonl）

    Returns:
        X (N, 96) int8
    """
    X = np.empty((n_people, tg.NUM_SLOTS), dtype=np.int8)
    with open(log_path, "w", encoding="utf-8") as log:
        for i in range(n_people):
            memory = ""
            if condition == "memory" and i >= warmup:
                memory = memory_text(X[:i], target, top_k, min_gap)
            messages = build_messages(d, memory)
            row, problem, content, meta, tries = None, "", "", {}, 0
            t0 = time.time()
            while row is None:
                content, meta = client.chat(messages, seed * 100000 + i * 10 + tries)
                tries += 1
                try:
                    segments = json.loads(content).get("segments")
                except (json.JSONDecodeError, AttributeError):
                    segments = None
                row, problem = segments_to_slots(segments, repair=tries >= max_tries)
                if row is None and tries >= max_tries:
                    raise RuntimeError(f"群 {d} の {i} 人目: {max_tries} 回やり直しても形式が直らない（{problem}）")
            X[i] = row
            sec = time.time() - t0
            log.write(json.dumps({"group": d, "condition": condition, "i": i, "tries": tries, "repaired": bool(problem),
                                  "problem": problem, "seconds": round(sec, 2), "memory": memory, "content": content,
                                  **meta}, ensure_ascii=False) + "\n")
            log.flush()
            print(f"[llm] 群 {d:02d} {condition:<6} {i + 1:>4}/{n_people}  {sec:6.1f} 秒  try {tries}"
                  f"{'  直した: ' + problem if problem else ''}", flush=True)
    return X


def evaluate(X: npt.NDArray[np.int8], target: npt.NDArray[np.float64],
             part: npt.NDArray[np.float64]) -> dict[str, float]:
    """1 群ぶんの生成物を採点する

    Args:
        X: (N, 96) 生成した活動スケジュール
        target: (20, 96) 集計表の時刻別行動者率（メモリに使った表）
        part: (20,) 日次行動者率の公表値（LLM に見せていない表）

    Returns:
        指標の名前 → 値。_pt はポイント（割合 × 100）
    """
    n_people = len(X)
    err = slot_rates(X) - target
    change = X[:, 1:] != X[:, :-1]
    # 1 スロットだけで終わる区間：両側が切り替わっているスロット（端は片側だけ見る）
    edge = np.concatenate([np.ones((n_people, 1), dtype=bool), change, np.ones((n_people, 1), dtype=bool)], axis=1)
    single = edge[:, :-1] & edge[:, 1:]
    n_episodes = change.sum() + n_people
    part_hat = np.array([(X == k).any(axis=1).mean() for k in range(tg.NUM_ACT20)])
    counts = np.stack([(X == k).sum(axis=0) for k in range(tg.NUM_ACT20)])
    over = np.maximum(counts - tg.round_slot_counts(target, n_people), 0).sum()
    return {
        "rate_mae_pt": float(np.abs(err).mean() * 100),
        "rate_rmse_pt": float(np.sqrt((err ** 2).mean()) * 100),
        "switches_mean": float(change.sum(axis=1).mean()),
        "single_slot_share": float(single.sum() / n_episodes),
        "part_mae_pt": float(np.abs(part_hat - part).mean() * 100),
        "n_unique": float(len(np.unique(X, axis=0))),
        "overfill_share": float(over / (n_people * tg.NUM_SLOTS)),
    }


def error_curve(X: npt.NDArray[np.int8], target: npt.NDArray[np.float64], step: int) -> list[dict[str, float]]:
    """最初の n 人だけで集計したときの時刻別行動者率の誤差を、n を step ずつ増やして並べる

    floor_mae_pt は、集計表どおりの分布から n 人を独立に引いたときの誤差の期待値（正規近似）。
    生成物の誤差がこれより小さければ、独立に引くよりも集計表に合っている。

    Args:
        X: (N, 96) 生成した順に並んだ活動スケジュール
        target: (20, 96) 集計表の時刻別行動者率
        step: n の刻み

    Returns:
        [{"n": n, "rate_mae_pt": ..., "floor_mae_pt": ...}, ...]
    """
    rows: list[dict[str, float]] = []
    for n in sorted(set(range(step, len(X) + 1, step)) | {len(X)}):
        floor = np.sqrt(2.0 * target * (1.0 - target) / (np.pi * n)).mean() * 100
        rows.append({"n": float(n), "rate_mae_pt": float(np.abs(slot_rates(X[:n]) - target).mean() * 100),
                     "floor_mae_pt": float(floor)})
    return rows


def main() -> None:
    """CLI"""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", type=int, nargs="+", default=[5, 0], help="生成する群の番号")
    ap.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    ap.add_argument("--N", type=int, default=100, help="群あたりの人数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-k", type=int, default=6, help="メモリに挙げる個数（不足・超過それぞれ）")
    ap.add_argument("--min-gap", type=float, default=0.03, help="メモリに挙げる差の下限（割合）")
    ap.add_argument("--warmup", type=int, default=5, help="この人数まではメモリを渡さない")
    ap.add_argument("--max-tries", type=int, default=3, help="形式が正しくないときにやり直す回数の上限")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080", help="OpenAI 互換のサーバーの URL")
    ap.add_argument("--model-tag", default="dry", help="保存名に入れるモデルの呼び名（例 gemma4-12b）")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--extra-json", default="{}", help='リクエストに足す項目の JSON（例 \'{"top_k": 64}\'）')
    ap.add_argument("--tag", default="", help="保存名の末尾に付ける印（試し回しの区別用）")
    ap.add_argument("--dry-run", action="store_true", help="サーバーにつながず、偽の応答で回す")
    args = ap.parse_args()

    stem = f"llm_{args.model_tag}_N{args.N}_seed{args.seed}" + (f"_{args.tag}" if args.tag else "")
    out_dir = OUT_ROOT / stem
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} に既に出力がある（取り違え防止）。--tag を変えるか、フォルダを消してから回す")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps({**vars(args), "system_prompt": SYSTEM_PROMPT}, indent=2,
                                                  ensure_ascii=False), encoding="utf-8")

    rates20, part20 = tg.load_rates20(), tg.load_participation20()
    summary: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    parts: list[dict[str, Any]] = []
    for d in args.groups:
        target = rates20[d]
        for condition in args.conditions:
            client: Chat = FakeChat(target) if args.dry_run else ServerChat(
                args.base_url, args.model_tag, args.temperature, args.top_p, args.max_tokens,
                json.loads(args.extra_json))
            t0 = time.time()
            X = generate_group(d, condition, args.N, target, client, args.seed, args.top_k, args.min_gap,
                               args.warmup, args.max_tries, out_dir / f"group{d:02d}_{condition}.jsonl")
            np.savez_compressed(out_dir / f"group{d:02d}_{condition}.npz", X=X)
            key = {"stem": stem, "group": d, "condition": condition}
            summary.append({**key, "n_people": args.N, **evaluate(X, target, part20[d]),
                            "minutes": (time.time() - t0) / 60})
            curves += [{**key, **row} for row in error_curve(X, target, step=10)]
            parts += [{**key, "activity": ACTIVITY_NAMES[k], "published": float(part20[d, k]),
                       "generated": float((X == k).any(axis=1).mean())} for k in range(tg.NUM_ACT20)]
            # 群と条件が 1 つ終わるごとに書く。途中で止まっても、終わった分の指標が残る
            pd.DataFrame(summary).to_csv(out_dir / "summary.csv", index=False)
            pd.DataFrame(curves).to_csv(out_dir / "curve.csv", index=False)
            pd.DataFrame(parts).to_csv(out_dir / "participation.csv", index=False)
    print(pd.DataFrame(summary).drop(columns=["stem"]).round(3).to_string(index=False), flush=True)
    print(f"[llm] 書いた: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
