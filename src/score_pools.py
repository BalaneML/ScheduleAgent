"""
score_pools.py
==============
LLM・SA・DGM（2 段階学習）の生成物を、schedule_metrics の同じ指標で採点して 1 つの CSV に書く

    SCHEDULE_AGENT_STULA_DIR=... .venv/bin/python src/score_pools.py \\
        --llm outputs/llm_memory_trial/llm_gemma4-12b_N100_seed42 \\
        --sa ../DomainTransfer_trial/outputs/generated/sa_aggregate/sa_rules_K100000_N100_seed42_pool.npz \\
        --groups 5 0 --out outputs/score_pools/scores.csv

入力

    --llm DIR   llm_memory_trial の出力フォルダ。group{d:02d}_{condition}.npz（X (N, 96)、20 分類）をすべて読む
    --sa NPZ    SA の pool。pool20 (28, N, 96)、20 分類
    --dgm NPZ   DGM の pool。pool (28, M, 96)、共通 12 分類（比べる npz はユーザーが指定する）

出力（1 行 = 生成物 × 群 × 分類）

    resolution = act20     LLM・SA だけ。較正に使う 2 表、系列、R1〜R4 の違反、起床・就寝（睡眠そのもので判定）
    resolution = common12  すべて。12 分類にまとめた較正の 2 表、系列、起床・就寝（睡眠・身の回りの用事で判定）

```mermaid
flowchart LR
    L["--llm DIR<br/>group*_*.npz"] --> P["Pool(kind, label, d, X, n_act)"]
    S["--sa NPZ<br/>pool20"] --> P
    G["--dgm NPZ<br/>pool（12 分類）"] --> P
    P --> R20["score_act20(pool)<br/>20 分類の行"]
    P --> R12["score_common12(pool)<br/>12 分類の行"]
    R20 --> CSV["--out CSV"]
    R12 --> CSV
```
"""
import argparse
import sys
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import schedule_metrics as sm  # noqa: E402
import stula_derived_times as sdt  # noqa: E402
import stula_targets as tg  # noqa: E402


class Pool(NamedTuple):
    """1 群ぶんの生成物

    Attributes:
        kind: "llm" / "sa" / "dgm"
        label: 生成物の名前（フォルダ名や npz の名前と条件）
        d: 群の番号
        X: (N, 96) 活動スケジュール
        n_act: 分類の数（20 か 12）
    """
    kind: str
    label: str
    d: int
    X: npt.NDArray[np.integer]
    n_act: int


def read_llm(out_dir: Path, groups: list[int]) -> list[Pool]:
    """llm_memory_trial の出力フォルダから、指定した群の npz を読む"""
    pools: list[Pool] = []
    for path in sorted(out_dir.glob("group*_*.npz")):
        d, condition = int(path.stem[5:7]), path.stem[8:]
        if d in groups:
            pools.append(Pool("llm", f"{out_dir.name}:{condition}", d, np.load(path)["X"], tg.NUM_ACT20))
    return pools


def read_pool(path: Path, kind: str, key: str, n_act: int, groups: list[int]) -> list[Pool]:
    """SA・DGM の pool（28 群 × 人 × 96）から、指定した群を取り出す"""
    arr = np.load(path)[key]
    if arr.shape[0] != tg.D_GROUPS or arr.shape[2] != tg.NUM_SLOTS:
        raise ValueError(f"{path}: {key} の形が (28, N, 96) でない（{arr.shape}）")
    return [Pool(kind, path.stem, d, arr[d], n_act) for d in groups]


def score_act20(pool: Pool, rates20: npt.NDArray[np.float64], part20: npt.NDArray[np.float64],
                mean_times: dict[str, Any]) -> dict[str, Any]:
    """20 分類の行。較正の 2 表・系列・R1〜R4・起床就寝"""
    X, d = np.asarray(pool.X), pool.d
    return {"kind": pool.kind, "label": pool.label, "group": d, "resolution": "act20", "n_people": len(X),
            "floor_mae_pt": sm.floor_mae_pt(rates20[d], len(X)), **sm.evaluate(X, rates20[d], part20[d]),
            **sm.sequence_metrics(X), **sm.rule_metrics(X), **sm.holdout_metrics(X, d, mean_times)}


def score_common12(pool: Pool, rates20: npt.NDArray[np.float64], part20: npt.NDArray[np.float64],
                   mean_times: dict[str, Any]) -> dict[str, Any]:
    """12 分類の行。20 分類の生成物は 12 分類にまとめてから採点する"""
    X12 = sm.to_common(pool.X) if pool.n_act == tg.NUM_ACT20 else np.asarray(pool.X, dtype=np.int64)
    d = pool.d
    return {"kind": pool.kind, "label": pool.label, "group": d, "resolution": "common12", "n_people": len(X12),
            **sm.evaluate_common(X12, sm.rates_to_common(rates20[d]), part20[d]),
            **sm.sequence_metrics(X12), **sm.holdout_metrics(X12, d, mean_times)}


def score(pools: list[Pool]) -> pd.DataFrame:
    """すべての生成物を採点し、1 行 = 生成物 × 群 × 分類 の表にする"""
    rates20, part20, mean_times = tg.load_rates20(), tg.load_participation20(), sdt.load_mean_times()
    rows: list[dict[str, Any]] = []
    for pool in pools:
        if pool.n_act == tg.NUM_ACT20:
            rows.append(score_act20(pool, rates20, part20, mean_times))
        rows.append(score_common12(pool, rates20, part20, mean_times))
    return pd.DataFrame(rows)


def main() -> None:
    """CLI"""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", type=Path, nargs="*", default=[], help="llm_memory_trial の出力フォルダ")
    ap.add_argument("--sa", type=Path, nargs="*", default=[], help="SA の pool の npz（pool20）")
    ap.add_argument("--dgm", type=Path, nargs="*", default=[], help="DGM の pool の npz（pool、12 分類）")
    ap.add_argument("--groups", type=int, nargs="+", default=[5, 0], help="採点する群の番号")
    ap.add_argument("--out", type=Path, default=tg.REPO_ROOT / "outputs" / "score_pools" / "scores.csv")
    args = ap.parse_args()

    pools: list[Pool] = []
    for out_dir in args.llm:
        pools += read_llm(out_dir, args.groups)
    for path in args.sa:
        pools += read_pool(path, "sa", "pool20", tg.NUM_ACT20, args.groups)
    for path in args.dgm:
        pools += read_pool(path, "dgm", "pool", tg.NUM_COMMON, args.groups)
    if not pools:
        raise SystemExit("採点する生成物が無い。--llm / --sa / --dgm のどれかを指定する")
    table = score(pools)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    cols = ["label", "group", "resolution", "n_people", "rate_mae_pt", "floor_mae_pt", "part_mae_pt", "switches_mean",
            "single_slot_share", "violations_mean", "pairwise_distance", "wake_err_h", "bed_err_h"]
    print(table[[c for c in cols if c in table.columns]].round(3).to_string(index=False), flush=True)
    print(f"[score] 書いた: {args.out}", flush=True)


if __name__ == "__main__":
    main()
