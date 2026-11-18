"""
schedule_metrics.py
===================
生成した活動スケジュールの採点。LLM・SA・DGM（2 段階学習）の生成物を、この 1 つのコードで採点する

指標の分け方（2026-11-18 の決定）:

| 分け方 | 指標 | 関数 |
|---|---|---|
| 較正に使う表 | 時刻別行動者率の MAE、日次行動者率の MAE | evaluate / evaluate_common |
| 評価だけに残す表 | 平均起床・就寝時刻の差、不詳の割合の差 | holdout_metrics |
| 系列 | 切替回数、1 スロットで終わる区間の割合、異なる 1 日の割合、人どうしの距離 | sequence_metrics |
| ルール（評価だけ） | R1〜R4 の違反数（20 分類だけ） | rule_metrics |
| メモリの条件 | 「この人の条件」に従った割合 | follow_rates |

20 分類（LLM・SA）はそのまま、共通 12 分類（DGM）と比べるときは STULA20_TO_COMMON で 12 分類へまとめる。
12 分類の「その他」（番号 11）は ATUS と対応が取れないので、時刻別行動者率の MAE から外す。
12 分類の日次行動者率は、20 分類と 1 対 1 で対応する 7 活動だけで比べる（まとめた活動の日次行動者率は
20 分類の値の和にならないため）。

```mermaid
flowchart LR
    X["X (N, 96)<br/>20 分類"] --> EV["evaluate(X, target, part)"]
    X --> SQ["sequence_metrics(X)"]
    X --> RU["rule_metrics(X)"]
    X --> HO["holdout_metrics(X, d, mean_times)"]
    X --> TC["to_common(X)<br/>X12 (N, 96)"]
    TC --> EC["evaluate_common(X12, target12, part20)"]
    DGM["DGM の pool (N, 96)<br/>12 分類"] --> EC
```
"""
from typing import Any

import numpy as np
import numpy.typing as npt

import schedule_rules as sr
import stula_derived_times as sdt
import stula_targets as tg

SLOTS_PER_HOUR = 4
START_HOUR = 4
OTHER_COMMON = 11                      # 12 分類の「その他」。時刻別行動者率の MAE から外す
# 12 分類のうち、20 分類のちょうど 1 活動と対応するもの（日次行動者率を厳密に比べられる）
EXACT_COMMON: dict[int, int] = {
    int(c): int(np.flatnonzero(tg.STULA20_TO_COMMON == c)[0])
    for c in range(tg.NUM_COMMON)
    if c != OTHER_COMMON and int((tg.STULA20_TO_COMMON == c).sum()) == 1
}


def slot_rates(X: npt.NDArray[np.integer], n_act: int = tg.NUM_ACT20) -> npt.NDArray[np.float64]:
    """生成した活動スケジュール (N, 96) から時刻別行動者率 (n_act, 96) を集計する"""
    X = np.asarray(X)
    rates = np.zeros((n_act, X.shape[1]))
    for k in range(n_act):
        rates[k] = (X == k).mean(axis=0)
    return rates


def participation(X: npt.NDArray[np.integer], n_act: int = tg.NUM_ACT20) -> npt.NDArray[np.float64]:
    """日次行動者率 (n_act,)：その活動を 1 日に 1 スロット以上した人の割合"""
    X = np.asarray(X)
    return np.array([(X == k).any(axis=1).mean() for k in range(n_act)])


def floor_mae_pt(target: npt.NDArray[np.float64], n_people: int) -> float:
    """床：集計表どおりの分布から n 人を独立に引いたときの時刻別 MAE の期待値（正規近似、ポイント）"""
    return float(np.sqrt(2.0 * target * (1.0 - target) / (np.pi * n_people)).mean() * 100)


def evaluate(X: npt.NDArray[np.int8], target: npt.NDArray[np.float64],
             part: npt.NDArray[np.float64]) -> dict[str, float]:
    """1 群ぶんの生成物を、較正に使う 2 表と基本の系列の指標で採点する（20 分類）

    Args:
        X: (N, 96) 生成した活動スケジュール
        target: (20, 96) 集計表の時刻別行動者率
        part: (20,) 日次行動者率の公表値

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
    part_hat = participation(X)
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

    Args:
        X: (N, 96) 生成した順に並んだ活動スケジュール
        target: (20, 96) 集計表の時刻別行動者率
        step: n の刻み

    Returns:
        [{"n": n, "rate_mae_pt": ..., "floor_mae_pt": ...}, ...]
    """
    rows: list[dict[str, float]] = []
    for n in sorted(set(range(step, len(X) + 1, step)) | {len(X)}):
        rows.append({"n": float(n), "rate_mae_pt": float(np.abs(slot_rates(X[:n]) - target).mean() * 100),
                     "floor_mae_pt": floor_mae_pt(target, n)})
    return rows


def pairwise_distance(X: npt.NDArray[np.integer], max_people: int = 400, seed: int = 0) -> float:
    """人どうしの距離：2 人を選んだとき、活動が違うスロットの割合の平均（多様さ）

    人数が max_people を超えるときは、種 seed で max_people 人を選んで計算する。

    Args:
        X: (N, 96) 活動スケジュール
        max_people: 計算に使う人数の上限
        seed: 人を選ぶ種

    Returns:
        0〜1。全員が同じ 1 日なら 0
    """
    X = np.asarray(X)
    if len(X) > max_people:
        X = X[np.random.default_rng(seed).choice(len(X), size=max_people, replace=False)]
    n = len(X)
    if n < 2:
        return float("nan")
    diff = (X[:, None, :] != X[None, :, :]).mean(axis=2)
    return float(diff[np.triu_indices(n, k=1)].mean())


def sequence_metrics(X: npt.NDArray[np.integer]) -> dict[str, float]:
    """系列の指標。分類の数（20・12）によらず使える

    Args:
        X: (N, 96) 活動スケジュール

    Returns:
        switches_mean（1 人あたりの切替回数）、single_slot_share（1 スロットで終わる区間の割合）、
        distinct_share（異なる 1 日の割合）、pairwise_distance（人どうしの距離）
    """
    X = np.asarray(X)
    n_people = len(X)
    change = X[:, 1:] != X[:, :-1]
    edge = np.concatenate([np.ones((n_people, 1), dtype=bool), change, np.ones((n_people, 1), dtype=bool)], axis=1)
    single = edge[:, :-1] & edge[:, 1:]
    return {
        "switches_mean": float(change.sum(axis=1).mean()),
        "single_slot_share": float(single.sum() / (change.sum() + n_people)),
        "distinct_share": float(len(np.unique(X, axis=0)) / n_people),
        "pairwise_distance": pairwise_distance(X),
    }


def rule_metrics(X: npt.NDArray[np.integer]) -> dict[str, float]:
    """R1〜R4 の違反（20 分類だけ）。ルールは LLM に見せず、評価にだけ使う

    Args:
        X: (N, 96) 20 分類の活動スケジュール

    Returns:
        violations_mean（1 人あたりの違反数）、violation_share（違反が 1 つ以上ある人の割合）、
        ルールごとの上限を超えた人の割合（over_R1_sleep など）
    """
    groups = sr.rule_groups(X)
    over = np.maximum(groups - sr.RULE_CAP[None, :], 0)
    v = over.sum(axis=1)
    out = {"violations_mean": float(v.mean()), "violation_share": float((v > 0).mean())}
    for r, name in enumerate(sr.RULE_NAMES):
        out[f"over_{name}"] = float((over[:, r] > 0).mean())
    return out


def holdout_metrics(X: npt.NDArray[np.integer], d: int, mean_times: dict[str, Any]) -> dict[str, float]:
    """評価だけに残す表（平均起床・就寝時刻）との差。睡眠のラベルは 0

    20 分類では睡眠そのもの、12 分類では「睡眠・身の回りの用事」で判定することになる。

    Args:
        X: (N, 96) 活動スケジュール
        d: 群の番号
        mean_times: stula_derived_times.load_mean_times() の戻り値

    Returns:
        wake_hour・bed_hour（生成物の平均、時）、wake_err_h・bed_err_h（生成 − 公表、時）、
        wake_undef_gap・bed_undef_gap（不詳の割合の差、生成 − 公表）
    """
    s = sdt.derived_time_summary(np.asarray(X, dtype=np.int64))
    out: dict[str, float] = {}
    for kind in ("wake", "bed"):
        out[f"{kind}_hour"] = s[f"{kind}_mean"]
        out[f"{kind}_err_h"] = s[f"{kind}_mean"] - float(mean_times[f"{kind}_hour"][d])
        out[f"{kind}_undef_gap"] = s[f"{kind}_undef_rate"] - (1.0 - float(mean_times[f"{kind}_actor_rate"][d]))
    return out


def to_common(X: npt.NDArray[np.integer]) -> npt.NDArray[np.int64]:
    """20 分類のスケジュールを共通 12 分類へまとめる"""
    return tg.STULA20_TO_COMMON[np.asarray(X, dtype=np.int64)]


def rates_to_common(rates20: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """20 分類の時刻別行動者率 (20, 96) を 12 分類 (12, 96) へまとめる（同じスロットの割合は足せる）"""
    out = np.zeros((tg.NUM_COMMON, rates20.shape[1]))
    np.add.at(out, tg.STULA20_TO_COMMON, rates20)
    return out


def evaluate_common(X12: npt.NDArray[np.integer], target12: npt.NDArray[np.float64],
                    part20: npt.NDArray[np.float64]) -> dict[str, float]:
    """12 分類で採点する（DGM と比べるとき）

    Args:
        X12: (N, 96) 12 分類の活動スケジュール
        target12: (12, 96) 12 分類の時刻別行動者率（rates_to_common で作る）
        part20: (20,) 20 分類の日次行動者率の公表値

    Returns:
        rate_mae_pt（「その他」を除く 11 分類）、floor_mae_pt（同じ人数の床）、
        part_mae_pt（1 対 1 で対応する 7 活動の日次行動者率）
    """
    keep = [c for c in range(tg.NUM_COMMON) if c != OTHER_COMMON]
    err = slot_rates(X12, tg.NUM_COMMON)[keep] - target12[keep]
    part_hat = participation(X12, tg.NUM_COMMON)
    exact_c = list(EXACT_COMMON)
    exact_k = [EXACT_COMMON[c] for c in exact_c]
    return {
        "rate_mae_pt": float(np.abs(err).mean() * 100),
        "floor_mae_pt": floor_mae_pt(target12[keep], len(X12)),
        "part_mae_pt": float(np.abs(part_hat[exact_c] - part20[exact_k]).mean() * 100),
    }


def hour_slots(hour: int) -> slice:
    """時計の時（0〜23）を、04:00 起点の 96 スロットの 4 スロットに直す"""
    s0 = ((hour - START_HOUR) % 24) * SLOTS_PER_HOUR
    return slice(s0, s0 + SLOTS_PER_HOUR)


def follow_rates(X: npt.NDArray[np.integer], conditions: list[list[dict[str, Any]]]) -> dict[str, float]:
    """「この人の条件」に従った割合と、条件を付けなかった人での同じ割合（比べる基準）

    - not（k をしない）: 従った = その人の 1 日に k が無い。基準 = k について条件を付けなかった人で k が無い割合
    - do（k をする）: 従った = k が 1 スロット以上ある。基準 = k について条件を付けなかった人で k がある割合
    - swap（h 時台は A、B ではない）: 従った = h 時台の 4 スロットのうち A が 2 つ以上、B が 0

    Args:
        X: (N, 96) 生成した活動スケジュール
        conditions: 人ごとの条件の並び。条件は {"kind", "activity", ...}。swap は "hour"・"instead_of" を持つ

    Returns:
        follow_not・base_not・n_not、follow_do・base_do・n_do、follow_swap・n_swap。該当が無ければ NaN
    """
    X = np.asarray(X)
    present = np.stack([(X == k).any(axis=1) for k in range(tg.NUM_ACT20)], axis=1)   # (N, 20)
    told = np.zeros_like(present)
    for i, conds in enumerate(conditions):
        for c in conds:
            if c["kind"] in ("not", "do"):
                told[i, c["activity"]] = True
    hits: dict[str, list[float]] = {"not": [], "do": [], "swap": []}
    base: dict[str, list[float]] = {"not": [], "do": []}
    for i, conds in enumerate(conditions):
        for c in conds:
            k = int(c["activity"])
            free = ~told[:, k]
            if c["kind"] == "not":
                hits["not"].append(float(not present[i, k]))
                base["not"].append(float((~present[free, k]).mean()) if free.any() else np.nan)
            elif c["kind"] == "do":
                hits["do"].append(float(present[i, k]))
                base["do"].append(float(present[free, k].mean()) if free.any() else np.nan)
            elif c["kind"] == "swap":
                block = X[i, hour_slots(int(c["hour"]))]
                hits["swap"].append(float((block == k).sum() >= 2 and not (block == int(c["instead_of"])).any()))
    out: dict[str, float] = {}
    for kind in ("not", "do", "swap"):
        out[f"follow_{kind}"] = float(np.mean(hits[kind])) if hits[kind] else float("nan")
        out[f"n_{kind}"] = float(len(hits[kind]))
        if kind in base:
            vals = [b for b in base[kind] if not np.isnan(b)]
            out[f"base_{kind}"] = float(np.mean(vals)) if vals else float("nan")
    return out
