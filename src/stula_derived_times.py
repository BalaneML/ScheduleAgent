"""
stula_derived_times.py
======================
社会生活基本調査の規則で「起床時刻」「就寝時刻」を生成物から決め、公表値（平均時刻編）と並べる

較正に使わない表（評価だけに残す表）として使う。規則の出所は wiki の
[[articles/stula-2016-terminology]] 3 節。

| 時刻 | 規則 |
|---|---|
| 起床 | 0 時以降 12 時前に始まり 60 分を超えて続く最初の睡眠の終了時刻 |
| 就寝 | 17 時以降 36 時（翌 12 時）前に始まり 60 分を超えて続く睡眠の開始時刻。2 つ以上なら継続時間が最長（同じなら早い方） |

- 睡眠と睡眠の間の非睡眠が 30 分以内なら、睡眠が続いているとみなす
- 就寝時刻は 0〜36 時の軸に載せる（翌 1:00 は 25.0）

写し元（DomainTransfer_trial、2026-11-18 に写した）:

    src/eval/stula_derived_times.py                     コミット 6403971  規則の本体（そのまま）
    src/eval/schedule_plausibility.py                   コミット 5c7db18  _circular_runs・slot_to_clock_min・
                                                                          in_window・_window_minutes（そのまま）
    src/models/DDPM_Aggregate_Simple/stage2_published.py コミット e2b51ef  load_mean_times（群の番号と
                                                                          CSV の置き場所だけ、このリポジトリの
                                                                          stula_targets に合わせた）

★写し元は共通 12 分類のスケジュールを前提にしているが、睡眠のラベルはどちらの分類でも 0 である
  （20 分類では睡眠、12 分類では睡眠＋身の回りの用事）。sleep_act の既定値 0 のまま両方に使える。
★規則を変えたときは、写し元と同じ値が返ることをテスト（test_llm_memory_trial.py の j）で確かめる。

```mermaid
flowchart TD
    X["X (N, 96)<br/>04:00 起点・円周"] --> B["bridge_short_gaps<br/>30 分以内の非睡眠を埋める"]
    B --> R["_circular_runs<br/>円周の run に分ける"]
    R --> E["sleep_episodes<br/>60 分を超える run だけ残す"]
    E --> W["wake_time<br/>終了が 0〜12 時の最初の run"]
    E --> BT["bed_time<br/>開始が 17〜36 時の最長の run"]
    W --> S["derived_time_summary<br/>wake_mean・bed_mean・不詳の割合"]
    BT --> S
    P["load_mean_times()<br/>公表値 (28,)"] --> C["比べる（schedule_metrics.holdout_errors）"]
    S --> C
```
"""
from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

import stula_targets as tg

IntArr = npt.NDArray[np.int64]
FloatArr = npt.NDArray[np.float64]
BoolArr = npt.NDArray[np.bool_]

NUM_SLOTS = 96
SLOT_MIN = 15
START_HOUR = 4
MINUTES_PER_DAY = 24 * 60
# 睡眠のラベル。20 分類では「睡眠」、共通 12 分類では「睡眠・身の回りの用事」で、どちらも 0
SLEEP_ACT = 0

# 60 分を「超えて」続くこと。15 分スロットなので 4 スロット超 = 5 スロット以上
MIN_SLEEP_MIN = 60
# 睡眠間の非睡眠がこの長さ以内なら睡眠が継続しているとみなす
MAX_GAP_MIN = 30
# 起床: 終了時刻がこの窓に入る睡眠（0:00-12:00）
WAKE_WINDOW = (0, 12 * 60)
# 就寝: 開始時刻がこの窓に入る睡眠（17:00-翌12:00 の 19 時間。日跨ぎ窓）
BED_WINDOW = (17 * 60, 12 * 60)
# 就寝時刻を載せる軸の折り返し。これより前の時刻は「翌日」として +24 する
BED_AXIS_WRAP_HOUR = 12

# 平均時刻編の軸の値。生活時間編（曜日 2_平日・地域 00_全国）と書き方が違う
MT_DAYTYPE_WEEKDAY = "1_平日"
MT_REGION_JAPAN = "0_全国"
LIFE_STAGE_TOTAL = "0_総数"
MEANTIME_TABLES = {"wake": "meantime_wake", "bed": "meantime_bed"}
# 平均時刻の妥当な範囲（時）。起床は 0〜12 時、就寝は 0〜36 時の軸に載る
MEANTIME_RANGE = {"wake": (0.0, 12.0), "bed": (17.0, 36.0)}
GENDER_CODE = {"1_男": 0, "2_女": 1}
EMPLOY_CODE = {"1_有業者": 1, "2_無業者": 0}


# ============================================================
# schedule_plausibility.py からの写し
# ============================================================
def _circular_runs(mask: BoolArr) -> tuple[IntArr, IntArr, IntArr]:
    """真偽マスクを円周上の連続 run に分解する

    各行の両端をゼロで挟んで線形の run に分け、行の先頭と末尾がどちらも True で run が 2 本以上
    ある行は、末尾の run に先頭の run を繋いで 1 本にする。

    Args:
        mask: 対象スロットの真偽値, dtype=bool, (N, NUM_SLOTS)

    Returns:
        (row, start, length)。いずれも dtype=int64, (R,)。row が行、start が開始スロット、
        length がスロット数。長さ 0 の run は含まない
    """
    n = mask.shape[0]
    padded = np.zeros((n, NUM_SLOTS + 2), dtype=np.int8)
    padded[:, 1:-1] = mask
    diff = np.diff(padded, axis=1)

    row, start = np.nonzero(diff == 1)
    _, end = np.nonzero(diff == -1)
    row = row.astype(np.int64)
    start = start.astype(np.int64)
    length = (end - start).astype(np.int64)

    first = np.searchsorted(row, np.arange(n), side="left")
    last = np.searchsorted(row, np.arange(n), side="right") - 1
    n_runs = last - first + 1
    # ★run が 1 本だけの行を除く。全スロットが True の行で先頭と末尾を繋ぐと同じ run を二重に数える
    merge = np.asarray(mask[:, 0] & mask[:, -1] & (n_runs >= 2))
    if merge.any():
        f, la = first[merge], last[merge]
        length[la] = length[la] + length[f]
        length[f] = 0

    keep = length > 0
    return row[keep], start[keep], length[keep]


def slot_to_clock_min(slot: IntArr | int) -> FloatArr:
    """スロット番号を時計の分（0 = 00:00）へ直す。04:00 起点を戻す

    Args:
        slot: スロット番号, 任意形状。範囲 [0, 96)

    Returns:
        時計の分, dtype=float64, 入力と同じ形状。範囲 [0, 1440)
    """
    return (np.asarray(slot, dtype=np.float64) * SLOT_MIN + START_HOUR * 60) % MINUTES_PER_DAY


def _window_minutes(window: tuple[int, int]) -> int:
    """時刻窓の長さ（分）。lo == hi は 1 日全体とみなす"""
    lo, hi = window
    return (hi - lo) % MINUTES_PER_DAY or MINUTES_PER_DAY


def in_window(clock_min: FloatArr, window: tuple[int, int]) -> BoolArr:
    """時刻が窓に入るか。日跨ぎの窓（22:00-08:00 など）も扱う。終端は含まない"""
    lo, _ = window
    offset = (np.asarray(clock_min, dtype=np.float64) - lo) % MINUTES_PER_DAY
    return np.asarray(offset < _window_minutes(window))


# ============================================================
# stula_derived_times.py からの写し（規則の本体）
# ============================================================
def bridge_short_gaps(sleep: BoolArr, max_gap_min: int = MAX_GAP_MIN) -> BoolArr:
    """睡眠と睡眠の間の短い非睡眠を睡眠で埋める（連結規則）。円周で埋める

    Args:
        sleep: 睡眠スロットの真偽値, dtype=bool, (N, NUM_SLOTS)
        max_gap_min: 埋める隙間の上限（分）

    Returns:
        隙間を埋めた睡眠マスク, dtype=bool, (N, NUM_SLOTS)。入力は変更しない
    """
    out = np.array(sleep, dtype=bool, copy=True)
    has_sleep = np.asarray(out.any(axis=1))
    row, start, length = _circular_runs(~out)
    sel = np.asarray((length * SLOT_MIN <= max_gap_min) & has_sleep[row])
    if not sel.any():
        return out
    r, s, ln = row[sel], start[sel], length[sel]
    offs = np.arange(int(ln.sum())) - np.repeat(np.cumsum(ln) - ln, ln)
    out[np.repeat(r, ln), (np.repeat(s, ln) + offs) % NUM_SLOTS] = True
    return out


def sleep_episodes(sched: IntArr, sleep_act: int = SLEEP_ACT, max_gap_min: int = MAX_GAP_MIN,
                   min_sleep_min: int = MIN_SLEEP_MIN) -> tuple[IntArr, IntArr, IntArr]:
    """規則を満たす睡眠エピソードを円周上で拾う

    終日睡眠（96 スロット全部）は落とす。円周では開始も終了も実在しないため、不詳が正しい。

    Args:
        sched: スケジュール, (N, NUM_SLOTS)
        sleep_act: 睡眠の活動ラベル
        max_gap_min: 連結する隙間の上限（分）
        min_sleep_min: 継続時間の下限（分）。この値を超えるものだけ残す

    Returns:
        (row, start, length), dtype=int64, (R,)
    """
    sleep = bridge_short_gaps(np.asarray(sched) == sleep_act, max_gap_min)
    row, start, length = _circular_runs(sleep)
    keep = length * SLOT_MIN > min_sleep_min
    keep &= length < NUM_SLOTS
    return row[keep], start[keep], length[keep]


def _first_per_row(row: IntArr, key: FloatArr, n: int) -> FloatArr:
    """行ごとに key が最小の値を返す, -> (n,)。該当が無い行は NaN"""
    out = np.full(n, np.inf)
    np.minimum.at(out, row, key)
    return np.where(np.isinf(out), np.nan, out)


def wake_time(sched: IntArr, **kw: Any) -> FloatArr:
    """起床時刻（0 時からの時間、[0,12)）, (N, 96) -> (N,)。不詳は NaN

    終了時刻が 0:00-12:00 に入る最初の睡眠の終了時刻。24 時間閉じの円周には日記の開始の
    境界が無いので、開始時刻でなく終了時刻で判定する。

    Args:
        sched: スケジュール, (N, NUM_SLOTS)
        **kw: sleep_episodes へ渡す（sleep_act / max_gap_min / min_sleep_min）

    Returns:
        起床時刻（時）, dtype=float64, (N,)
    """
    n = np.asarray(sched).shape[0]
    row, start, length = sleep_episodes(sched, **kw)
    end_min = slot_to_clock_min((start + length) % NUM_SLOTS)
    sel = in_window(end_min, WAKE_WINDOW)
    return _first_per_row(row[sel], end_min[sel] / 60.0, n)


def bed_time(sched: IntArr, **kw: Any) -> FloatArr:
    """就寝時刻（1 日目 0 時からの時間、[17,36)）, (N, 96) -> (N,)。不詳は NaN

    開始時刻が 17:00-翌12:00 に入る睡眠のうち、継続時間が最長のもの（同じなら早い方）の開始時刻。

    Args:
        sched: スケジュール, (N, NUM_SLOTS)
        **kw: sleep_episodes へ渡す（sleep_act / max_gap_min / min_sleep_min）

    Returns:
        就寝時刻（時）, dtype=float64, (N,)
    """
    n = np.asarray(sched).shape[0]
    row, start, length = sleep_episodes(sched, **kw)
    start_min = slot_to_clock_min(start)
    sel = in_window(start_min, BED_WINDOW)
    row, start_min, length = row[sel], start_min[sel], length[sel]
    hour = start_min / 60.0
    hour36 = np.where(hour < BED_AXIS_WRAP_HOUR, hour + 24.0, hour)

    out = np.full(n, np.nan)
    if len(row) == 0:
        return out
    order = np.lexsort((hour36, -length, row))
    r_sorted = row[order]
    _, first = np.unique(r_sorted, return_index=True)
    out[r_sorted[first]] = hour36[order][first]
    return out


def derived_time_summary(sched: IntArr, w: FloatArr | None = None) -> dict[str, float]:
    """起床・就寝の平均・標準偏差・不詳の割合

    不詳（NaN）は平均から除外し、その割合を別に返す。不詳率そのものも指標で、生成物に
    60 分を超える夜間睡眠の無い人が多ければ、平均が合っていても睡眠の作りが壊れている。

    Args:
        sched: スケジュール, (N, NUM_SLOTS)。睡眠のラベルは 0
        w: 人ごとの重み, (N,)。None なら一様

    Returns:
        wake_mean / wake_sd / wake_undef_rate / bed_mean / bed_sd / bed_undef_rate（時。就寝は 0〜36 時の軸）
    """
    n = np.asarray(sched).shape[0]
    wn = np.ones(n) / n if w is None else np.asarray(w, dtype=np.float64) / np.sum(w)
    out: dict[str, float] = {}
    for name, fn in (("wake", wake_time), ("bed", bed_time)):
        x = fn(sched)
        ok = ~np.isnan(x)
        ws = wn[ok].sum()
        if ws <= 0:
            out[f"{name}_mean"] = float("nan")
            out[f"{name}_sd"] = float("nan")
        else:
            mean = float((wn[ok] * x[ok]).sum() / ws)
            var = float((wn[ok] * (x[ok] - mean) ** 2).sum() / ws)
            out[f"{name}_mean"] = mean
            out[f"{name}_sd"] = float(np.sqrt(max(var, 0.0)))
        out[f"{name}_undef_rate"] = float(wn[~ok].sum())
    return out


# ============================================================
# stage2_published.py からの写し（公表値の読み込み）
# ============================================================
def load_mean_times(daytype: str = MT_DAYTYPE_WEEKDAY) -> dict[str, Any]:
    """平均時刻編の起床・就寝を 28 群へ読む

    年齢 5 歳 15 区分を 7 区分へ畳む重みは行動者数（推定人口 × 行動者率）。公表の平均時刻は
    「時刻が定まった人」の平均なので、人口でなく行動者数で束ねる。ライフステージは 0_総数 を使う。

    Args:
        daytype: 曜日の値。平均時刻編では平日が "1_平日"（生活時間編の "2_平日" とは違う）

    Returns:
        wake_hour / bed_hour (28,) 平均時刻（時、就寝は 0〜36 時の軸）、
        wake_actor_rate / bed_actor_rate (28,) 時刻が定まった割合 [0,1]、pop (28,) 推定人口（千人）、daytype

    Raises:
        FileNotFoundError: CSV が無いとき
        ValueError: 60 層（2 性 × 2 就業 × 5 歳 15 区分）がそろわないとき、平均時刻が想定の範囲を外れるとき
    """
    out: dict[str, Any] = {"daytype": daytype}
    pop_out = np.zeros(tg.D_GROUPS)
    for kind, name in MEANTIME_TABLES.items():
        path = tg.stula_dir() / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} が無い")
        df = pd.read_csv(path)
        sub = cast(pd.DataFrame, df[(df["daytype"] == daytype)
                                   & (df["region"] == MT_REGION_JAPAN)
                                   & (df["life_stage"] == LIFE_STAGE_TOTAL)
                                   & (df["gender"].isin(list(GENDER_CODE)))
                                   & (df["employment"].isin(list(EMPLOY_CODE)))]).copy()
        sub["a15"] = cast(pd.Series, sub["age_class"]).str[:2].map(
            lambda c: int(c) if str(c).isdigit() and str(c) != "00" else np.nan)
        sub = cast(pd.DataFrame, sub.dropna(subset=["a15"]))
        n_need = tg.N_G * tg.N_E * 15
        if len(sub) != n_need:
            raise ValueError(f"{name}: 層が {len(sub)} 行で {n_need} 行に足りない（daytype={daytype}）")

        g_i = cast(pd.Series, sub["gender"]).map(GENDER_CODE).to_numpy(dtype=np.int64)
        e_i = cast(pd.Series, sub["employment"]).map(EMPLOY_CODE).to_numpy(dtype=np.int64)
        a7_i = np.array([tg.AGE15_TO_7[int(a)] for a in sub["a15"].to_numpy()], dtype=np.int64)
        d_i = np.array([tg.group_index(int(g), int(a), int(e)) for g, a, e in zip(g_i, a7_i, e_i)],
                       dtype=np.int64)

        pop15 = sub[["population_k"]].to_numpy(dtype=np.float64).ravel()
        rate15 = sub[["actor_rate"]].to_numpy(dtype=np.float64).ravel() / 100.0
        mean15 = sub[["mean_time_hour"]].to_numpy(dtype=np.float64).ravel()
        actors = pop15 * rate15
        # 行動者数で加重して 7 区分へ畳む。非公表（NaN）の層は分子・分母の両方から外す
        pub = ~np.isnan(mean15)
        num = np.zeros(tg.D_GROUPS)
        den = np.zeros(tg.D_GROUPS)
        pop_sum = np.zeros(tg.D_GROUPS)
        act_sum = np.zeros(tg.D_GROUPS)
        np.add.at(num, d_i, np.where(pub, actors * mean15, 0.0))
        np.add.at(den, d_i, np.where(pub, actors, 0.0))
        np.add.at(pop_sum, d_i, pop15)
        np.add.at(act_sum, d_i, actors)
        with np.errstate(invalid="ignore", divide="ignore"):
            hour = np.where(den > 0, num / den, np.nan)
            out[f"{kind}_actor_rate"] = np.where(pop_sum > 0, act_sum / pop_sum, np.nan)
        lo, hi = MEANTIME_RANGE[kind]
        bad = ~np.isnan(hour) & ((hour < lo) | (hour >= hi))
        if bad.any():
            raise ValueError(f"{name}: 畳んだ平均時刻が {lo}〜{hi} 時の外に {int(bad.sum())} 群ある")
        out[f"{kind}_hour"] = hour
        pop_out = pop_sum
    out["pop"] = pop_out
    return out
