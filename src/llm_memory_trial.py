"""
llm_memory_trial.py
===================
LLM が 1 人ずつ活動スケジュールを書き、生成済みの集計と集計表の差をメモリとして次の人に渡す試行

書いた活動スケジュールは確定で、後から直さない（足すだけ）。メモリは 1 人足すごとに作り直す。
LLM には OpenAI 互換の API（llama.cpp の llama-server など）でつなぐ。プロンプトは英語。

    条件 none           メモリなし。毎回同じプロンプトで 1 人ぶん書く
    条件 memory         時刻別行動者率の差が大きい箇所を、不足・超過それぞれ上位 top_k 個だけ文にして渡す（11/17 の版）
    条件 memory_day     日次行動者率の差（不足・超過それぞれ上位 top_k 個）と、差が大きい時間帯の
                        「多すぎる活動・足りない活動」の組（上位 top_k 時間）を渡す。集団の数字だけ
    条件 memory_person  memory_day に「この人の条件」を足す。日次の条件は最大 person_lines 行、時刻の組は 1 行

「この人の条件」の引き方（person_conditions）: 活動 k ごとに、残りの人数で調査の値に届くのに要る実施率
q_k = (t_k·N − H_k)/(N − n) を出し、条件を付けなかった人での実施率 b_k と比べる。q_k < b_k なら確率
1 − q_k/b_k で「k をしない」、q_k > b_k なら確率 (q_k − b_k)/(1 − b_k) で「k をする」と書く。

    # サーバーなしの空回し（形式とループの確認）
    .venv/bin/python src/llm_memory_trial.py --dry-run --groups 5 --N 10 --tag dry
    # サーバーにつなぐ
    .venv/bin/python src/llm_memory_trial.py --model-tag gemma4-12b --groups 5 0 --N 100 \\
        --conditions memory_day memory_person --tag v2

出力（outputs/llm_memory_trial/{stem}/）

    group{d:02d}_{condition}.npz     X (N, 96) int8。20 分類・04:00 起点（DomainTransfer_trial の SA の best_X と同じ並び）
    group{d:02d}_{condition}.jsonl   1 人 1 行。プロンプト・応答・所要時間・この人の条件（conditions）
    summary.csv                      群 × 条件ごとの指標（較正に使う 2 表、系列、条件に従った割合）
    curve.csv                        n 人目までの時刻別行動者率の誤差
    participation.csv                日次行動者率（公表値と生成物）

評価だけに残す表（平均起床・就寝時刻）と R1〜R4 の違反は、src/score_pools.py でまとめて採点する。

```mermaid
flowchart LR
    T["tg.load_rates20()[d]<br/>target (20, 96)"] --> M1["memory_text(X_done, target, top_k, min_gap)<br/>条件 memory"]
    PART["tg.load_participation20()[d]<br/>part (20,)"] --> MD["memory_day_text(X_done, target, part, top_k, min_gap)<br/>条件 memory_day・memory_person"]
    T --> MD
    PART --> PC["person_conditions(X_done, target, part, n_people, told, u, person_lines, min_gap)<br/>conds"]
    PC --> PT["person_text(conds)<br/>条件 memory_person"]
    M1 --> P["build_messages(d, memory)"]
    MD --> P
    PT --> P
    P --> L["client.chat(messages, seed)<br/>content"]
    L --> S["segments_to_slots(segments)<br/>row (96,)"]
    S --> X["X_done に row を足す（確定）"]
    X --> M1
    X --> MD
    X --> PC
    X --> E["evaluate(X, target, part)<br/>follow_rates(X, conditions)"]
```

★較正に使うのは時刻別行動者率と日次行動者率（第70-3表）。条件 memory だけは 11/17 の版のまま、時刻別だけを見る。
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, NamedTuple, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stula_targets as tg  # noqa: E402
from schedule_metrics import error_curve, evaluate, follow_rates, participation, slot_rates  # noqa: E402

OUT_ROOT = tg.REPO_ROOT / "outputs" / "llm_memory_trial"

SLOT_MIN = 15                 # 1 スロットの長さ（分）
DAY_MIN = 24 * 60
ORIGIN_MIN = 4 * 60           # スロット 0 は 04:00〜04:15（parse_timeband.py の回転と同じ）
SLOTS_PER_HOUR = 60 // SLOT_MIN
CONDITIONS = ("none", "memory", "memory_day", "memory_person")

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
CLOSING_TEXT = ["Choose this person's day so that the group as a whole moves closer to the survey statistics,",
                "while the day stays realistic for one person."]


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


class SwapPair(NamedTuple):
    """ある時間帯の「多すぎる活動」と「足りない活動」の組

    Attributes:
        hour: 04:00 起点の時間の番号（0 が 04:00〜05:00）
        over: 多すぎる活動の番号
        under: 足りない活動の番号
        size: 組の差の大きさ。多すぎる側と足りない側の差の小さい方（割合）
    """
    hour: int
    over: int
    under: int
    size: float


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


def hour_label(hour: int) -> str:
    """04:00 起点の時間の番号を「HH:MM-HH:MM」にする"""
    return f"{clock(hour * 60)}-{clock((hour + 1) * 60)}"


def clock_hour(hour: int) -> int:
    """04:00 起点の時間の番号を、時計の時（0〜23）にする"""
    return (ORIGIN_MIN // 60 + hour) % 24


def memory_text(X_done: npt.NDArray[np.int8], target: npt.NDArray[np.float64], top_k: int, min_gap: float) -> str:
    """条件 memory のメモリ（11/17 の版）。時刻別行動者率の差が大きい箇所を、不足・超過それぞれ上位 top_k 個

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
            out.append(f"- {ACTIVITY_NAMES[k]}, {hour_label(h)}: "
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
    text += CLOSING_TEXT
    return "\n".join(text)


def hour_swap_pairs(X_done: npt.NDArray[np.int8], target: npt.NDArray[np.float64], top_k: int,
                    min_gap: float) -> list[SwapPair]:
    """時間帯ごとに「最も多すぎる活動」と「最も足りない活動」の組を作り、組の差が大きい順に top_k 個

    組の差は、多すぎる側の差と足りない側の差の小さい方。1 人がその時間帯の活動を入れ替えると、
    両方の差が同時に縮む。

    Args:
        X_done: (n, 96) これまでに確定した活動スケジュール
        target: (20, 96) 集計表の時刻別行動者率
        top_k: 挙げる組の数の上限
        min_gap: 組の差がこれより小さい時間帯は挙げない（割合）

    Returns:
        SwapPair の並び（組の差が大きい順）
    """
    gap = hourly_mean(slot_rates(X_done)) - hourly_mean(target)     # (20, 24)
    hours = np.arange(24)
    over, under = gap.argmax(axis=0), gap.argmin(axis=0)
    size = np.minimum(gap[over, hours], -gap[under, hours])
    pairs: list[SwapPair] = []
    for h in np.argsort(-size, kind="stable")[:top_k]:
        if size[h] < min_gap:
            break
        pairs.append(SwapPair(int(h), int(over[h]), int(under[h]), float(size[h])))
    return pairs


def memory_day_text(X_done: npt.NDArray[np.int8], target: npt.NDArray[np.float64], part: npt.NDArray[np.float64],
                    top_k: int, min_gap: float) -> str:
    """条件 memory_day のメモリ。日次行動者率の差と、時間帯ごとの「多すぎる・足りない」の組。集団の数字だけ

    Args:
        X_done: (n, 96) これまでに確定した活動スケジュール
        target: (20, 96) 集計表の時刻別行動者率
        part: (20,) 日次行動者率の公表値
        top_k: 日次は不足・超過それぞれ、時間帯は組の数で、挙げる個数の上限
        min_gap: これより差の小さい箇所は挙げない（割合）

    Returns:
        英語の文章。挙げる箇所が 1 つも無いときは空文字
    """
    have = participation(X_done)
    gap = have - part
    order = np.argsort(-np.abs(gap), kind="stable")
    many = [int(k) for k in order if gap[k] >= min_gap][:top_k]
    few = [int(k) for k in order if -gap[k] >= min_gap][:top_k]
    pairs = hour_swap_pairs(X_done, target, top_k, min_gap)
    if not many and not few and not pairs:
        return ""
    want_h, have_h = hourly_mean(target), hourly_mean(slot_rates(X_done))
    text = [f"MEMORY. {len(X_done)} diaries have been written for this group so far. Compared with the",
            "official survey statistics:"]
    if many or few:
        text.append("Share of people who do the activity at some time in the day:")
        if many:
            text += ["Too MANY people so far are doing:"]
            text += [f"- {ACTIVITY_NAMES[k]}: survey {part[k]:.0%}, diaries so far {have[k]:.0%}" for k in many]
        if few:
            text += ["Too FEW people so far are doing:"]
            text += [f"- {ACTIVITY_NAMES[k]}: survey {part[k]:.0%}, diaries so far {have[k]:.0%}" for k in few]
    if pairs:
        text.append("Hours with the largest differences (share of people doing the activity in that hour):")
        for p in pairs:
            text.append(f"- {hour_label(p.hour)}: too many doing {ACTIVITY_NAMES[p.over]} "
                        f"(survey {want_h[p.over, p.hour]:.0%}, diaries so far {have_h[p.over, p.hour]:.0%}), "
                        f"too few doing {ACTIVITY_NAMES[p.under]} "
                        f"(survey {want_h[p.under, p.hour]:.0%}, diaries so far {have_h[p.under, p.hour]:.0%})")
    text += CLOSING_TEXT
    return "\n".join(text)


def person_conditions(X_done: npt.NDArray[np.int8], target: npt.NDArray[np.float64], part: npt.NDArray[np.float64],
                      n_people: int, told: npt.NDArray[np.int8], u: npt.NDArray[np.float64], person_lines: int,
                      min_gap: float) -> list[dict[str, Any]]:
    """次の 1 人に付ける条件を引く（条件 memory_person）

    日次の条件: 生成済みの日次行動者率が公表値と min_gap 以上違う活動 k について、残りの人数で
    公表値に届くのに要る実施率 q_k = (t_k·N − H_k)/(N − n) を出す（0〜1 に収める）。条件を付けなかった人で
    の実施率 b_k と比べ、q_k < b_k なら確率 1 − q_k/b_k で「k をしない」、q_k > b_k なら確率
    (q_k − b_k)/(1 − b_k) で「k をする」を付ける。LLM が条件に従い、条件の無い人が b_k の率で k をするなら、
    N 人目で日次行動者率が公表値に届く。付いた条件が person_lines を超えたら |q_k − b_k| の小さい方から落とす。

    時刻の条件: hour_swap_pairs の組を差の大きい順に見て、日次の条件と矛盾しない最初の 1 組を
    「その時間帯は足りない活動をしている（多すぎる活動ではない）」として付ける。

    Args:
        X_done: (n, 96) これまでに確定した活動スケジュール
        target: (20, 96) 集計表の時刻別行動者率
        part: (20,) 日次行動者率の公表値
        n_people: 群で書く人数 N
        told: (n, 20) これまでの人に付けた日次の条件。+1 = する、−1 = しない、0 = 条件なし
        u: (20,) この人のための一様乱数。活動 k の条件を付けるかは u[k] < 確率 で決める
        person_lines: 日次の条件の数の上限
        min_gap: 日次・時刻とも、これより差の小さい箇所には条件を付けない（割合）

    Returns:
        条件の並び。{"kind": "not" | "do", "activity": k, "priority": |q_k − b_k|} と
        {"kind": "swap", "activity": 足りない活動, "instead_of": 多すぎる活動, "hour": 時計の時}
    """
    n = len(X_done)
    present = np.stack([(X_done == k).any(axis=1) for k in range(tg.NUM_ACT20)], axis=1)   # (n, 20)
    have = present.mean(axis=0)
    conds: list[dict[str, Any]] = []
    for k in range(tg.NUM_ACT20):
        if abs(have[k] - part[k]) < min_gap:
            continue
        q = float(np.clip((part[k] * n_people - present[:, k].sum()) / (n_people - n), 0.0, 1.0))
        free = told[:n, k] == 0
        b = float(present[free, k].mean()) if free.any() else float(have[k])
        if q < b:
            kind, prob = "not", 1.0 - q / b
        elif q > b and b < 1.0:
            kind, prob = "do", (q - b) / (1.0 - b)
        else:
            continue
        if u[k] < prob:
            conds.append({"kind": kind, "activity": k, "priority": round(abs(q - b), 4)})
    conds = sorted(conds, key=lambda c: -c["priority"])[:person_lines]
    excluded = {c["activity"] for c in conds if c["kind"] == "not"}
    for p in hour_swap_pairs(X_done, target, 24, min_gap):
        if p.under in excluded:
            continue
        conds.append({"kind": "swap", "activity": p.under, "instead_of": p.over, "hour": clock_hour(p.hour)})
        break
    return conds


def person_text(conds: list[dict[str, Any]]) -> str:
    """「この人の条件」を英語の文にする。条件が無いときは空文字"""
    if not conds:
        return ""
    lines = ["This person (so that the group moves closer to the survey):"]
    for c in conds:
        name = ACTIVITY_NAMES[c["activity"]]
        if c["kind"] == "not":
            lines.append(f"- does not do {name} at any time today.")
        elif c["kind"] == "do":
            lines.append(f"- does {name} at some time today.")
        else:
            h = int(c["hour"])
            lines.append(f"- During {h:02d}:00-{(h + 1) % 24:02d}:00 this person is doing {name}, "
                         f"not {ACTIVITY_NAMES[c['instead_of']]}.")
    lines.append("Keep the rest of the day realistic for this person.")
    return "\n".join(lines)


def group_text(d: int) -> str:
    """群の番号 d = (性 · 7 + 年齢) · 2 + 就業 を英語の人物の説明にする"""
    g, rest = divmod(d, tg.N_A * tg.N_E)
    a, e = divmod(rest, tg.N_E)
    return f"a {SEX_TEXT[g]} {AGE_TEXT[a]} living in Japan {EMPLOY_TEXT[e]}"


def build_messages(d: int, memory: str) -> list[dict[str, str]]:
    """1 人ぶんのプロンプトを作る

    Args:
        d: 群の番号
        memory: メモリの文章。メモリなしのときは空文字

    Returns:
        system と user のメッセージ
    """
    user = ["Write the diary of one person for a weekday (Monday to Friday) in October 2021.",
            f"Person: {group_text(d)}.",
            "Write one individual from this group."]
    if memory:
        user += ["", memory]
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "\n".join(user)}]


def generate_group(d: int, condition: str, n_people: int, target: npt.NDArray[np.float64],
                   part: npt.NDArray[np.float64], client: Chat, seed: int, top_k: int, min_gap: float, warmup: int,
                   max_tries: int, log_path: Path,
                   person_lines: int = 5) -> tuple[npt.NDArray[np.int8], list[list[dict[str, Any]]]]:
    """1 群ぶんを 1 人ずつ生成する。書いた活動スケジュールは確定で、後から直さない

    Args:
        d: 群の番号
        condition: CONDITIONS のどれか
        n_people: 生成する人数 N
        target: (20, 96) 集計表の時刻別行動者率
        part: (20,) 日次行動者率の公表値（memory_day・memory_person で使う）
        client: LLM への問い合わせ
        seed: 種。i 人目の try 回目は seed · 100000 + i · 10 + try を使う。条件を引く乱数は (seed, d, 条件の番号) から作る
        top_k: メモリに挙げる個数
        min_gap: メモリに挙げる差の下限
        warmup: この人数までは、どの条件でもメモリを渡さない（集計が粗すぎるため）
        max_tries: 形式が正しくないときにやり直す回数の上限。超えたら直して採用する
        log_path: 1 人 1 行の記録の書き出し先（jsonl）
        person_lines: memory_person の日次の条件の数の上限

    Returns:
        X (N, 96) int8 と、人ごとに付けた条件の並び（memory_person 以外は空の並び）
    """
    X = np.empty((n_people, tg.NUM_SLOTS), dtype=np.int8)
    told = np.zeros((n_people, tg.NUM_ACT20), dtype=np.int8)
    conditions: list[list[dict[str, Any]]] = []
    rng = np.random.default_rng([seed, d, CONDITIONS.index(condition)])
    with open(log_path, "w", encoding="utf-8") as log:
        for i in range(n_people):
            memory = ""
            conds: list[dict[str, Any]] = []
            u = rng.random(tg.NUM_ACT20)              # 人ごとに必ず 20 個引く（条件の有無で乱数の並びがずれない）
            if i >= warmup:
                if condition == "memory":
                    memory = memory_text(X[:i], target, top_k, min_gap)
                elif condition in ("memory_day", "memory_person"):
                    memory = memory_day_text(X[:i], target, part, top_k, min_gap)
                    if condition == "memory_person":
                        conds = person_conditions(X[:i], target, part, n_people, told, u, person_lines, min_gap)
                        memory = "\n\n".join(t for t in (memory, person_text(conds)) if t)
            for c in conds:
                if c["kind"] in ("not", "do"):
                    told[i, c["activity"]] = 1 if c["kind"] == "do" else -1
            conditions.append(conds)
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
                                  "problem": problem, "seconds": round(sec, 2), "memory": memory,
                                  "conditions": conds, "content": content, **meta}, ensure_ascii=False) + "\n")
            log.flush()
            print(f"[llm] 群 {d:02d} {condition:<13} {i + 1:>4}/{n_people}  {sec:6.1f} 秒  try {tries}"
                  f"{'  直した: ' + problem if problem else ''}", flush=True)
    return X, conditions


def main() -> None:
    """CLI"""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", type=int, nargs="+", default=[5, 0], help="生成する群の番号")
    ap.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=["none", "memory"])
    ap.add_argument("--N", type=int, default=100, help="群あたりの人数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-k", type=int, default=6, help="メモリに挙げる個数")
    ap.add_argument("--min-gap", type=float, default=0.03, help="メモリに挙げる差・条件を付ける差の下限（割合）")
    ap.add_argument("--warmup", type=int, default=5, help="この人数まではメモリを渡さない")
    ap.add_argument("--person-lines", type=int, default=5, help="memory_person の日次の条件の数の上限")
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
            X, conditions = generate_group(d, condition, args.N, target, part20[d], client, args.seed, args.top_k,
                                           args.min_gap, args.warmup, args.max_tries,
                                           out_dir / f"group{d:02d}_{condition}.jsonl", args.person_lines)
            np.savez_compressed(out_dir / f"group{d:02d}_{condition}.npz", X=X)
            key = {"stem": stem, "group": d, "condition": condition}
            follow = follow_rates(X, conditions) if condition == "memory_person" else {}
            summary.append({**key, "n_people": args.N, **evaluate(X, target, part20[d]), **follow,
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
