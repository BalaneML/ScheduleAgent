"""
llm_memory_trial.py・schedule_metrics.py・score_pools.py と写し 3 つの単体テスト

    a. 区間 → スロット: 04:00 起点で 96 スロットに並ぶ。日付をまたぐ時刻（00:30 など）も正しい位置に入る
    b. 形式が正しくない応答: 時刻が進まない・04:00 で終わらない・活動名が違う、を理由つきで退ける
    c. 直して採用: repair=True なら、時刻が進まない区間を捨て、最後を 04:00 まで延ばして 96 スロットを返す
    d. 往復: スロット → 区間 → スロット で元に戻る
    e. メモリ（memory）: 生成済みの集計が集計表より多い箇所・少ない箇所が、正しい活動と時刻で挙がる
    f. 空回し: 偽の応答で 1 群を最後まで回し、形と指標を確かめる（前処理済みの社会生活基本調査の表を読む）
    g. 写し元と一致: stula_targets の読み込みと 12 分類への対応表が DomainTransfer_trial の sa_targets と同じ値を返す
    h. メモリ（memory_day）: 日次行動者率の差と、時間帯の「多すぎる・足りない」の組が正しく挙がる
    i. この人の条件（memory_person）: 引き方が規則どおりで、同じ種なら同じ条件・同じ生成物になる
    j. 写し元と一致: 起床・就寝の規則、平均時刻の読み込み、R1〜R4 の数え方が写し元と同じ値を返す
    k. 採点: score_pools の 20 分類の行が evaluate と同じ値になり、条件に従った割合を正しく数える
    l. 書き直し: 選ぶ人が総当たりと一致し、採った後の誤差が増えず、記録と推移の行数が試した回数と合う

g と j は DomainTransfer_trial が無い端末では飛ばす。場所は環境変数 DOMAIN_TRANSFER_ROOT で指定できる。

    .venv/bin/python src/test_llm_memory_trial.py
"""
import importlib.util
import json
import os
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import llm_memory_trial as trial  # noqa: E402
import rewrite_trial as rw  # noqa: E402
import schedule_metrics as sm  # noqa: E402
import schedule_rules as sr  # noqa: E402
import score_pools as sp  # noqa: E402
import stula_derived_times as sdt  # noqa: E402

tg = trial.tg
SLEEP, MEALS, WORK, HOUSEWORK, SPORTS = 0, 2, 4, 6, 15
SOURCE_ROOT = Path(os.environ.get("DOMAIN_TRANSFER_ROOT", HERE.parents[1] / "DomainTransfer_trial"))


def _has_mean_times() -> bool:
    """平均時刻の表（採点だけに使う）が置いてあるか。LLM を回すだけの端末には無くてよい"""
    return all((tg.stula_dir() / f"{name}.csv").is_file() for name in sdt.MEANTIME_TABLES.values())


def _seg(name_code: int, end: str) -> dict[str, str]:
    """テスト用の区間を作る"""
    return {"activity": trial.ACTIVITY_NAMES[name_code], "end": end}


def _load_source(name: str, path: Path) -> Any:
    """写し元のモジュールを、ファイルの場所から読む"""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_segments_to_slots() -> None:
    """a. 04:00 起点で並び、日付をまたぐ時刻も正しい位置に入る"""
    segments = [_seg(SLEEP, "07:00"), _seg(MEALS, "07:30"), _seg(WORK, "18:00"), _seg(MEALS, "00:30"),
                _seg(SLEEP, "04:00")]
    row, problem = trial.segments_to_slots(segments)
    assert row is not None and problem == ""
    assert row.shape == (96,) and row.min() >= 0
    assert (row[:12] == SLEEP).all(), "04:00〜07:00 は 12 スロット"
    assert (row[12:14] == MEALS).all() and (row[14:56] == WORK).all()
    assert (row[56:82] == MEALS).all(), "18:00〜00:30 は 26 スロット"
    assert (row[82:] == SLEEP).all(), "00:30〜04:00 は 14 スロット"
    print("a. 区間 → スロット: OK")


def test_invalid_segments() -> None:
    """b. 形式が正しくない応答を、理由つきで退ける"""
    cases = {
        "時刻が進まない": [_seg(SLEEP, "07:00"), _seg(WORK, "06:00"), _seg(SLEEP, "04:00")],
        "04:00 で終わらない": [_seg(SLEEP, "07:00"), _seg(WORK, "18:00")],
        "活動名が違う": [{"activity": "Working", "end": "04:00"}],
        "空": [],
    }
    for label, segments in cases.items():
        row, problem = trial.segments_to_slots(segments)
        assert row is None and problem != "", label
        print(f"b. {label}: 退けた（{problem}）")


def test_repair() -> None:
    """c. repair=True なら必ず 96 スロットを返す"""
    row, problem = trial.segments_to_slots([_seg(SLEEP, "07:00"), _seg(WORK, "06:00"), _seg(MEALS, "12:00")],
                                           repair=True)
    assert row is not None and problem != ""
    assert (row[:12] == SLEEP).all() and (row[12:] == MEALS).all(), "進まない区間は捨て、最後を 04:00 まで延ばす"
    print(f"c. 直して採用: OK（{problem}）")


def test_round_trip() -> None:
    """d. スロット → 区間 → スロット で元に戻る"""
    rng = np.random.default_rng(0)
    for _ in range(50):
        row = np.repeat(rng.integers(0, tg.NUM_ACT20, size=24), 4).astype(np.int8)
        back, problem = trial.segments_to_slots(trial.slots_to_segments(row))
        assert back is not None and problem == "" and (back == row).all()
    print("d. 往復: OK")


def test_memory_text() -> None:
    """e. 多い箇所と少ない箇所が、正しい活動と時刻で挙がる"""
    target = np.zeros((tg.NUM_ACT20, tg.NUM_SLOTS))
    target[SLEEP] = 1.0
    target[SLEEP, 32:36], target[WORK, 32:36] = 0.5, 0.5        # 12:00〜13:00 は睡眠と仕事が半々
    X_done = np.full((4, tg.NUM_SLOTS), SLEEP, dtype=np.int8)   # 生成済みは全員ずっと睡眠
    text = trial.memory_text(X_done, target, top_k=3, min_gap=0.03)
    few, many = text.split("Too MANY")
    assert "Work, 12:00-13:00: survey 50%, diaries so far 0%" in few
    assert "Sleep, 12:00-13:00: survey 50%, diaries so far 100%" in many
    assert text.count("\n- ") == 2, "差が min_gap に満たない箇所は挙げない"
    assert trial.memory_text(X_done, trial.slot_rates(X_done), top_k=3, min_gap=0.03) == "", "差が無ければ空"
    print("e. メモリ: OK")
    print(text)


def test_dry_run() -> None:
    """f. 偽の応答で 1 群を回し、形と指標を確かめる"""
    d, n_people = 5, 12
    target, part = tg.load_rates20()[d], tg.load_participation20()[d]
    with tempfile.TemporaryDirectory() as tmp:
        for condition in trial.CONDITIONS:
            log_path = Path(tmp) / f"{condition}.jsonl"
            X, conditions = trial.generate_group(d, condition, n_people, target, part, trial.FakeChat(target), seed=1,
                                                 top_k=4, min_gap=0.03, warmup=3, max_tries=3, log_path=log_path)
            assert X.shape == (n_people, 96) and X.min() >= 0 and X.max() < tg.NUM_ACT20
            records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
            assert len(records) == n_people and len(conditions) == n_people
            assert all(r["memory"] == "" for r in records[:3]), "warmup の間はメモリを渡さない"
            if condition == "none":
                assert all(r["memory"] == "" for r in records)
            if condition != "memory_person":
                assert all(c == [] for c in conditions)
            else:
                assert any(c for c in conditions), "memory_person では条件が付く"
                assert all(r["conditions"] == c for r, c in zip(records, conditions))
            scores = trial.evaluate(X, target, part)
            assert all(np.isfinite(v) for v in scores.values())
            assert 0.0 <= scores["single_slot_share"] <= 1.0 and 0.0 <= scores["overfill_share"] <= 1.0
            curve = trial.error_curve(X, target, step=5)
            assert [row["n"] for row in curve] == [5.0, 10.0, 12.0]
            print(f"f. 空回し {condition}: OK  {({k: round(v, 3) for k, v in scores.items()})}")
    assert "aged 35-44" in trial.group_text(5) and "usually works" in trial.group_text(5)
    assert "aged 15-24" in trial.group_text(0) and "does not work" in trial.group_text(0)
    assert "woman" in trial.group_text(27) and "75 or older" in trial.group_text(27)


def test_matches_source() -> None:
    """g. 写し元の sa_targets と同じ値を返す"""
    src = SOURCE_ROOT / "src" / "models" / "SA_Aggregate" / "sa_targets.py"
    if not src.is_file():
        print(f"g. 写し元と一致: 飛ばした（{src} が無い）")
        return
    sa = _load_source("sa_targets", src)
    assert np.array_equal(tg.load_rates20(), sa.load_rates20()), "時刻別行動者率が写し元と違う"
    assert np.array_equal(tg.load_participation20(), sa.load_participation20()), "日次行動者率が写し元と違う"
    target = tg.load_rates20()[5]
    assert np.array_equal(tg.round_slot_counts(target, 100), sa.round_slot_counts(target, 100))
    assert np.array_equal(tg.STULA20_TO_COMMON, sa.STULA20_TO_COMMON), "12 分類への対応表が写し元と違う"
    print("g. 写し元と一致: OK")


def _day_with(housework_slots: slice) -> np.ndarray:
    """ずっと睡眠で、指定したスロットだけ家事をする 1 日"""
    row = np.full(tg.NUM_SLOTS, SLEEP, dtype=np.int8)
    row[housework_slots] = HOUSEWORK
    return row


def test_memory_day_text() -> None:
    """h. 日次行動者率の差と、時間帯の「多すぎる・足りない」の組が正しく挙がる"""
    target = np.zeros((tg.NUM_ACT20, tg.NUM_SLOTS))
    target[SLEEP] = 1.0
    target[SLEEP, 32:36], target[WORK, 32:36] = 0.5, 0.5        # 12:00〜13:00 は睡眠と仕事が半々
    part = np.zeros(tg.NUM_ACT20)
    part[SLEEP], part[WORK] = 1.0, 0.5
    X_done = np.full((4, tg.NUM_SLOTS), SLEEP, dtype=np.int8)
    text = trial.memory_day_text(X_done, target, part, top_k=3, min_gap=0.03)
    assert "Too MANY" not in text
    assert "- Work: survey 50%, diaries so far 0%" in text
    assert ("- 12:00-13:00: too many doing Sleep (survey 50%, diaries so far 100%), "
            "too few doing Work (survey 50%, diaries so far 0%)") in text
    pairs = trial.hour_swap_pairs(X_done, target, top_k=3, min_gap=0.03)
    assert pairs == [trial.SwapPair(8, SLEEP, WORK, 0.5)], "04:00 起点で 8 番目の時間が 12:00〜13:00"
    assert trial.memory_day_text(X_done, trial.slot_rates(X_done), sm.participation(X_done), 3, 0.03) == ""
    print("h. メモリ（memory_day）: OK")
    print(text)


def test_person_conditions() -> None:
    """i. 条件の引き方が規則どおりで、同じ種なら同じ条件・同じ生成物になる"""
    n_people = 10
    X_done = np.stack([_day_with(slice(60, 64))] * 5)            # 5 人とも 19:00〜20:00 に家事
    target = trial.slot_rates(X_done)
    target[HOUSEWORK, 60:64], target[SLEEP, 60:64] = 0.0, 1.0   # 公表では 19 時台は全員が睡眠
    part = sm.participation(X_done)
    part[HOUSEWORK], part[SPORTS] = 0.2, 0.5                    # 家事は多すぎ、スポーツは足りない
    told = np.zeros((n_people, tg.NUM_ACT20), dtype=np.int8)
    # q（家事）= clip((0.2·10 − 5)/5) = 0、b = 1 → 確率 1 で「しない」。q（スポーツ）= 1、b = 0 → 確率 1 で「する」
    conds = trial.person_conditions(X_done, target, part, n_people, told, np.zeros(tg.NUM_ACT20), 5, 0.03)
    assert [(c["kind"], c["activity"]) for c in conds] == [("not", HOUSEWORK), ("do", SPORTS), ("swap", SLEEP)]
    assert conds[2]["instead_of"] == HOUSEWORK and conds[2]["hour"] == 19
    text = trial.person_text(conds)
    assert "- does not do Housework at any time today." in text and "- does Sports at some time today." in text
    assert "- During 19:00-20:00 this person is doing Sleep, not Housework." in text
    one = trial.person_conditions(X_done, target, part, n_people, told, np.zeros(tg.NUM_ACT20), 1, 0.03)
    assert [c["kind"] for c in one] == ["not", "swap"], "日次の条件は person_lines 個まで"
    none_drawn = trial.person_conditions(X_done, target, part, n_people, told, np.ones(tg.NUM_ACT20), 5, 0.03)
    assert [c["kind"] for c in none_drawn] == ["swap"], "乱数が確率以上なら日次の条件は付かない"
    # 家事を「しない」と決めた人には、家事を足りない側にした時刻の組を付けず、次に大きい組を使う
    target2 = trial.slot_rates(X_done)
    target2[HOUSEWORK, 0:4], target2[SLEEP, 0:4] = 1.0, 0.0     # 04 時台：睡眠が多すぎ、家事が足りない（差 1.0）
    target2[MEALS, 8:12], target2[SLEEP, 8:12] = 0.5, 0.5       # 06 時台：睡眠が多すぎ、食事が足りない（差 0.5）
    conds2 = trial.person_conditions(X_done, target2, part, n_people, told, np.zeros(tg.NUM_ACT20), 5, 0.03)
    assert ("not", HOUSEWORK) in [(c["kind"], c["activity"]) for c in conds2]
    swap2 = [c for c in conds2 if c["kind"] == "swap"]
    assert len(swap2) == 1 and swap2[0]["activity"] == MEALS and swap2[0]["hour"] == 6
    # 同じ種なら同じ条件・同じ生成物
    d = 5
    rates, part5 = tg.load_rates20()[d], tg.load_participation20()[d]
    runs = []
    with tempfile.TemporaryDirectory() as tmp:
        for rep in range(2):
            runs.append(trial.generate_group(d, "memory_person", 15, rates, part5, trial.FakeChat(rates), seed=3,
                                             top_k=4, min_gap=0.03, warmup=3, max_tries=3,
                                             log_path=Path(tmp) / f"rep{rep}.jsonl"))
    assert np.array_equal(runs[0][0], runs[1][0]) and runs[0][1] == runs[1][1]
    print("i. この人の条件: OK")
    print(text)


def test_sources_derived_and_rules() -> None:
    """j. 起床・就寝の規則、平均時刻の読み込み、R1〜R4 の数え方が写し元と同じ値を返す"""
    paths = {
        "src_stula_derived_times": SOURCE_ROOT / "src" / "eval" / "stula_derived_times.py",
        "src_stage2_published": SOURCE_ROOT / "src" / "models" / "DDPM_Aggregate_Simple" / "stage2_published.py",
        "src_sa_core": SOURCE_ROOT / "src" / "models" / "SA_Aggregate" / "sa_core.py",
    }
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if not _has_mean_times():
        missing.insert(0, "平均時刻の表（meantime_*.csv）")
    if missing:
        print(f"j. 写し元と一致: 飛ばした（{missing[0]} が無い）")
        return
    src_dt = _load_source("src_stula_derived_times", paths["src_stula_derived_times"])
    src_pub = _load_source("src_stage2_published", paths["src_stage2_published"])
    # sa_core は numba の njit を使う。numba が無ければ、素の Python で動く置き換えを入れて読む
    stubbed = importlib.util.find_spec("numba") is None
    if stubbed:
        stub = types.ModuleType("numba")
        setattr(stub, "njit", lambda *a, **k: (lambda f: f))
        sys.modules["numba"] = stub
    try:
        src_sa = _load_source("src_sa_core", paths["src_sa_core"])
    finally:
        if stubbed:
            del sys.modules["numba"]

    rng = np.random.default_rng(0)
    blocks = [0, 2, 3, 4, 5, 9, 10, 11, 12, 17, 18]
    samples = [rng.integers(0, 20, size=(200, 96)),                                   # 細切れ（20 分類）
               np.repeat(rng.choice(blocks, size=(200, 24)), 4, axis=1),               # 1 時間単位（20 分類）
               rng.integers(0, 12, size=(200, 96))]                                    # 12 分類
    for X in samples:
        X = X.astype(np.int64)
        a, b = sdt.derived_time_summary(X), src_dt.derived_time_summary(X)
        assert a.keys() == b.keys()
        assert all((np.isnan(a[k]) and np.isnan(b[k])) or a[k] == b[k] for k in a), "起床・就寝の要約が写し元と違う"
        assert np.array_equal(sdt.wake_time(X), src_dt.wake_time(X), equal_nan=True)
        assert np.array_equal(sdt.bed_time(X), src_dt.bed_time(X), equal_nan=True)
    mine, theirs = sdt.load_mean_times(), src_pub.load_mean_times()
    for key in ("wake_hour", "bed_hour", "wake_actor_rate", "bed_actor_rate", "pop"):
        assert np.array_equal(mine[key], theirs[key], equal_nan=True), f"平均時刻の {key} が写し元と違う"
    for X in samples[:2]:
        X8 = X.astype(np.int8)
        ref_groups = np.array([[src_sa.count_groups(X8[i], src_sa.RULE_MEMBER[r], int(src_sa.RULE_MAIN[r]))
                                for r in range(len(src_sa.RULE_MAIN))] for i in range(len(X8))])
        ref_viol = np.array([src_sa.person_violation(X8[i], src_sa.RULE_MEMBER, src_sa.RULE_MAIN, src_sa.RULE_CAP)
                             for i in range(len(X8))])
        assert np.array_equal(sr.rule_groups(X8), ref_groups), "グループ数が写し元と違う"
        assert np.array_equal(sr.violations(X8), ref_viol), "違反数が写し元と違う"
        assert all(sr.person_violation(X8[i]) == ref_viol[i] for i in range(len(X8)))
    print(f"j. 写し元と一致: OK（numba の置き換え {'あり' if stubbed else 'なし'}）")


def test_scoring() -> None:
    """k. score_pools の 20 分類の行が evaluate と同じ値になり、条件に従った割合を正しく数える"""
    if not _has_mean_times():
        print("k. 採点: 飛ばした（平均時刻の表 meantime_*.csv が無い）")
        return
    d = 5
    rates, part = tg.load_rates20(), tg.load_participation20()
    with tempfile.TemporaryDirectory() as tmp:
        X, _ = trial.generate_group(d, "none", 20, rates[d], part[d], trial.FakeChat(rates[d]), seed=2, top_k=4,
                                    min_gap=0.03, warmup=3, max_tries=3, log_path=Path(tmp) / "none.jsonl")
    table = sp.score([sp.Pool("llm", "dry:none", d, X, tg.NUM_ACT20)])
    assert list(table["resolution"]) == ["act20", "common12"]
    row20 = table.iloc[0]
    for key, value in trial.evaluate(X, rates[d], part[d]).items():
        assert np.isclose(row20[key], value), key
    assert row20["floor_mae_pt"] > 0 and 0.0 <= row20["violation_share"] <= 1.0
    assert 0.0 <= row20["pairwise_distance"] <= 1.0
    row12 = table.iloc[1]
    assert np.isclose(row12["rate_mae_pt"], sm.evaluate_common(sm.to_common(X), sm.rates_to_common(rates[d]),
                                                               part[d])["rate_mae_pt"])
    assert set(sm.EXACT_COMMON) == {1, 2, 3, 4, 6, 9, 10}, "1 対 1 で対応する 12 分類は 7 つ"

    # 条件に従った割合：0 は家事をしない（従った）、1 はスポーツをする（従わない）、2 は 12 時台に仕事（従った）
    X3 = np.full((3, tg.NUM_SLOTS), SLEEP, dtype=np.int8)
    X3[1, 60:64] = HOUSEWORK
    X3[2, sm.hour_slots(12)] = WORK
    conditions: list[list[dict[str, Any]]] = [
        [{"kind": "not", "activity": HOUSEWORK}],
        [{"kind": "do", "activity": SPORTS}],
        [{"kind": "swap", "activity": WORK, "instead_of": MEALS, "hour": 12}],
    ]
    rates3 = sm.follow_rates(X3, conditions)
    assert rates3["follow_not"] == 1.0 and rates3["follow_do"] == 0.0 and rates3["follow_swap"] == 1.0
    assert rates3["base_not"] == 0.5, "家事に条件の無い 2 人のうち、家事をしない人は 1 人"
    assert rates3["base_do"] == 0.0 and rates3["n_not"] == 1.0
    print("k. 採点: OK")


def test_rewrite() -> None:
    """l. 選ぶ人が総当たりと一致し、誤差は増えず、記録の行数が合う"""
    d = 5
    rates, part = tg.load_rates20()[d], tg.load_participation20()[d]
    with tempfile.TemporaryDirectory() as tmp:
        X0, _ = trial.generate_group(d, "none", 30, rates, part, trial.FakeChat(rates), seed=4, top_k=4, min_gap=0.03,
                                     warmup=3, max_tries=3, log_path=Path(tmp) / "start.jsonl")
        # 総当たり：1 人ずつ抜いて和を計算する
        brute = [sum(rw.fit_error(np.delete(X0, j, axis=0), rates, part)) for j in range(len(X0))]
        assert rw.pick_person(X0, rates, part, set()) == int(np.argmin(brute))
        assert rw.pick_person(X0, rates, part, {int(np.argmin(brute))}) == int(np.argsort(brute, kind="stable")[1])
        log_path = Path(tmp) / "rewrite.jsonl"
        X0 = np.repeat(X0[:1], len(X0), axis=0)                 # 全員同じ 1 日（誤差が大きく、床に届かない）
        X, curve = rw.rewrite_group(d, X0, rates, part, trial.FakeChat(rates), seed=5, top_k=4, min_gap=0.03,
                                    person_lines=5, max_rewrites=12, cooldown=3, max_tries=3, log_path=log_path)
        records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    total = [row["rate_mae_pt"] + row["part_mae_pt"] for row in curve]
    assert all(b <= a + 1e-12 for a, b in zip(total, total[1:])), "採るのは和が下がるときだけ"
    assert len(records) == 12, "床に届かなければ max_rewrites 回まで試す"
    assert len(records) == len(curve) - 1 and curve[-1]["accepted"] == sum(r["accepted"] for r in records)
    assert X.shape == X0.shape and int((X != X0).any(axis=1).sum()) <= int(curve[-1]["accepted"])
    assert "This is the person's current diary:" in rw.rewrite_messages(d, X0[0], "")[1]["content"]
    assert rw.diary_text(X0[0]).startswith("- 04:00-")
    print(f"l. 書き直し: OK（{len(records)} 回試し、{int(curve[-1]['accepted'])} 回採った、和 {total[0]:.2f} → {total[-1]:.2f}）")


def main() -> None:
    """全テストを順に実行する"""
    test_segments_to_slots()
    test_invalid_segments()
    test_repair()
    test_round_trip()
    test_memory_text()
    test_dry_run()
    test_matches_source()
    test_memory_day_text()
    test_person_conditions()
    test_sources_derived_and_rules()
    test_scoring()
    test_rewrite()
    print("すべて OK")


if __name__ == "__main__":
    main()
