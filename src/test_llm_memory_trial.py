"""
llm_memory_trial.py の単体テスト

    a. 区間 → スロット: 04:00 起点で 96 スロットに並ぶ。日付をまたぐ時刻（00:30 など）も正しい位置に入る
    b. 形式が正しくない応答: 時刻が進まない・04:00 で終わらない・活動名が違う、を理由つきで退ける
    c. 直して採用: repair=True なら、時刻が進まない区間を捨て、最後を 04:00 まで延ばして 96 スロットを返す
    d. 往復: スロット → 区間 → スロット で元に戻る
    e. メモリ: 生成済みの集計が集計表より多い箇所・少ない箇所が、正しい活動と時刻で挙がる
    f. 空回し: 偽の応答で 1 群を最後まで回し、形と指標を確かめる（前処理済みの社会生活基本調査の表を読む）
    g. 写し元と一致: stula_targets の読み込みが DomainTransfer_trial の sa_targets と同じ値を返す
       （DomainTransfer_trial が無い端末では飛ばす。場所は環境変数 DOMAIN_TRANSFER_ROOT で指定できる）

    .venv/bin/python src/test_llm_memory_trial.py
"""
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import llm_memory_trial as trial  # noqa: E402

tg = trial.tg
SLEEP, MEALS, WORK = 0, 2, 4


def _seg(name_code: int, end: str) -> dict[str, str]:
    """テスト用の区間を作る"""
    return {"activity": trial.ACTIVITY_NAMES[name_code], "end": end}


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
            X = trial.generate_group(d, condition, n_people, target, trial.FakeChat(target), seed=1, top_k=4,
                                     min_gap=0.03, warmup=3, max_tries=3, log_path=Path(tmp) / f"{condition}.jsonl")
            assert X.shape == (n_people, 96) and X.min() >= 0 and X.max() < tg.NUM_ACT20
            lines = (Path(tmp) / f"{condition}.jsonl").read_text(encoding="utf-8").splitlines()
            assert len(lines) == n_people
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
    root = Path(os.environ.get("DOMAIN_TRANSFER_ROOT", HERE.parents[1] / "DomainTransfer_trial"))
    src = root / "src" / "models" / "SA_Aggregate" / "sa_targets.py"
    if not src.is_file():
        print(f"g. 写し元と一致: 飛ばした（{src} が無い）")
        return
    spec = importlib.util.spec_from_file_location("sa_targets", src)
    assert spec is not None and spec.loader is not None
    sa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sa)
    assert np.array_equal(tg.load_rates20(), sa.load_rates20()), "時刻別行動者率が写し元と違う"
    assert np.array_equal(tg.load_participation20(), sa.load_participation20()), "日次行動者率が写し元と違う"
    target = tg.load_rates20()[5]
    assert np.array_equal(tg.round_slot_counts(target, 100), sa.round_slot_counts(target, 100))
    print("g. 写し元と一致: OK")


def main() -> None:
    """全テストを順に実行する"""
    test_segments_to_slots()
    test_invalid_segments()
    test_repair()
    test_round_trip()
    test_memory_text()
    test_dry_run()
    test_matches_source()
    print("すべて OK")


if __name__ == "__main__":
    main()
