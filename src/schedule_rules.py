"""
schedule_rules.py
=================
SA（Matsumura & Murata, IEEE SMC 2025）のルール R1〜R4 の違反を数える。評価にだけ使う

LLM にはルールを見せない。「ルールを書かずに、ルールを守った 1 日になるか」を測るために、
SA が目的関数 f1 で減らしている違反の数を、同じ数え方で生成物に当てる。

| ルール | 主の活動 | 上限（グループ数） | 途中に入ってよい活動 |
|---|---|---|---|
| R1 | 睡眠 | 1 | なし |
| R2 | 通勤・通学 | 2 | 食事・買い物・移動・交際・受診 |
| R3 | 仕事 | 1 | 食事・買い物・移動・テレビ等・休養 |
| R4 | 学業 | 1 | R3 と同じ |

グループは「主の活動か途中に入ってよい活動だけが続く最長の区間のうち、主の活動を 1 スロット以上
含むもの」。スロット 95 と 0 をつなぐ円環として数えるので、04:00 をまたぐ夜の睡眠は 1 グループになる。
1 人の違反数は Σ_r max(0, グループ数 − 上限)。

写し元: DomainTransfer_trial の src/models/SA_Aggregate/sa_core.py（コミット 5a68b78。作業ツリーの
未コミットの変更は anneal の n の式だけで、ここで写した部分には無い）。写したのはルールの定数と
count_groups・person_violation（numba を外した素の Python）。全員ぶんをまとめて数える
rule_groups は、このリポジトリで書いた配列版で、count_groups と同じ値になることをテスト
（test_llm_memory_trial.py の j）で確かめる。

```mermaid
flowchart LR
    X["X (N, 96)<br/>20 分類"] --> M["RULE_MEMBER[r][X]<br/>member (N, 96)"]
    M --> ROT["最初の非 member スロットから回す<br/>（円環を 1 本の列にする）"]
    ROT --> RUN["run ごとに主の活動を含むか"]
    RUN --> G["rule_groups(X)<br/>(N, 4)"]
    G --> V["violations(X)<br/>Σ_r max(0, G − RULE_CAP)"]
```
"""
import numpy as np
import numpy.typing as npt

NUM_ACT20 = 20
NUM_SLOTS = 96

# 活動の番号は 20 分類の コード − 1
SLEEP, MEAL, COMMUTE, WORK, SCHOOL = 0, 2, 3, 4, 5
SHOPPING, TRAVEL, MEDIA, REST, SOCIAL, MEDICAL = 9, 10, 11, 12, 17, 18

RULE_NAMES = ["R1_sleep", "R2_commute", "R3_work", "R4_school"]
RULE_MAIN = np.array([SLEEP, COMMUTE, WORK, SCHOOL], dtype=np.int64)
RULE_CAP = np.array([1, 2, 1, 1], dtype=np.int64)
_INSERTS: list[list[int]] = [
    [],                                         # R1 睡眠
    [MEAL, SHOPPING, TRAVEL, SOCIAL, MEDICAL],  # R2 通勤・通学
    [MEAL, SHOPPING, TRAVEL, MEDIA, REST],      # R3 仕事
    [MEAL, SHOPPING, TRAVEL, MEDIA, REST],      # R4 学業（R3 と同じ）
]
# RULE_MEMBER[r, k]: 活動 k がルール r のグループに入れるか（主の活動か、途中に入ってよい活動）
RULE_MEMBER = np.zeros((len(RULE_MAIN), NUM_ACT20), dtype=np.bool_)
for _r, _main in enumerate(RULE_MAIN):
    RULE_MEMBER[_r, _main] = True
    RULE_MEMBER[_r, _INSERTS[_r]] = True


def count_groups(row: npt.NDArray[np.integer], member: npt.NDArray[np.bool_], main: int) -> int:
    """1 人の 1 日で、ルールの活動のグループ数を円環として数える（写し元と同じ手順）

    Args:
        row: (96,) 1 人の活動
        member: (20,) グループに入れる活動
        main: ルールの主の活動

    Returns:
        グループ数
    """
    n_slots = row.shape[0]
    start = -1
    for s in range(n_slots):
        if not member[row[s]]:
            start = s
            break
    if start == -1:                     # 1 日中 member の活動だけ
        return int(bool((row == main).any()))
    groups = 0
    in_run = False
    has_main = False
    # start は member でないので、start の次から 1 周して start で必ず区間が閉じる
    for j in range(1, n_slots + 1):
        a = row[(start + j) % n_slots]
        if member[a]:
            if not in_run:
                in_run = True
                has_main = False
            if a == main:
                has_main = True
        elif in_run:
            if has_main:
                groups += 1
            in_run = False
    return groups


def person_violation(row: npt.NDArray[np.integer]) -> int:
    """1 人の違反数 Σ_r max(0, グループ数 − 上限)"""
    v = 0
    for r in range(len(RULE_MAIN)):
        over = count_groups(row, RULE_MEMBER[r], int(RULE_MAIN[r])) - int(RULE_CAP[r])
        if over > 0:
            v += over
    return v


def rule_groups(X: npt.NDArray[np.integer]) -> npt.NDArray[np.int64]:
    """全員ぶんのグループ数をまとめて数える（count_groups の配列版）

    各行を「最初の非 member スロット」から始まるように回すと、円環の run が列の端をまたがなくなる。
    あとは線形の run ごとに、主の活動を含むかを数えればよい。

    Args:
        X: (N, 96) 20 分類の活動

    Returns:
        groups (N, 4)。列はルール R1〜R4
    """
    X = np.asarray(X, dtype=np.int64)
    n = X.shape[0]
    out = np.zeros((n, len(RULE_MAIN)), dtype=np.int64)
    rows = np.arange(n)
    for r, main in enumerate(RULE_MAIN):
        member = RULE_MEMBER[r][X]                                   # (N, 96)
        is_main = X == main
        all_member = member.all(axis=1)
        start = np.argmin(member, axis=1)                            # 最初の非 member（全 member の行は 0）
        idx = (start[:, None] + np.arange(NUM_SLOTS)[None, :]) % NUM_SLOTS
        m = np.take_along_axis(member, idx, axis=1)
        a = np.take_along_axis(is_main, idx, axis=1)
        prev = np.concatenate([np.zeros((n, 1), dtype=bool), m[:, :-1]], axis=1)
        run_id = np.cumsum(m & ~prev, axis=1)                        # 何本目の run か（1 始まり）
        rr, cc = np.nonzero(m & a)
        # 主の活動を含む (行, run) の組を重複なく数える。run は 1 行に高々 48 本
        pairs = np.unique(rr * (NUM_SLOTS + 1) + run_id[rr, cc])
        counts = np.bincount(pairs // (NUM_SLOTS + 1), minlength=n)
        counts[all_member] = is_main[all_member].any(axis=1)
        out[rows, r] = counts
    return out


def violations(X: npt.NDArray[np.integer]) -> npt.NDArray[np.int64]:
    """全員ぶんの違反数 Σ_r max(0, グループ数 − 上限), (N, 96) -> (N,)"""
    return np.maximum(rule_groups(X) - RULE_CAP[None, :], 0).sum(axis=1)
