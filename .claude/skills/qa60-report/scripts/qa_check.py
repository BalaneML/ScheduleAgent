"""
qa_check.py
===========
60QA 形式の報告書を、質問一覧（references/questions.md）と突き合わせる。

確かめること:
    1. 質問一覧の ID が、報告書にすべて 1 回ずつ出ているか（抜け・重複・一覧に無い ID）
    2. 章ごとに、記号（▲ ▽ △ ○ □ ◎）がいくつあるか

書くときに増やす質問（2-2「N 個目の関連分野」）は、報告書の枝番（2-2a、2-2b、…）を
それぞれ 1 問として数える。枝番が 1 つも無く 2-2 も無ければ、抜けとして報告する。

使い方:
    .venv/bin/python .claude/skills/qa60-report/scripts/qa_check.py docs/qa60/QA60_20261116_xxx.md
    .venv/bin/python .claude/skills/qa60-report/scripts/qa_check.py docs/qa60/QA60_20261116_xxx.md --write

--write を付けると、報告書の `<!-- qa60:status:start -->` と `<!-- qa60:status:end -->` の間を
完了率の表で置き換える。ID の抜け・重複があるときは終了コード 1 を返す。

報告書の見出しの形（これ以外の見出しは回答として数えない）:
    ### ▽ 3-1. 本研究では何というタスクを対象とするか？

データフロー:

```mermaid
flowchart LR
    QMD["questions_path<br/>references/questions.md"] --> LQ["load_questions(questions_path)"]
    RMD["report_path<br/>docs/qa60/QA60_*.md"] --> PA["parse_answers(report_text)"]
    LQ --> EX["expand_questions(listed_questions, answers)<br/>2-2 → 2-2a・2-2b…"]
    PA --> EX
    EX --> CK["check_ids(questions, answers)"]
    PA --> CK
    EX --> ST["status_table(questions, answers, chapter_titles)"]
    PA --> ST
    CK --> OUT["print(問題の一覧)"]
    ST --> OUT
    ST --> WR["write_status(report_path, table)<br/>--write のときだけ"]
```
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = SKILL_DIR / "references" / "questions.md"

# 記号の意味（SKILL.md §4 と同じ並び）
MARK_OPEN = "▲"            # 未完了
MARK_DRAFT = "▽"           # 下書き（Claude が作成、本人未確認）
MARKS_CHECKED = "△○□◎"     # 本人または教員が確認済み
ALL_MARKS = MARK_OPEN + MARK_DRAFT + MARKS_CHECKED

STATUS_START = "<!-- qa60:status:start -->"
STATUS_END = "<!-- qa60:status:end -->"

CHAPTER_RE = re.compile(r"^## (\d+)\. (.+?)\s*$")
QUESTION_ROW_RE = re.compile(r"^\| (\d+-\d+[a-z]?) \| (全体|実験) \|")
ANSWER_RE = re.compile(rf"^### ([{ALL_MARKS}]) (\d+-\d+[a-z]?)\.", re.MULTILINE)


@dataclass(frozen=True)
class Question:
    """質問一覧の 1 行

    Attributes:
        qid: 質問の ID（例 "4-8a"）
        chapter: 章の番号（例 "4"）
        scope: 範囲（"全体" か "実験"）
    """

    qid: str
    chapter: str
    scope: str


@dataclass(frozen=True)
class Answer:
    """報告書の回答の見出し 1 つ

    Attributes:
        qid: 質問の ID
        mark: 見出しに付いた記号（▲ ▽ △ ○ □ ◎ のどれか）
    """

    qid: str
    mark: str


def load_questions(questions_path: Path) -> tuple[list[Question], dict[str, str]]:
    """質問一覧の Markdown から、質問と章の名前を読む

    Args:
        questions_path: references/questions.md のパス

    Returns:
        (質問のリスト（一覧の並び順）, 章の番号 → 章の名前)
    """
    questions: list[Question] = []
    chapter_titles: dict[str, str] = {}
    for line in questions_path.read_text(encoding="utf-8").splitlines():
        chapter_match = CHAPTER_RE.match(line)
        if chapter_match:
            chapter_titles[chapter_match.group(1)] = chapter_match.group(2)
            continue
        row_match = QUESTION_ROW_RE.match(line)
        if row_match:
            qid = row_match.group(1)
            questions.append(Question(qid=qid, chapter=qid.split("-")[0], scope=row_match.group(2)))
    return questions, chapter_titles


def parse_answers(report_text: str) -> list[Answer]:
    """報告書の本文から、回答の見出し（### 記号 ID.）を出てきた順に取り出す

    Args:
        report_text: 報告書の全文

    Returns:
        回答の見出しのリスト
    """
    return [Answer(qid=m.group(2), mark=m.group(1)) for m in ANSWER_RE.finditer(report_text)]


def expand_questions(questions: list[Question], answers: list[Answer]) -> tuple[list[Question], list[str]]:
    """書くときに増やす質問（例 2-2）を、報告書にある枝番（2-2a、2-2b、…）で置き換える

    一覧に `2-2` があり `2-2a` が無いとき、報告書の `2-2a`・`2-2b` は `2-2` を増やした回答とみなす。
    一覧に最初から枝番で載っている質問（4-8a など）は、そのまま扱う。

    Args:
        questions: 質問一覧（一覧に書いてあるとおり）
        answers: 報告書の回答の見出し

    Returns:
        (報告書に合わせて増やした質問のリスト, 問題を説明する文のリスト)
    """
    listed_ids = {q.qid for q in questions}
    answer_ids = list(dict.fromkeys(a.qid for a in answers))
    expanded: list[Question] = []
    problems: list[str] = []
    for question in questions:
        branch_ids = sorted(
            qid for qid in answer_ids
            if qid not in listed_ids and re.fullmatch(re.escape(question.qid) + r"[a-z]", qid)
        )
        if not branch_ids:
            expanded.append(question)
            continue
        if question.qid in answer_ids:
            problems.append(f"{question.qid} と枝番（{'・'.join(branch_ids)}）の両方がある。枝番だけにする")
        expanded.extend(Question(qid=qid, chapter=question.chapter, scope=question.scope) for qid in branch_ids)
    return expanded, problems


def check_ids(questions: list[Question], answers: list[Answer]) -> list[str]:
    """ID の抜け・重複・一覧に無い ID を調べる

    Args:
        questions: 質問一覧（expand_questions で増やした後のもの）
        answers: 報告書の回答の見出し

    Returns:
        問題を説明する文のリスト（問題が無ければ空）
    """
    known_ids = [q.qid for q in questions]
    answer_counts = Counter(a.qid for a in answers)
    problems: list[str] = []

    missing = [qid for qid in known_ids if answer_counts[qid] == 0]
    duplicated = [qid for qid in known_ids if answer_counts[qid] > 1]
    unknown = sorted(set(answer_counts) - set(known_ids))

    if missing:
        problems.append("報告書に無い ID: " + "・".join(missing))
    if duplicated:
        problems.append("2 回以上出てくる ID: " + "・".join(duplicated))
    if unknown:
        problems.append("質問一覧に無い ID: " + "・".join(unknown))
    return problems


def status_table(questions: list[Question], answers: list[Answer], chapter_titles: dict[str, str]) -> str:
    """章ごとの完了率を Markdown の表にする

    「下書き以上」は ▲ 以外の回答、「確認済み」は △ ○ □ ◎ の回答を数える。

    Args:
        questions: 質問一覧
        answers: 報告書の回答の見出し
        chapter_titles: 章の番号 → 章の名前

    Returns:
        完了率の表と、▲ の ID の一覧（Markdown）
    """
    mark_of = {a.qid: a.mark for a in answers}
    chapters = list(dict.fromkeys(q.chapter for q in questions))

    lines = [
        "| 章 | 問数 | ▲ 未完了 | ▽ 下書き | 確認済み（△ ○ □ ◎） | 下書き以上 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    totals: Counter[str] = Counter()
    for chapter in chapters:
        marks = [mark_of.get(q.qid, MARK_OPEN) for q in questions if q.chapter == chapter]
        n_open = sum(m == MARK_OPEN for m in marks)
        n_draft = sum(m == MARK_DRAFT for m in marks)
        n_checked = sum(m in MARKS_CHECKED for m in marks)
        totals.update(n=len(marks), open=n_open, draft=n_draft, checked=n_checked)
        lines.append(
            f"| {chapter}. {chapter_titles.get(chapter, '')} | {len(marks)} | {n_open} | {n_draft} | {n_checked} "
            f"| {len(marks) - n_open}/{len(marks)} |"
        )
    lines.append(
        f"| 合計 | {totals['n']} | {totals['open']} | {totals['draft']} | {totals['checked']} "
        f"| {totals['n'] - totals['open']}/{totals['n']} |"
    )

    open_ids = [q.qid for q in questions if mark_of.get(q.qid, MARK_OPEN) == MARK_OPEN]
    lines.append("")
    lines.append("▲ の質問: " + ("・".join(open_ids) if open_ids else "なし"))
    return "\n".join(lines)


def write_status(report_path: Path, table: str) -> bool:
    """報告書の目印の間を、完了率の表で置き換える

    Args:
        report_path: 報告書のパス
        table: status_table の出力

    Returns:
        置き換えたら True。目印が見つからなければ False（報告書は変えない）
    """
    text = report_path.read_text(encoding="utf-8")
    start = text.find(STATUS_START)
    end = text.find(STATUS_END)
    if start < 0 or end < start:
        return False
    new_text = text[: start + len(STATUS_START)] + "\n" + table + "\n" + text[end:]
    report_path.write_text(new_text, encoding="utf-8")
    return True


def main() -> int:
    """コマンドラインの入口

    Returns:
        終了コード（ID の抜け・重複・未知の ID があれば 1）
    """
    parser = argparse.ArgumentParser(description="60QA 形式の報告書の ID と完了率を確かめる")
    parser.add_argument("report", type=Path, help="報告書（docs/qa60/QA60_*.md）")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS, help="質問一覧の Markdown")
    parser.add_argument("--write", action="store_true", help="報告書の目印の間に完了率の表を書き込む")
    args = parser.parse_args()

    listed_questions, chapter_titles = load_questions(args.questions)
    answers = parse_answers(args.report.read_text(encoding="utf-8"))
    questions, problems = expand_questions(listed_questions, answers)
    problems += check_ids(questions, answers)
    table = status_table(questions, answers, chapter_titles)

    print(table)
    for problem in problems:
        print("★ " + problem)
    if args.write:
        if write_status(args.report, table):
            print(f"完了率の表を書き込んだ: {args.report}")
        else:
            print(f"★ 目印（{STATUS_START} と {STATUS_END}）が見つからないので、書き込まなかった")
            return 1
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
