# 質問一覧（論文 L 用に改良した 60QA）

- **版：** 2026-11-18（論文 P 用の 2026-11-16 版を、論文 L 用に書き換えた）
- **元：** 慶應義塾大学 杉浦孔明研究室「60 Questions & Answers 公開版」（7 章 60 問）
- **論文 L：** LLM（オープンウェイト）に活動スケジュールを 1 人ずつ書かせ、生成済みの集計と集計表の差をメモリとして次の人に渡し、社会生活基本調査 2021 の集計表に合わせる。個票は使わない
- **本研究用：** 8 章 73 問。元の 60 問のうち 41 問をそのまま残し、18 問の文言を変え、1 問を削り、12 問を足した。4-8 は構成要素ごとに a〜c に分けたので、ID は 73 個になる。2-2 は元のシートのまま 1 問にしてあり、報告書を書くときに関連分野の数だけ 2-2a、2-2b、… と増やす。変更の理由は [changes-from-original.md](changes-from-original.md)
- **ID：** 元のシートと同じ番号を使う。足した質問は各章の末尾の番号にした

## 列の読み方

| 列 | 意味 |
| --- | --- |
| 範囲 | **全体** ＝ 研究全体で 1 つに決まる回答。前回の QA から引き継ぎ、根拠が変わったときだけ直す。**実験** ＝ 報告する実験の結果で毎回書き直す回答 |
| 答え方 | この研究で何を書けば回答になるか。「（未決）」は、2026-11-18 の時点でユーザーが決めていない点 |
| 根拠 | 回答の材料がある場所（下の略号） |

| 略号 | 場所 |
| --- | --- |
| 骨 | 論文の骨組み `docs/YYYY-MM-DD_paper_skeleton.md` のうち最も新しいもの（6 項目と未決の点） |
| 稿 | 論文 L のドラフト（2026-11-18 の時点で無い。できたら場所をここに書く） |
| 関 | 関連研究の地図 `/Users/balane/Desktop/MurataLab/Write/Conferences/2027-6-29_PAKDD/docs/LLM_MAP.md`・同フォルダの `survey/ochiai/` |
| wiki | 研究 wiki `/Users/balane/Desktop/MurataLab/LLM-wiki/LLM-wiki/2_wiki/`（読み取り専用。`index.md` から辿る） |
| 設 | 実験の設定の文書（`docs/*_setup.md`） |
| 結 | 実験の結果の報告書（`docs/*_results.md`） |
| 出 | 試行の出力 `outputs/llm_memory_trial/<stem>/`（`summary.csv`・`curve.csv`・`participation.csv`・`*.jsonl`・`*.npz`） |
| 比 | 比較手法（SA・2 段階学習）の結果。DomainTransfer_trial の `src/models/<モデル>/docs/*_results.md`（読むだけ） |
| コ | コード（`src/llm_memory_trial.py`・`src/stula_targets.py`）と docstring、`README.md` |

## 0. 前提・要旨 / Premise and Abstract

| ID | 範囲 | 質問 | 答え方 | 根拠 |
| --- | --- | --- | --- | --- |
| 0-1 | 実験 | この報告は、どの実験のどの時点の結果をまとめたものか？ / What does this report cover? | 対象の試行・LLM・条件・群・人数・種・成果物の日付を表にする。前回の QA から回答が変わった ID を並べる | 設・結・出の更新時刻 |
| 0-2 | 全体 | 論文の要旨を 5 文で書け（背景 → 既存手法の限界 → 提案 → 仕組み → 結果） / Write the abstract in five sentences. | 結果の文には、主指標の値と比較対象を入れる。数値は 6-1・6-6 と同じものを使う | 骨・6 章の回答 |

## 1. はじめに・序論 / Introduction

| ID | 範囲 | 質問 | 答え方 | 根拠 |
| --- | --- | --- | --- | --- |
| 1-1 | 全体 | 本研究の社会的背景は何か？ / What is the social background? | 第 1 文は、読み手の 9 割が賛成する事実から始める。交通需要予測などの応用に限って、エージェントに活動スケジュールを与える流れを書く（「ABSS の入力として必要」とは書かない） | 骨・wiki `queries/motivation-activity-schedule-generation` |
| 1-2 | 全体 | 本研究の target task/problem は何か？ / What target problem does this study address? | 個票を公開していない国について、公表された集計表だけから、人の 1 日として妥当な活動スケジュールを合成する | 骨 |
| 1-3 | 全体 | その target problem のユースケース（具体例）は何か？ / Describe a typical use case. | 日本の合成人口の各エージェントに、群に応じた 1 日の活動スケジュールを付ける例 | 骨、wiki `concepts/transportation-simulation` |
| 1-4 | 全体 | その target problem が難しいと言う根拠は何か？既存手法が誤る例はどんなケースか？ / Why is this task challenging? Describe typical failure cases of existing methods. | 2 つを数値で書く。(1) 集計表は活動の順序・回数・継続時間を決めない。(2) 集計表だけの SA は空白を人手の規則と目的関数で埋めるので、設定しだいで切替回数が 6.6〜25.6 回と動く | 骨、比（SA の報告書） |
| 1-5 | 全体 | 既存手法群には何が不足しているのか？ / Why are conventional studies insufficient? | 既存手法を 1 つだけ引かない。3 つの系統（集計表だけから組み立てる手法群・個票で学ぶ生成モデル群・LLM で人の行動を書く研究群）の不足を抽象化して書く | 骨、関 |
| 1-6 | 全体 | 本研究では何を提案し、何を解決するのか？ / What is proposed and solved in this study? | LLM が 1 人ずつ活動スケジュールを書き、生成済みの集計と集計表の差を文章にしたメモリを次の人に渡して、時刻別行動者率に合わせる手法 | 骨、コ |
| 1-7 | 全体 | 提案手法は既存手法と何が違うのか？主要な違い 1 点のみに絞って述べよ。 / What is the main difference between the proposed and existing methods? | 1 点だけ：集計表が決めない部分（順序・継続時間）の出所を、人手の規則や他国の個票ではなく LLM の知識にする | 骨、2-5 |
| 1-8 | 全体 | 既存手法と違う部分は、なぜ導入するべきなのか？なぜ導入するとうまくいくと予想されるのか？ / Explain why the difference should be introduced. | LLM は人の 1 日のつながり（起床 → 食事 → 通勤など）を知っているので、地域ごとの規則を人手で書かずに済む、という理屈を書く。集計表へ合わせるのはメモリの役目と分けて書く | 骨 |
| 1-9 | 全体 | 本研究の core contribution をまとめよ / List the contributions of this study | 3 項目（定式化・手法・評価）。結果で支えられない項目は ▲ にする | 骨、7-2 |
| 1-10 | 全体 | 【査読がある会議のみ】提案手法全体の構成を eye-catch figure を用いて示せ / Show the eye-catch figure. | Mermaid で「メモリ → LLM が 1 人書く → 確定 → 集計 → メモリ」の輪を描く。変数名はコードと同じにする（`target`・`X_done`・`memory_text`） | コ（`README.md` の図） |

## 2. 関連研究 / Related Work

| ID | 範囲 | 質問 | 答え方 | 根拠 |
| --- | --- | --- | --- | --- |
| 2-1 | 全体 | 論文を複数挙げて、「集計表だけから個人を合成する研究」を説明せよ / Explain the first related subfield and several related papers | IPF・組合せ最適化・SA（Matsumura & Murata SMC 2025）。段落の最後に本研究との違いを書く | 関（`321_STULASchedule.md`）、wiki `concepts/simulated-annealing`・`concepts/IPF` |
| 2-2 | 全体 | 論文を複数挙げて、N 個目の関連分野を説明せよ（この項目を個数分コピーしてください） / Explain the N-th related subfield and several related papers (copy this question for each subfield) | 報告書を書くときに、2 個目以降の関連分野の数だけ 2-2a、2-2b、… と増やして答える。候補は「LLM で人の行動・移動を生成する研究」「LLM で集計量に合う合成データを作る研究」「個票で学ぶ生成モデル」。前回の QA の分け方を引き継ぐ。各回答の最後に本研究との違いを書く。新しさの中心になる分野を最も厚く書く | 関（LLM_MAP の 6 系統）、wiki `papers/` |
| 2-3 | 全体 | この分野で使われるデータ（生活時間調査・交通行動調査）と評価の慣行を説明せよ。標準ベンチマークが無いことをどう補うか？ / Explain the datasets and evaluation practice in the related fields. | 標準ベンチマークは無い。調査ごとのデータと、評価の軸（集計量への適合・個人の 1 日の妥当さ・下流）を書き、本研究がどの軸を測るかを示す。LLM の研究で多い「人が見て妥当か」の評価との関係も書く | 関（LLM_MAP の「実データとどこまで合わせるか」の 5 段）、wiki `concepts/evaluation-metrics` |
| 2-4 | 全体 | 提案手法と類似手法（LLMSynthor・CityReal・MobEvolve・Matsumura らの SA）との違いは何か？ / What is the difference(s) between the proposed and related methods? | 最も近い 2〜3 手法について、近い点と仕組みの違いを 1 つずつ書く。研究群の中の位置は 2-5 に書き、ここでは繰り返さない | 関（R1-018・R1-030・R1-039・321） |
| 2-5 | 全体 | 関連研究群を整理する軸を示し、各研究群と本研究の位置を表（または図）で示せ。本研究の位置がこれまで空いていた理由は何か？ / Organize the related work along explicit axes and show where each group and this study sit. Why has this position been left open? | 軸は書くときに決める（候補：集計表が決めない部分の出所〔人手の規則・他国の個票・LLM の知識〕、合わせる集計量の粒度〔96 スロットか日単位か〕、対象側で使うデータ）。関連研究の章で最初に書き、2-2 の分野の分け方と、1-5・1-7・4-3 の言い方をここから決める。網羅性は主張せず、調べた範囲（件数と出所）を書く。全文を読んでいない論文は表に入れない。空いていた理由が推定なら「推定」と書く | 関、wiki `concepts/synthetic-population`・`concepts/aggregate-supervision` |

## 3. 問題設定 / Problem Statement

| ID | 範囲 | 質問 | 答え方 | 根拠 |
| --- | --- | --- | --- | --- |
| 3-1 | 全体 | 本研究では何というタスクを対象とするか？その内容を説明せよ / What is the target problem of the study? | 対象国の集計表だけを与え、群ごとに指定人数の活動スケジュールを合成する | 骨 |
| 3-2 | 全体 | 対象タスクの望ましい解・出力について説明せよ（何をもって良い解だとするのか） / What is the expected behavior of the system? | 「本タスクでは、…が望ましい」の形で 2 つ：集計量が集計表に合う、1 人分の活動スケジュールが人の 1 日として妥当 | 骨 |
| 3-3 | 全体 | 対象タスクの代表例を示せ（図を付けること） / Explain a typical sample with a figure. | 1 つの群の集計表（時刻別行動者率）と、LLM が書いたその群の活動スケジュール 1 人分 | 出（`*.jsonl`・`*.npz`） |
| 3-4 | 全体 | このタスクで与えられる入力は何か？ / What are the inputs of the task? | 提案手法に限らず、どの手法にも与えられる入力（集計表・群の定義・群ごとの人数） | 骨、コ |
| 3-5 | 全体 | タスクで求められる出力は何か？ / What kind of outputs are expected for the task? | 群ごとの活動スケジュール（20 活動 × 96 スロット、04:00 起点） | コ（`stula_targets.py`） |
| 3-6 | 全体 | 使用する用語を定義せよ / Define the terms used in the paper. | 表にする。論文の用語とコードの変数名の対応も書く（SKILL.md §5） | SKILL.md §5、`Write/Conferences/CLAUDE.md`「本研究の用語」 |
| 3-7 | 全体 | 本研究では何を扱わないか（＝何を前提にしているか） / What is the assumption in the paper? | 平日・全国・20 活動・場所なし・世帯なし・個票なし・オープンウェイトの LLM・英語のプロンプト。個人の 1 日の妥当さを何と比べるか（ATUS の個票を評価だけに使うか）は（未決） | 骨、設 |
| 3-8 | 全体 | タスクの評価尺度は何か？ / Which metric is used? | 軸の名前を挙げる（集計表への適合・較正に使っていない集計表への適合・個人の 1 日の妥当さ）。式は 6-4 に書く。尺度は ABSS での使い道から逆算する方針で（未決） | 骨 |

## 4. 手法 / Methodology

| ID | 範囲 | 質問 | 答え方 | 根拠 |
| --- | --- | --- | --- | --- |
| 4-1 | 全体 | 本研究は何の手法を拡張した何を提案するものか？ Start with "We propose ..." / Which method do you extend? | LLM に合成データを書かせ、集計との差を返して書き直させる方法（LLMSynthor など）を、96 スロットの時刻別行動者率に合わせる活動スケジュールの合成へ拡張する | 関（R1-018） |
| 4-2 | 実験 | 提案手法で行った主たる拡張は、上記の既存手法以外にも広く適用可能であることを説明せよ / Explain that the major extension is widely applicable to other methods. | メモリが LLM に求める条件を書き、LLM を替えた実験（Gemma 4・Qwen3.8 など）と、合わせる集計表を替えた実験の結果で支える。効かなかった LLM と条件も書く | 設・結 |
| 4-3 | 全体 | 提案手法と既存手法の違いを箇条書きせよ / List the differences between the proposed method and the conventional methods | SA・個票で学ぶ生成モデル（2 段階学習を含む）・LLMSynthor との違いを 1 行ずつ | 2-5 |
| 4-4 | 全体 | 提案手法の構造を示せ（図） / Explain the structure of the method. | Mermaid。1 人分の生成からメモリの更新までを描く。変数名はコードと同じにする | コ（`README.md` の図・`generate_group`） |
| 4-5 | 全体 | 提案手法は何個の主要な構成要素を持つか？各々の名称を示せ / How many main components does the proposed method have? | 3 つ：LLM による 1 人分の生成、メモリ（集計の差の文章化）、確定して足す逐次の生成。構成を変えたら 4-8 の枝番も直す | コ |
| 4-6 | 全体 | 入力を数式記号を用いて説明せよ。各入力はそれぞれ何次元か？ / Define the input to the proposed method. | 群 $d$（28 群）、時刻別行動者率 $A^{*}_{d}$（20 × 96）、人数 $N$。日次行動者率（20）は評価だけに使う | コ（`target`・`part`） |
| 4-7 | 全体 | 群・時刻・活動を LLM にどう伝え、LLM の出力をどうスロットに直したか？ / How are the group, time, and activities given to the LLM, and how is the output converted to slots? | 群は英語の文、活動は調査票 A の英語の表記、出力は「活動名と終わりの時刻」の並びを JSON の型で縛る。スロットへの変換の規則を書く | コ（`group_text`・`build_messages`・`segments_to_slots`） |
| 4-8a | 全体 | 構成要素 1（LLM による 1 人分の生成）の motivation・役割・入出力・構造を説明せよ / Explain the motivation, role, input-output, and structure of the first component. | 順序・継続時間を LLM の知識から取る。入力はプロンプト、出力は 1 人分の活動スケジュール | コ（`build_messages`・`Chat`） |
| 4-8b | 全体 | 構成要素 2（メモリ）の motivation・役割・入出力・構造を説明せよ | 生成済みの集計と集計表の差のうち、不足・超過それぞれ大きい箇所を英語の文にする | コ（`memory_text`） |
| 4-8c | 全体 | 構成要素 3（確定して足す逐次の生成）の motivation・役割・入出力・構造を説明せよ | 書いた人は直さずに確定し、次の人のメモリに入れる。warmup の人数まではメモリを渡さない | コ（`generate_group`） |
| 4-9 | 全体 | 生成と集計量を数式で定義せよ。4-8 で定義済みの記号を用いよ / Define the generation and the aggregate. | $x_i \sim \mathrm{LLM}(\cdot \mid d, m_{i-1})$ と、生成済み $i$ 人の時刻別行動者率 $\tilde{A}_{d}$ | コ（`slot_rates`） |
| 4-10 | 全体 | メモリの作り方を数式で定義せよ（差の計算・挙げる箇所の選び方） / Define how the memory is built. | 差 $\tilde{A}_{d} - A^{*}_{d}$ の 1 時間平均、不足・超過それぞれ上位 `top_k` 個、差が `min_gap` 未満は挙げない | コ（`memory_text`・`hourly_mean`） |
| 4-11 | 全体 | プロンプトの全文を示せ / Show the full prompt. | system と user の全文（メモリありの例を 1 つ）。論文では付録に置く | コ（`build_messages`）、出（`*.jsonl`） |

## 5. 実験設定 / Experimental Setup

この章から後ろの回答は過去形で書く。

| ID | 範囲 | 質問 | 答え方 | 根拠 |
| --- | --- | --- | --- | --- |
| 5-2 | 全体 | 何というデータセットを使用したか？なぜそのデータセットを使用したのか？ / Which dataset was used? Why was the dataset chosen? | 社会生活基本調査 2021 の集計表（時刻別行動者率・日次行動者率）。年版と表番号を書く。評価に ATUS を使うならその年版も | コ（`stula_targets.py`）、wiki `concepts/survey-time-use-leisure` |
| 5-3 | 全体 | 集計表を LLM の入出力とどう対応づけたか（活動分類・群・1 日の起点）？ / How were the published tables aligned with the LLM's input and output? | 20 分類と英語の活動名、28 群、4 時起点、年齢区分の扱い | コ、wiki `concepts/activity-schema-alignment` |
| 5-4 | 全体 | データセットと LLM の出力をどのように事前処理したか？ / How were the dataset and the LLM outputs pre-processed? | 集計表を 15 分のスロットに直す規則、LLM の出力をスロットに直す規則と直せなかったときの扱い | コ（`segments_to_slots`） |
| 5-5 | 全体 | データセットの統計情報を示せ（群の数・集計表の大きさ・標本の大きさ・活動の割合） / Explain the statistics of the dataset. | 表にする。加重の有無を書く | コ、設 |
| 5-6 | 全体 | 集計表を、較正に使う表と評価だけに使う表にどう分けたか？各々の大きさを示せ / How were the published tables divided into calibration and evaluation? | 分け方は（未決）。現在の試行は時刻別行動者率をメモリに使い、日次行動者率を評価だけに使う | 設、コ |
| 5-7 | 全体 | 生成の途中で LLM に見せた情報と、最後まで見せなかった情報は何か？ / What information was shown to the LLM during generation, and what was withheld? | プロンプトに入れた量（群・メモリ）と入れなかった量（日次行動者率など）を表にする | コ（`build_messages`） |
| 5-8 | 実験 | 各手法の設定（LLM・量子化・温度・top_p・max_tokens・top_k・min_gap・warmup・種）を表にまとめよ / Present the experimental setup in a table. | 手法ごとに 1 列。共通の値と、手法ごとに違う値を分ける | 設、出 |
| 5-9 | 実験 | 各手法のパラメータ数と、活動スケジュール 1 人分の計算量（トークン数）を示せ / How many parameters does each model have, and how many tokens per schedule? | LLM のパラメータ数と量子化。1 人分の入力・出力トークン数。SA は反復回数 | 設、出（`*.jsonl`）、比 |
| 5-10 | 実験 | 生成に用いたハードウェアとソフトウェアを示せ / Describe the hardware and software used for generation. | 計算機（M5・SQUID）、llama.cpp の版、GGUF の名前 | 設 |
| 5-11 | 実験 | 活動スケジュール 1 人分の生成に要した時間を示せ / How long did it take to generate one schedule? | 条件ごとの 1 人分の時間と、群あたりの合計。SA の時間と並べる | 出、比 |
| 5-12 | 実験 | 手法ごとに、構築に使うデータと、評価だけに使うデータを表にせよ / Which data does each method use for construction, and which are used only for evaluation? | 評価に使う量を目的関数やメモリに入れている手法があれば明記する（SA の日次行動者率、2 段階学習の他国の個票） | 設、比 |
| 5-13 | 実験 | 比較手法をどのように実装し、原論文から何を変えたか？変えた理由は何か？ / How were the baselines implemented, and what was changed from the original papers? | 変更点を表にする。比較手法を弱く作っていないことの根拠（SA の収束の確認）を書く | 比（SA_setup・SA の報告書） |
| 5-14 | 全体 | LLM が事前学習で集計表や調査の記述を見ている可能性を、どう扱ったか？ / How was possible contamination of the LLM's pre-training data handled? | 集計表の値をプロンプトなしで答えさせる確認など、確かめた方法を書く。確かめていなければ ▲ | 設、結 |

## 6. 実験結果 / Experimental Results

| ID | 範囲 | 質問 | 答え方 | 根拠 |
| --- | --- | --- | --- | --- |
| 6-1 | 実験 | ベースライン手法との定量的比較結果を示せ / Show the quantitative comparison results with the baseline method | 表の行と列の意味、数値が平均か中央値か範囲か、種の数、群あたりの人数を書く。標本誤差の床の値を基準の列に置く | 出（`summary.csv`）、比 |
| 6-2 | 実験 | 何をベースライン手法（群）としたか？ / What was used as the baseline method(s)? | (a) 集計表だけの SA、(b) メモリなし（none）。2 段階学習を入れるかは（未決） | 設 |
| 6-3 | 実験 | 上記ベースラインを選んだ理由を説明せよ / Explain the reason for choosing the above baseline | (a) は個票を使わない従来手法、(b) はメモリの効果を測る基準 | 設、骨 |
| 6-4 | 全体 | 評価尺度（群）について数式で説明せよ。どれが主要尺度か？ / Explain the metric(s) by using equations. Which one is the primary metric? | 主要尺度は（未決）。現在の試行は時刻別行動者率の MAE（pt）。加重は尺度ごとに書く | コ（`evaluate`） |
| 6-5 | 全体 | なぜそれらの評価尺度を使用したのか？他の評価尺度ではダメなのか？ / Why did you use those evaluation metrics? Why not other metrics? | ABSS が活動スケジュールから何を読むか（ある時刻の状態・1 人の 1 日のつながり など）から理由を書く | 骨、wiki `concepts/evaluation-metrics` |
| 6-6 | 実験 | ベースラインと提案手法の性能を（相対的な性能差ではなく）絶対的な値で示せ / Show the performance in absolute values | 同じ文型を繰り返さない | 出（`summary.csv`） |
| 6-7 | 実験 | 差は、種によるばらつきの範囲を超えているか？ / Do the differences exceed the variation across random seeds? | 種の数と [最小, 最大] を書く。検定をしていなければ「有意」と書かず、範囲が重ならないことを「分離」と書く。種 1 つなら ▲ | 出（種ごと） |
| 6-8 | 実験 | 定性的結果：提案手法がうまくいった例を示せ。集計表・ベースライン・提案手法の値をそれぞれ示せ / Qualitative results: show successful cases. | 第 1 文は「表（図）に…を示す」。結果の説明だけで終えず、どの点が適切かとその根拠を書く | 出（`curve.csv`・`*.jsonl`）、図 |
| 6-9 | 実験 | 定性的結果：提案手法がうまくいかなかった例を示せ。なぜうまくいかなかったのか？ / Qualitative results: show failure cases. Why did they fail? | 6-8 より少ない数。原因が推定なら「推定」と書く | 結、出 |
| 6-10 | 実験 | Ablation study において、何のために何を取り除いたかを説明せよ。取り除いた要素が有効だったことを示せ / Explain what is ablated | 取り除いたモデルを Model (i)、(ii)、… と呼ぶ（メモリなし・warmup なし など）。条件には (i) を付けない。未実行なら ▲ | 設、結 |
| 6-11 | 実験 | 誤差の内訳を、活動ごと・群ごとに示せ / Break down the error by activity and by group. | どの活動・どの群で誤差が残るかを表にする | 出（`*.npz`・`participation.csv`） |
| 6-12 | 実験 | メモリを入れて集計表との誤差が増えた群・活動は、いくつあったか？ / How many groups and activities became worse with the memory? | 群・20 活動のうちの数を書く。「多すぎる」活動が減ったかを分けて書く | 出、結 |
| 6-13 | 実験 | うまくいかなかった例を型に分類せよ。各型の定義を示せ / Categorize the failure cases. Show the definition of each failure mode. | 型の定義と、どの条件で起きたかを表にする（例：全員に同じ活動を詰め込む、超過を減らさない、途中で下げ止まる） | 結 |
| 6-14 | 実験 | うまくいかなかった主な要因（main bottleneck）と possible solution について説明せよ / Explain the main bottleneck and the possible solution | 確かめた原因と推定を分ける | 結 |
| 6-15 | 実験 | メモリに入れていない量（日次行動者率・平均時間・切替回数・個人の 1 日の妥当さ）でも、主張は支えられるか？ / Is the claim supported by quantities not used in the memory? | SA は時刻別行動者率に構成上一致するので、その表への適合だけでは比べられない。メモリにも目的関数にも入れていない量を並べる | 出（`participation.csv`）、比 |
| 6-16 | 実験 | 結果を見る前に決めた読み方と予測は何か？予測は当たったか？ / What were the pre-registered rules and predictions, and did they hold? | 予測と結果を 1 行ずつ対にする。結果を見てから決めた基準は使わない | 設（回す前の予測）、結 §2 |
| 6-17 | 実験 | LLM の出力が形式に合わなかった割合と、その扱いはどうだったか？ / How often did the LLM output fail the format, and how was it handled? | 条件ごとの失敗率・再試行の回数・修復（`repair`）した数。捨てた出力が結果を偏らせていないかを書く | 出（`*.jsonl`）、コ（`max_tries`） |

## 7. 結論 / Conclusion

| ID | 範囲 | 質問 | 答え方 | 根拠 |
| --- | --- | --- | --- | --- |
| 7-1 | 全体 | 本研究ではどのようなタスクを扱ったか？ / What kind of task was addressed in this study? | 1〜2 文 | 3-1 |
| 7-2 | 実験 | 本研究の貢献（contributions）を過去形で書け / Summarize the contributions of this study in the past tense | 今回の結果で言えることだけを書く。暫定の結果には「暫定」と付ける | 6 章の回答 |
| 7-3 | 実験 | 将来研究は何か？ / What is the future study? | 今回の実験で得た結果に基づく課題に限る | 結 |
| 7-4 | 実験 | この結果から言えないこと・限界は何か？ / What are the limitations? | 事実・推定・言えないことを分ける | 結 §4 |
| 7-5 | 実験 | 次に決めること・次に回す実験は何か？ / What needs to be decided or run next? | 論点・選択肢・コストの表。▲ の質問を埋めるのに要る実験を対応づける | 結 §5、この QA の ▲ |
