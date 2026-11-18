# 質問一覧（本研究用に改良した 60QA）

- **版：** 2026-11-16
- **元：** 慶應義塾大学 杉浦孔明研究室「60 Questions & Answers 公開版」（7 章 60 問）
- **本研究用：** 8 章 71 問。元の 60 問のうち 45 問をそのまま残し、14 問の文言を変え、1 問を削り、9 問を足した。4-8 は構成要素ごとに a〜d に分けたので、ID は 71 個になる。2-2 は元のシートのまま 1 問にしてあり、報告書を書くときに関連分野の数だけ 2-2a、2-2b、… と増やす。変更の理由は [changes-from-original.md](changes-from-original.md)
- **ID：** 元のシートと同じ番号を使う。足した質問は各章の末尾の番号にした

## 列の読み方

| 列 | 意味 |
| --- | --- |
| 範囲 | **全体** ＝ 研究全体で 1 つに決まる回答。前回の QA から引き継ぎ、根拠が変わったときだけ直す。**実験** ＝ 報告する実験の結果で毎回書き直す回答 |
| 答え方 | この研究で何を書けば回答になるか |
| 根拠 | 回答の材料がある場所（下の略号） |

| 略号 | 場所 |
| --- | --- |
| 稿 | 論文のドラフト `/Users/balane/Desktop/MurataLab/Write/Conferences/2027-6-29_PAKDD/main.tex`（本文とコメント） |
| 関 | 同フォルダの `docs/RELATED_WORK_CANDIDATES.md`・`survey/ochiai/` |
| wiki | 研究 wiki `/Users/balane/Desktop/MurataLab/LLM-wiki/LLM-wiki/2_wiki/`（読み取り専用。`index.md` から辿る） |
| 設 | 実験の設定の文書（`src/models/<モデル>/docs/*_setup.md`） |
| 結 | 実験の結果の報告書（同 `*_results.md`） |
| CSV | 採点 CSV（`data/processed/aggregates/`。`experiment-report` の `report_numbers.py` で読む） |
| 履 | Stage 2 の記録（`outputs/checkpoints/<stem>/history.csv`） |
| コ | コード（`model.py`・`stage2_common.py`・`sa_core.py` など）と docstring |

## 0. 前提・要旨 / Premise and Abstract

| ID | 範囲 | 質問 | 答え方 | 根拠 |
| --- | --- | --- | --- | --- |
| 0-1 | 実験 | この報告は、どの実験のどの時点の結果をまとめたものか？ / What does this report cover? | 対象の実験・比べた手法・種の数・成果物の日付を表にする。前回の QA から回答が変わった ID を並べる | 設・結・CSV の更新時刻 |
| 0-2 | 全体 | 論文の要旨を 5 文で書け（背景 → 既存手法の限界 → 提案 → 仕組み → 結果） / Write the abstract in five sentences. | 結果の文には、主指標の値と比較対象を入れる。数値は 6-1・6-6 と同じものを使う | 稿（概要）・6 章の回答 |

## 1. はじめに・序論 / Introduction

| ID | 範囲 | 質問 | 答え方 | 根拠 |
| --- | --- | --- | --- | --- |
| 1-1 | 全体 | 本研究の社会的背景は何か？ / What is the social background? | 第 1 文は、読み手の 9 割が賛成する事実から始める。交通需要予測などの応用に限って、エージェントに活動スケジュールを与える流れを書く（「ABSS の入力として必要」とは書かない） | 稿 §1・wiki `queries/motivation-activity-schedule-generation` |
| 1-2 | 全体 | 本研究の target task/problem は何か？ / What target problem does this study address? | 個票を公開していない国について、公表された集計表に合う活動スケジュールを合成する | 稿 §1 |
| 1-3 | 全体 | その target problem のユースケース（具体例）は何か？ / Describe a typical use case. | 日本の合成人口の各エージェントに、群に応じた 1 日の活動スケジュールを付ける例 | 稿 §1・§5、wiki `concepts/transportation-simulation` |
| 1-4 | 全体 | その target problem が難しいと言う根拠は何か？既存手法が誤る例はどんなケースか？ / Why is this task challenging? Describe typical failure cases of existing methods. | 2 つを数値で書く。(1) 集計表は活動の順序と継続時間を決めない。(2) 他国の個票で学んだ生成器はそのままでは合わない（例：12 時の食事） | 稿 §1・§3.1、結 |
| 1-5 | 全体 | 既存手法群には何が不足しているのか？ / Why are conventional studies insufficient? | 既存手法を 1 つだけ引かない。2 つの系統（集計表だけから組み立てる手法群・個票で学ぶ生成モデル群）の不足を抽象化して書く | 稿 §1・§2、関 |
| 1-6 | 全体 | 本研究では何を提案し、何を解決するのか？ / What is proposed and solved in this study? | 1 日の構造を他国の個票から、活動の時刻を対象国の集計表から取る 2 段階の学習 | 稿 §1 |
| 1-7 | 全体 | 提案手法は既存手法と何が違うのか？主要な違い 1 点のみに絞って述べよ。 / What is the main difference between the proposed and existing methods? | 1 点だけ：対象国の個票を使わず、対象国の集計表だけで第二段階の学習を行う | 稿 §2（表 1） |
| 1-8 | 全体 | 既存手法と違う部分は、なぜ導入するべきなのか？なぜ導入するとうまくいくと予想されるのか？ / Explain why the difference should be introduced. | 集計表が縛らない量（順序・継続時間）を、事前学習損失で保つという理屈を書く | 稿 §3.1・§3.3 |
| 1-9 | 全体 | 本研究の core contribution をまとめよ / List the contributions of this study | 3 項目（定式化・手法・評価）。結果で支えられない項目は ▲ にする | 稿 §1、7-2 |
| 1-10 | 全体 | 【査読がある会議のみ】提案手法全体の構成を eye-catch figure を用いて示せ / Show the eye-catch figure. | Mermaid で Stage 1 と Stage 2 の流れを描く。変数名はコードと同じにする | 稿 Fig. 1 の計画、設 |

## 2. 関連研究 / Related Work

| ID | 範囲 | 質問 | 答え方 | 根拠 |
| --- | --- | --- | --- | --- |
| 2-1 | 全体 | 論文を複数挙げて、「集計表だけから個人を合成する研究」を説明せよ / Explain the first related subfield and several related papers | IPF・組合せ最適化・SA。段落の最後に本研究との違いを書く | 稿 §2、wiki `concepts/simulated-annealing`・`concepts/IPF` |
| 2-2 | 全体 | 論文を複数挙げて、N 個目の関連分野を説明せよ（この項目を個数分コピーしてください） / Explain the N-th related subfield and several related papers (copy this question for each subfield) | 報告書を書くときに、2 個目以降の関連分野の数だけ 2-2a、2-2b、… と増やして答える。分野は論文のドラフトの関連研究の段落と wiki から決め、前回の QA の分け方を引き継ぐ。使ったデータは原論文の記述に基づき、各回答の最後に本研究との違いを書く。新しさの中心になる分野を最も厚く書く | 稿 §2、関、wiki `papers/`・`concepts/aggregate-supervision`・`concepts/reward-fine-tuning` |
| 2-3 | 全体 | この分野で使われるデータ（生活時間調査・交通行動調査）と評価の慣行を説明せよ。標準ベンチマークが無いことをどう補うか？ / Explain the datasets and evaluation practice in the related fields. | 標準ベンチマークは無い。調査ごとのデータと、評価の軸（集計量への適合・個人の 1 日の妥当さ・下流）を書き、本研究がどの軸を測るかを示す | wiki `concepts/evaluation-metrics`・`papers/kapp-2023-genmobility-survey` |
| 2-4 | 全体 | 提案手法と類似手法（ATLAS・DeepAM・Matsumura らの SA）との違いは何か？ / What is the difference(s) between the proposed and related methods? | 最も近い 2〜3 手法について、近い点と仕組みの違いを 1 つずつ書く。研究群の中の位置は 2-5 に書き、ここでは繰り返さない | 稿 §2、関、wiki `papers/` |
| 2-5 | 全体 | 関連研究群を整理する軸を示し、各研究群と本研究の位置を表（または図）で示せ。本研究の位置がこれまで空いていた理由は何か？ / Organize the related work along explicit axes and show where each group and this study sit. Why has this position been left open? | 軸は書くときに決める（候補：対象側で使うデータ、他国・他地域のデータを使うか、1 日の構造の出所、合わせる集計量の粒度）。関連研究の章で最初に書き、2-2 の分野の分け方と、1-5・1-7・4-3 の言い方をここから決める。網羅性は主張せず、調べた範囲（件数と出所）を書く。全文を読んでいない論文は表に入れない。空いていた理由が推定なら「推定」と書く | 関、稿 表 1、wiki `concepts/synthetic-population`・`concepts/aggregate-supervision` |

## 3. 問題設定 / Problem Statement

| ID | 範囲 | 質問 | 答え方 | 根拠 |
| --- | --- | --- | --- | --- |
| 3-1 | 全体 | 本研究では何というタスクを対象とするか？その内容を説明せよ / What is the target problem of the study? | 他国の個票と対象国の集計表から、群を条件とする生成器を学習する | 稿 §3.1 |
| 3-2 | 全体 | 対象タスクの望ましい解・出力について説明せよ（何をもって良い解だとするのか） / What is the expected behavior of the system? | 「本タスクでは、…が望ましい」の形で 2 つ：集計量が集計表に合う、1 人分の活動スケジュールが人の 1 日として妥当 | 稿 §3.1 |
| 3-3 | 全体 | 対象タスクの代表例を示せ（図を付けること） / Explain a typical sample with a figure. | 1 つの群の集計表（時刻別行動者率）と、その群の活動スケジュール 1 人分。既存の図があればそれを使う | `src/models/LSTM_Aggregate/figures/` |
| 3-4 | 全体 | このタスクで与えられる入力は何か？ / What are the inputs of the task? | 提案手法に限らず、どの手法にも与えられる入力（個票の集合・集計表・群の定義） | 稿 §3.1 |
| 3-5 | 全体 | タスクで求められる出力は何か？ / What kind of outputs are expected for the task? | 群ごとの活動スケジュール（12 活動 × 96 スロット） | 稿 §3.1 |
| 3-6 | 全体 | 使用する用語を定義せよ / Define the terms used in the paper. | 表にする。論文の用語とコードの変数名の対応も書く（SKILL.md §5） | `Write/Conferences/CLAUDE.md`「本研究の用語」 |
| 3-7 | 全体 | 本研究では何を扱わないか（＝何を前提にしているか） / What is the assumption in the paper? | 平日・全国・12 活動・場所なし・対象国の個票なし。個人の 1 日の妥当さは米国の個票と比べる | 稿 §4.1、設 |
| 3-8 | 全体 | タスクの評価尺度は何か？ / Which metric is used? | 4 つの軸の名前を挙げる。式は 6-4 に書く | 稿 §4.1（Metrics） |

## 4. 手法 / Methodology

| ID | 範囲 | 質問 | 答え方 | 根拠 |
| --- | --- | --- | --- | --- |
| 4-1 | 全体 | 本研究は何の手法を拡張した何を提案するものか？ Start with "We propose ..." / Which method do you extend? | 事前学習した生成器を集計量の誤差で更新する方法（ATLAS など）を、他国の個票と対象国の集計表の組合せへ拡張する | 稿 §3、wiki `concepts/reward-fine-tuning` |
| 4-2 | 実験 | 提案手法で行った主たる拡張は、上記の既存手法以外にも広く適用可能であることを説明せよ / Explain that the major extension is widely applicable to other methods. | 第二段階が生成器に求める条件を書き、生成器を替えた実験の結果で支える。効かなかった生成器と条件も書く | 設・結（Stage2_Common） |
| 4-3 | 全体 | 提案手法と既存手法の違いを箇条書きせよ / List the differences between the proposed method and the conventional methods | SA・個票で学ぶ生成モデル・ATLAS との違いを 1 行ずつ | 稿 §2 |
| 4-4 | 全体 | 提案手法のモデル構造を示せ（図） / Explain the structure of the model. | Mermaid。変数名はコードと同じにする | コ（`model.py`） |
| 4-5 | 全体 | 提案手法は何個の主要な構成要素（段階・損失）を持つか？各々の名称を示せ / How many main components does the proposed method have? | 4 つ：事前学習、集計損失、サンプリングの連鎖を通る勾配、事前学習損失 | 稿 §3 |
| 4-6 | 全体 | 入力を数式記号を用いて説明せよ。各入力はそれぞれ何次元か？ / Define the input to the proposed method. | $x$（96 スロット、12 活動）、$d$（性別 2・年齢 7・就業状態 2）、$A^{*}$（28 × 12 × 96） | 稿 §3.1 |
| 4-7 | 全体 | 条件（群）と時刻を、どのように符号化したか？ / Explain how the group and the slot index are encoded. | 社会属性ごとの埋め込みを連結して線形変換。時刻は学習型の埋め込み | コ（COND_SPEC）、設 §2.2 |
| 4-8a | 全体 | 構成要素 1（事前学習）の motivation・役割・入出力・構造を説明せよ / Explain the motivation, role, input-output, and structure of the first component. | 順序と継続時間を他国の個票から学ぶ | 稿 §3.2 |
| 4-8b | 全体 | 構成要素 2（集計損失）の motivation・役割・入出力・構造を説明せよ | 群ごとに生成して集計し、集計表との二乗誤差を取る | 稿 §3.3 |
| 4-8c | 全体 | 構成要素 3（サンプリングの連鎖を通る勾配）の motivation・役割・入出力・構造を説明せよ | 前向きは実際の標本、後ろ向きは連続な近似の微分 | 稿 §3.3、設 §2.1 |
| 4-8d | 全体 | 構成要素 4（事前学習損失）の motivation・役割・入出力・構造を説明せよ | 集計損失が縛らない量を保つ | 稿 §3.3 |
| 4-9 | 全体 | 生成（サンプリング）と集計量を数式で定義せよ。4-8 で定義済みの記号を用いよ / Define the generation and the aggregate. | $p_{\theta}(x \mid d)$ と $\tilde{A}_{d,c,s}$ | 稿 式 (2)・(4) |
| 4-10 | 全体 | 損失関数の定義を示せ / Explain the loss function | $L = L_{\mathrm{agg}} + \lambda L_{\mathrm{pre}}$ | 稿 式 (3)・(5)・(7) |

## 5. 実験設定 / Experimental Setup

この章から後ろの回答は過去形で書く。

| ID | 範囲 | 質問 | 答え方 | 根拠 |
| --- | --- | --- | --- | --- |
| 5-2 | 全体 | 何というデータセットを使用したか？なぜそのデータセットを使用したのか？ / Which dataset was used? Why was the dataset chosen? | 米国生活時間調査（ATUS）2024 の個票と、社会生活基本調査 2021 の集計表。年版と表番号を書く | 稿 §4.1、wiki `concepts/ATUS`・`concepts/survey-time-use-leisure` |
| 5-3 | 全体 | 2 つの調査をどのように対応づけたか（活動分類・群・1 日の起点・教師の作り方）？ / How were the two surveys harmonized? | 17 → 12 分類と 20 → 12 分類、28 群、4 時起点、年齢 15 → 7 区分の加重平均 | 稿 §4.1、設、wiki `concepts/activity-schema-alignment` |
| 5-4 | 全体 | データセットをどのように事前処理したか？ / How was the dataset pre-processed? | 個票を 15 分のスロットに変換する規則、除外した記録 | コ（`src/common/preprocess/`） |
| 5-5 | 全体 | データセットの統計情報を示せ（回答者数・群ごとの人数・活動の割合・1 日の切り替え回数・集計表の標本の大きさ） / Explain the statistics of the dataset. | 表にする。加重の有無を書く | 設、結 |
| 5-6 | 全体 | 訓練集合・検証集合・テスト集合をどのように分割したか？各々のサイズを示せ / How was the dataset divided? | 個票は訓練と検証。日本側は、学習に使う集計表と評価だけに使う集計表に分ける | 設 §1 |
| 5-7 | 全体 | 訓練集合・検証集合・テスト集合を各々どのように使用したか？ / How was each set used? | 検証は第一段階の早期終了だけ。第二段階は最終 step を使う | 設 §2 |
| 5-8 | 実験 | 各手法の設定（最適化手法・学習率・更新回数・ハイパーパラメータ・種）を表にまとめよ / Present the experimental setup in a table. | 手法ごとに 1 列。共通の値と、手法ごとに違う値を分ける | 設 |
| 5-9 | 実験 | 各手法の学習可能パラメータ数と、生成 1 回の計算量を示せ / How many trainable parameters does each model have? | 生成に使う部分のパラメータ数。FLOPs を測っていなければ ▲ にして、代わりの量（逆過程のステップ数など）を書く | 設 §3.1 |
| 5-10 | 実験 | 訓練に用いたハードウェア構成を示せ / Describe the specifications of the machine used in the experiment. | 手法ごとの計算場所 | 設、結 |
| 5-11 | 実験 | 訓練に要した時間を示せ。活動スケジュール 1 人分の生成に要した時間を示せ / How long did it take for training and for generating one schedule? | 第一段階・第二段階・SA を分ける。測っていない量は ▲ | 履（`sec` 列）、結 |
| 5-12 | 実験 | 手法ごとに、学習・構築に使うデータと、評価だけに使うデータを表にせよ / Which data does each method use for training, and which are used only for evaluation? | 評価に使う量を目的関数に入れている手法があれば明記する（SA の日次行動者率） | 設、稿 §4.1 |
| 5-13 | 実験 | 比較手法をどのように実装し、原論文から何を変えたか？変えた理由は何か？ / How were the baselines implemented, and what was changed from the original papers? | 変更点を表にする。比較手法を弱く作っていないことの根拠（収束の確認）を書く | 設（SA_setup）、結 |

## 6. 実験結果 / Experimental Results

| ID | 範囲 | 質問 | 答え方 | 根拠 |
| --- | --- | --- | --- | --- |
| 6-1 | 実験 | ベースライン手法との定量的比較結果を示せ / Show the quantitative comparison results with the baseline method | 表の行と列の意味、数値が平均か中央値か範囲か、種の数を書く。米国の個票の値を基準の列に置く | CSV、結 |
| 6-2 | 実験 | 何をベースライン手法（群）としたか？ / What was used as the baseline method(s)? | (a) 集計表だけの SA、(b) 事前学習のみ（Pre-trained） | 設 |
| 6-3 | 実験 | 上記ベースラインを選んだ理由を説明せよ / Explain the reason for choosing the above baseline | (a) は個票を使わない従来手法、(b) は第二段階の学習の効果を測る基準 | 設、稿 §4.1 |
| 6-4 | 全体 | 評価尺度（群）について数式で説明せよ。どれが主要尺度か？ / Explain the metric(s) by using equations. Which one is the primary metric? | 主要尺度は集計表との二乗誤差。加重は尺度ごとに書く | 稿 §4.1、コ（`gs2.score_pool`） |
| 6-5 | 全体 | なぜそれらの評価尺度を使用したのか？他の評価尺度ではダメなのか？ / Why did you use those evaluation metrics? Why not other metrics? | 絶対誤差の平均は平滑化で下がるので主要尺度にしない、などの理由 | 結、wiki `concepts/evaluation-metrics` |
| 6-6 | 実験 | ベースラインと提案手法の性能を（相対的な性能差ではなく）絶対的な値で示せ / Show the performance in absolute values | 同じ文型を繰り返さない | CSV |
| 6-7 | 実験 | 差は、種によるばらつきの範囲を超えているか？ / Do the differences exceed the variation across random seeds? | 種の数と [最小, 最大] を書く。検定をしていなければ「有意」と書かず、範囲が重ならないことを「分離」と書く。種 1 つなら ▲ | CSV（種ごと） |
| 6-8 | 実験 | 定性的結果：提案手法がうまくいった例を示せ。集計表・ベースライン・提案手法の値をそれぞれ示せ / Qualitative results: show successful cases. | 第 1 文は「表（図）に…を示す」。結果の説明だけで終えず、どの点が適切かとその根拠を書く | 生成物の集計量（`*_rates.npz`）、図 |
| 6-9 | 実験 | 定性的結果：提案手法がうまくいかなかった例を示せ。なぜうまくいかなかったのか？ / Qualitative results: show failure cases. Why did they fail? | 6-8 より少ない数。原因が推定なら「推定」と書く | 結 |
| 6-10 | 実験 | Ablation study において、何のために何を取り除いたかを説明せよ。取り除いた要素が有効だったことを示せ / Explain what is ablated | 取り除いたモデルを Model (i)、(ii)、… と呼ぶ。条件には (i) を付けない。未実行なら ▲ | 設、結 |
| 6-11 | 実験 | 誤差の内訳を、活動ごと・群ごとに示せ / Break down the error by activity and by group. | 混同行列の代わり。どの活動・どの群で誤差が残るかを表にする | `*_rates.npz`、結 |
| 6-12 | 実験 | 第二段階の学習で集計表との誤差が増えた群・活動は、いくつあったか？ / How many groups and activities became worse after fine-tuning? | 28 群・12 活動のうちの数を手法ごとに書く | `*_rates.npz`、結 |
| 6-13 | 実験 | うまくいかなかった例を型に分類せよ。各型の定義を示せ / Categorize the failure cases. Show the definition of each failure mode. | 型の定義と、どの手法で起きたかを表にする | 結 |
| 6-14 | 実験 | うまくいかなかった主な要因（main bottleneck）と possible solution について説明せよ / Explain the main bottleneck and the possible solution | 確かめた原因と推定を分ける | 結 |
| 6-15 | 実験 | 学習に使っていない量（教師から外した群・別の集計表・個人の 1 日の妥当さ）でも、主張は支えられるか？ / Is the claim supported by quantities not used in training? | SA は集計表に構成上一致するので、教師への適合だけでは比べられない。学習に使っていない量を並べる | CSV、稿 §4.2 |
| 6-16 | 実験 | 結果を見る前に決めた読み方と予測は何か？予測は当たったか？ / What were the pre-registered rules and predictions, and did they hold? | 予測と結果を 1 行ずつ対にする。結果を見てから決めた基準は使わない | 設（回す前の予測）、結 §2 |

## 7. 結論 / Conclusion

| ID | 範囲 | 質問 | 答え方 | 根拠 |
| --- | --- | --- | --- | --- |
| 7-1 | 全体 | 本研究ではどのようなタスクを扱ったか？ / What kind of task was addressed in this study? | 1〜2 文 | 3-1 |
| 7-2 | 実験 | 本研究の貢献（contributions）を過去形で書け / Summarize the contributions of this study in the past tense | 今回の結果で言えることだけを書く。暫定の結果には「暫定」と付ける | 6 章の回答 |
| 7-3 | 実験 | 将来研究は何か？ / What is the future study? | 今回の実験で得た結果に基づく課題に限る | 結 |
| 7-4 | 実験 | この結果から言えないこと・限界は何か？ / What are the limitations? | 事実・推定・言えないことを分ける | 結 §4 |
| 7-5 | 実験 | 次に決めること・次に回す実験は何か？ / What needs to be decided or run next? | 論点・選択肢・コストの表。▲ の質問を埋めるのに要る実験を対応づける | 結 §5、この QA の ▲ |
