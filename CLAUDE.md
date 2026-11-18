# CLAUDE.md

## このリポジトリ
LLM（オープンウェイト）に活動スケジュールを書かせ、社会生活基本調査の集計表に合わせる研究（論文 L）。
2 段階学習の研究（論文 P・S）は別のリポジトリ `/Users/balane/Desktop/MurataLab/Repos/DomainTransfer_trial/` にある。

- `src/stula_targets.py` は DomainTransfer_trial の `src/models/SA_Aggregate/sa_targets.py` の写し。
  読み込みを変えたら、テスト g（写し元と同じ値か）を通す
- SA・Pre-trained・Fine-tuned と比べるときは、それぞれの生成物（npz）を読み込み、このリポジトリの同じ評価コードで採点する。
  評価のコードを 2 つのリポジトリに分けて持たない
- LLM へのプロンプトは英語で書く

## 知識ベース連携（LLM-wiki vault を参照源にする）

**概念・研究・手法・評価指標・関連研究に関わる質問**には、
ローカルの研究wiki（Obsidian vault）を参照源（RAG）として回答する。
必要な場合は、wiki外情報も参照する。

- Vault root: `/Users/balane/Desktop/MurataLab/LLM-wiki/LLM-wiki/`
- カタログ: `/Users/balane/Desktop/MurataLab/LLM-wiki/LLM-wiki/2_wiki/index.md`
- スキーマ: `/Users/balane/Desktop/MurataLab/LLM-wiki/CLAUDE.md`（vault側の規約）
- LLM と人の行動シミュレーションの関連研究の地図: `/Users/balane/Desktop/MurataLab/Write/Conferences/2027-6-29_PAKDD/docs/LLM_MAP.md`

### いつ参照するか
- **自動で引く**: 評価指標、LLM エージェント・LLM による合成データ、合成人口、活動スケジュール表現、
  関連研究・先行手法など「知識を問う」質問。
- **引かない**: 単純なコード編集・実行・デバッグなど、wikiの知見が関係しない作業。

### 参照プロトコル
1. まず `2_wiki/index.md` を読み、関連する概念/論文ページを特定する。
2. `[[wikilink]]` を辿って近傍ページに drill-in。概念ページの「横断的知見」「未解決の問い」を確認。
   必要なら `0_raw/<topic>/` の元PDFも参照する。
3. **vault内容を根拠に、ページ名を明示して引用**しつつ回答する。
4. vaultに十分な根拠がなければ率直に述べ、補う場合は「wiki外の情報」と明記する。

### 重要な制約
- **vaultは読み取り専用**として扱う。このリポジトリからは編集・ingestしない。
- 引用の捏造禁止。根拠のない断定をしない。

## 目標
ハイレベル国際学会へ論文を通すことを目指す。
llm-wiki vault の Conference（一覧）に目指す学会の一覧がある。論文を通すために必要な水準を意識する。

## データについて
- このリポジトリは公表統計（集計表）だけを使う。個票は読まない
- `data/` と `outputs/` は git で管理しない（`.gitignore` 済み）
- DomainTransfer_trial の `data/raw/protected/` : アクセス禁止（読み取り・編集とも不可）
- DomainTransfer_trial の `data/raw/open/` : アクセス可（読み取り・参照を許可）

## Compact instructions
When you are using compact, please focus on test output and code changes

## 表現
- 曖昧な表現は避け、明確に書くこと
- 単語や表現は一意に定めること
- わかりにくい、誤解を招きやすい表現は避けること
- 日本語でmdファイルなどの文章を書くときは、natural-japaneseのskillを使うこと

## コーディング
- コードは、理解しやすい形で実装する
- コードの可読性を意識すること
- Pylance等の型チェックによる警告が出ないようにすること（`pyrightconfig.json` で `src` を検索先に入れてある）
- 編集者が全ての実装を理解できるようにすること
- Mermaidの図を用いて、
  - データフローや処理の流れを図示すること
  - 図の中で使用する変数名はコード中の変数名と一致させること
  - モデルの構造や処理の流れを図示すること
- docstringはGoogle styleで記述すること
