<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../img/synalinks-dark.svg">
  <img height=200 alt="Synalinks" src="../img/synalinks-light.svg">
</picture>
</div>

<div align="center">

<b>アイデアから本番環境まで、わずか数行で</b>

<em>Keras のシンプルさとディープラーニングのベストプラクティスの厳密さを活かした、初のニューロシンボリック言語モデル（LM）フレームワーク。</em>

<b>[RAG](https://synalinks.github.io/synalinks/guides/Knowledge%20Base/)、[ツールを使うエージェント](https://synalinks.github.io/synalinks/guides/Agents/)、マルチエージェントシステム、[再帰的エージェント](https://synalinks.github.io/synalinks/guides/Recursive%20Language%20Model%20Agent/)などを、わずか数行で構築</b>

[Deutsch](README_de.md) | 
[English](../README.md) | 
[Español](README_es.md) | 
[Français](README_fr.md) | 
[Italiano](README_it.md) | 
[日本語](README_ja.md) | 
[한국어](README_ko.md) | 
[Português](README_pt.md) | 
[Русский](README_ru.md) | 
[中文](README_zh.md)

<p align="center">
  <a href="https://synalinks.github.io/synalinks" target="_blank"><strong>ドキュメント</strong></a> ·
  <a href="https://synalinks.github.io/synalinks/FAQ/" target="_blank"><strong>FAQ</strong></a> ·
  <a href="https://discord.gg/82nt97uXcM" target="_blank"><strong>Discord</strong></a> ·
  <a href="https://github.com/SynaLinks/synalinks/tree/main/examples" target="_blank"><strong>コード例</strong></a> .
  <a href="https://github.com/SynaLinks/synalinks/tree/main/guides" target="_blank"><strong>ガイド</strong></a>
</p>

</div>

<div align="center">

Synalinks が役に立ったら、ぜひリポジトリにスターを付けてください！より多くの AI/ML エンジニアに届け、コミュニティを成長させる助けになります。

![Beta](https://img.shields.io/badge/Release-Beta-blue.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Coverage Badge](https://raw.githubusercontent.com/SynaLinks/synalinks/refs/heads/main/coverage-badge.svg)
[![Downloads](https://static.pepy.tech/badge/synalinks)](https://pepy.tech/project/synalinks)
[![Discord](https://img.shields.io/discord/1118241178723291219)](https://discord.gg/82nt97uXcM)
[![Python package](https://github.com/SynaLinks/Synalinks/actions/workflows/tests.yml/badge.svg)](https://github.com/SynaLinks/SynaLinks/actions/workflows/tests.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/license/apache-2-0)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/SynaLinks/synalinks)

</div>

<div align="center">

お手持ちのコーディングエージェント（Claude Code、Cursor、Copilot など）で Synalinks を使いたいですか？GitHub の [`synalinks-skills`](https://github.com/SynaLinks/synalinks-skills) にある Synalinks 専用スキルをエージェントに追加してください。フレームワークの規約を教え込み、すぐに Synalinks プログラムを構築するのに必要なコンテキストを与えてくれます。

</div>

## Synalinks とは？

Synalinks は、RAG、自律エージェント、自己進化する推論システムなど、高度な LM ベースのアプリケーションの作成・訓練・評価・デプロイをシンプルにする、オープンソースのニューロシンボリックフレームワークです。

言語モデルアプリケーションのための Keras と考えてください。クリーンで宣言的な API により、次のことができます。

- ディープラーニングの `Layer` と同じ感覚で [`Module`](https://synalinks.github.io/synalinks/guides/Modules/) を**組み合わせる**。
- インコンテキスト強化学習で**[訓練・最適化](https://synalinks.github.io/synalinks/guides/Training/)**する。
- [REST API](https://synalinks.github.io/synalinks/guides/FastAPI%20Deployment/) や [MCP サーバー](https://synalinks.github.io/synalinks/guides/FastMCP%20Deployment/)として**デプロイ**する。

### 主要な原則

- **段階的な複雑さ**: [シンプルに始めて、自然に高度な構成へ発展](https://synalinks.github.io/synalinks/guides/Getting%20Started/)。
- **ニューロシンボリック学習**: [ロジック・構造](https://synalinks.github.io/synalinks/guides/Data%20Models/)と[言語モデル](https://synalinks.github.io/synalinks/guides/Getting%20Started/)を組み合わせる。
- **インコンテキスト最適化**: [重みを再学習することなくモデルの推論を改善](https://synalinks.github.io/synalinks/guides/Trainable%20Variables/)。

## 誰のためのものか？

<div align="center">

| 役割                      | Synalinks が役立つ理由                                       |
| ------------------------- | ----------------------------------------------------------- |
| **AI 開発者**      | ボイラープレートなしで、本番品質の複雑な LM アプリを構築。 |
| **AI 研究者**     | ニューロシンボリックやインコンテキスト RL のシステムを素早くプロトタイピング。    |
| **データサイエンティスト**    | LM ワークフローを API やデータベースと統合。               |
| **学生・ホビイスト** | クリーンで直感的なフレームワークで AI の組み立て方を学べる。       |

</div>

## なぜ Synalinks なのか？

今日では多くのフレームワークが存在します。Synalinks が異なる点は次のとおりです。

- **組み込み・コンテナ不要のサンドボックス** : エージェントは、**Docker も外部サンドボックスサービスも不要**な[安全で隔離されたランタイム](https://synalinks.github.io/synalinks/guides/Agents/)で、信頼できないコードやツールを実行します。スタック全体が純粋な Python で組み込み可能なので、スクリプティング、研究、サーバーレス／クラウドデプロイ（S3、Lambda、ノートブックなど）、さらには CLI ハーネスの作成にも最適です！
- **組み込みデータベースのサポート** : 組み込みグラフデータベースの上で、**制約付きナレッジグラフ抽出**と**自動セマンティック重複排除**を備えた[グラフベースの RAG やエージェントメモリ](https://synalinks.github.io/synalinks/guides/Knowledge%20Base/)を構築できます。別途グラフサーバーを立てる必要はありません。さらに、リレーショナルデータを保存してベクトル／SQL RAG を構築できる、高速な組み込み **SQL ナレッジベース**も利用できます。
- **プロンプト（や他の何でも）を最適化するインコンテキスト RL** : おなじみの `.compile()` / `.fit()` / `.evaluate()` / `.predict()` API を使い、**モデルの重みに一切触れずに**、プロンプト、few-shot 例、そしてモジュールごとの[任意の学習可能変数](https://synalinks.github.io/synalinks/guides/Trainable%20Variables/)を[訓練・最適化](https://synalinks.github.io/synalinks/guides/Training/)できます。
- **手間いらずのモデル切り替え** : `synalinks.set_default_language_model(...)` で一度デフォルトを設定するか、文字列識別子を渡すだけで、[LiteLLM](https://docs.litellm.ai/docs/) 経由で Ollama、vLLM、OpenAI、Azure、Anthropic、Mistral、Groq、Gemini、xAI、Cohere、DeepSeek、Together AI、OpenRouter、AWS Bedrock、Doubleword を切り替えられます。コストと品質の観点から最適なモデルを選ぶ[多目的モデル選択](https://synalinks.github.io/synalinks/guides/Multi-Objective%20LM%20Selection/)にも対応。
- **コマンド一発でスキャフォールド、コーディングエージェントは持ち込みで** : `synalinks init` で本番運用可能なプロジェクトを一発で構築し（スクリプト、エージェント、訓練用のテンプレートを完備）、公式の [Synalinks スキル](https://github.com/SynaLinks/synalinks-skills)を追加すれば、Claude Code、Cursor、Copilot などが最初から慣用的な Synalinks コードを書いてくれます。

加えて、本番品質のフレームワークに期待されるすべてが揃っています。

- **NEW**: すべてのエージェントが [Agent Skills](https://agentskills.io/home)、`AGENTS.md`、サブエージェントに対応。
- 正確性のための**[制約付き構造化出力](https://synalinks.github.io/synalinks/guides/Data%20Models/)**（JSON）
- **Chat Completions 互換のメッセージ API**: メッセージは OpenAI Chat Completions フォーマットとキー単位で完全に一致し、さらに litellm 拡張の `reasoning_content` と `thinking_blocks` により、プロバイダの推論内容がマルチターンの往復でも失われません。**[マルチモーダル入力](https://synalinks.github.io/synalinks/guides/Multimodal%20Inputs/)**（画像と音声を標準のコンテンツパートとして）も扱えます
- **バージョン管理可能**で JSON シリアライズ可能な[パイプライン](https://synalinks.github.io/synalinks/guides/Programs/)
- デフォルトで**自動的な[非同期・並列実行](https://synalinks.github.io/synalinks/guides/Programs/)**
- **[メトリクス](https://synalinks.github.io/synalinks/guides/Metrics/)、[報酬](https://synalinks.github.io/synalinks/guides/Rewards/)、[データセット](https://synalinks.github.io/synalinks/guides/Datasets/)**を標準搭載
- **API 対応**: [FastAPI](https://synalinks.github.io/synalinks/guides/FastAPI%20Deployment/) や [FastMCP](https://synalinks.github.io/synalinks/guides/FastMCP%20Deployment/) でデプロイ
- ハイパーパラメータ探索のための **[KerasTuner 互換性](https://synalinks.github.io/synalinks/guides/Hyperparameter%20Search/)**
- [オブザーバビリティ](https://synalinks.github.io/synalinks/guides/Observability/)のための**組み込み[コールバック](https://synalinks.github.io/synalinks/guides/Callbacks/)とフック**（MLflow の `Monitor` コールバックを含む）

# 動作要件

- Python 3.12 以上
- Windows ユーザーは WSL2

## `uv` で 3 秒クイックスタート（推奨）

`uv` をご存じない場合は、[こちら](https://docs.astral.sh/uv/getting-started/installation/)からインストールしてください。

指示に従えば、3 秒で新しい synalinks プロジェクトを開始できます。

```shell
uvx synalinks init
```

---

新しいプロジェクトに、以下のようにライブラリをインストールすることもできます。

```shell
uv add synalinks
```

コーディングエージェントを AI エンジニアに変えるには、プロジェクトのルートで次を実行します。

```shell
npx skills add SynaLinks/synalinks-skills --skill synalinks
```

## 例

Synalinks のエージェントは、プロジェクトの [`AGENTS.md`](https://agents.md)
の規約を読み取り、[Agent Skills](https://agentskills.io/home) を使えるようになりました。
以下の例では、公式の [Synalinks スキル](https://github.com/SynaLinks/synalinks-skills)を
[`DeepAgent`](https://synalinks.github.io/synalinks/guides/Agents/)（サンドボックス化された
コーディングエージェント）に組み込み、あるタスクの入出力データモデルを設計して
`workspace/` フォルダに書き込むよう依頼します。

まずワークスペースをセットアップします。[`skills`](https://skills.sh) CLI で公式の
Synalinks スキルをインストールし、`AGENTS.md` を追加します。スキルは作業ディレクトリ配下に
配置されるので、サンドボックス化されたエージェントは必要に応じてその本文を読めます。

```shell
mkdir -p workspace && cd workspace
# `synalinks` スキルを ./.agents/skills/ にインストールし、skills-lock.json を書き出します。
npx skills add SynaLinks/synalinks-skills --skill synalinks
```

これで以下のレイアウトになります。`.agents/skills` がスキルの*ルート*です
（スキルごとに 1 つのサブフォルダがあり、それぞれが `SKILL.md` を持ちます）。

```text
workspace/
├── AGENTS.md                     # エージェントの規約として注入される
├── skills-lock.json              # スキルをソースリポジトリ + コンテンツハッシュに固定する
└── .agents/
    └── skills/                   # スキルのルート
        └── synalinks/
            └── SKILL.md          # 名前と説明が提示され、本文は必要に応じて読まれる
```

`main.py`:

```python
import synalinks
import asyncio

# デフォルトを一度設定すれば、モジュールが自動的に使用します。
synalinks.set_default_language_model("gemini/gemini-3.1-flash-lite-preview")


# エージェントの構造化された最終回答。
class Deliverable(synalinks.DataModel):
    summary: str = synalinks.Field(
        description="What was created and where",
    )
    files: list[str] = synalinks.Field(
        description="Paths of the files written into the workspace",
    )


async def main():
    # DeepAgent は ChatMessages で対話します（コーディングエージェントであるため）。
    inputs = synalinks.Input(data_model=synalinks.ChatMessages)

    agent = synalinks.DeepAgent(
        data_model=Deliverable,
        # サンドボックスはこのディレクトリから初期化されます（ホストに安全: エージェントの
        # 書き込みはサンドボックスのコピーに反映され、あなたのディスクには一切触れません）。
        # `AGENTS.md` が注入されるため、エージェントはあなたの規約に従います。
        workdir="workspace",
        # スキルのルート（`skills add` でインストールされたもの）。エージェントには
        # `<available_skills>` として提示され、各 `SKILL.md` は必要に応じてサンドボックスから
        # 読み込まれます。スキルを `workdir` 配下に置くのはそのためです。
        skills=["workspace/.agents/skills"],
    )
    outputs = await agent(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="datamodel_designer",
        description="Designs Synalinks data models for a given task",
    )

    task = (
        "Define the input and output Synalinks DataModels for a support-ticket "
        "triage task: the input is a raw customer message; the output is the "
        "predicted category, a priority, and a short suggested reply. Write them "
        "to `models.py` using idiomatic Synalinks; consult the skills first."
    )
    result = await program(
        synalinks.ChatMessages(
            messages=[synalinks.ChatMessage(role="user", content=task)],
        )
    )
    print(result.prettify_json())


if __name__ == "__main__":
    asyncio.run(main())
```

## データモデル演算子

Synalinks は、データモデルを組み合わせたり操作したりするための Python 演算子を提供し、高度な制御フローを実現します。これらの演算子が可能にするルーティング、ファンアウト、マージのパターンについては、[制御フローガイド](https://synalinks.github.io/synalinks/guides/Control%20Flow/)を参照してください。

<div align="center">

| 演算子 | 名前 | 説明 | ユースケース |
| :---: | --- | --- | --- |
| `+` | 連結 | 両方のデータモデルのフィールドを結合します。どちらかが `None` の場合は例外を送出します。 | 並列ブランチの出力のマージ |
| `&` | 論理 And | どちらかの入力が `None` の場合に `None` を返す安全な連結。 | `None` になり得るブランチ出力との結合 |
| `\|` | 論理 Or | `None` でない方のデータモデルを返します。両方が `None` でない場合はマージします。 | 条件分岐ブランチの出力の収集 |
| `^` | 論理 Xor | ちょうど 1 つの入力だけが `None` でない場合にそのデータを返し、それ以外は `None` を返します。 | 排他的なブランチ選択 |
| `~` | 論理 Not | 入力が `None` でなければ `None` を返し、`None` であれば空のデータモデルを返します。 | ブランチ条件の反転 |
| `in` | 包含 | 文字列キーがスキーマのプロパティに存在するか、または別のデータモデルのスキーマが含まれているかを確認します。`True` または `False` を返します。 | 条件付きフィールドチェック、スキーマ検証 |

</div>

```python
# 連結を使った並列ブランチ
x1 = await generator1(inputs)
x2 = await generator2(inputs)
# combined = x1 *and* x2
combined = x1 & x2  # 両方の出力をマージ（キーが衝突した場合は _{i} サフィックスを付加）
# [...]
# 論理 Or を使った条件分岐ブランチ
(easy, hard) = await synalinks.Branch(
    question="Is this query complex?",
    labels=["easy", "hard"],
    branches=[simple_generator, complex_generator],
)(inputs)
# result = easy *or* hard
result = easy | hard  # 選択された方のブランチを取得
```

## プログラムのサマリーを取得する

プログラムの表形式のサマリーを表示するには、次のようにします。

```python
program.summary()
```

またはプロット（システムのドキュメント化に便利）を出力できます。

```python
synalinks.utils.plot_program(
    program,
    show_module_names=True,
    show_trainable=True,
    show_schemas=True,
)
```

<div align="center">
<img src="../docs/assets/examples/datamodel_designer.png" alt="データモデルデザイナープログラム" width="600">

<em>plot_program で可視化したデータモデルデザイナープログラム: Input → DeepAgent。学習可能なモジュールは緑色で示されます。</em>
</div>

## プログラムの実行

プログラムを実行するには、次のようにします。

```python
result = await program(
    Query(
        query=(
            "A bookstore receives a shipment of 135 new books."
            "They place the books evenly onto 9 shelves."
            "Later, they decide to move 3 books from each shelf to a display table"
            " at the front of the store. "
            "How many books are left on the shelves after the books are moved?"
        )
    ),
)
```

## プログラム／エージェントの訓練

```python
# デフォルトの言語モデル／埋め込みモデルを設定しておくと、
# 文字列識別子（Keras 流）でパイプラインや訓練を設定できます。
# きめ細かく制御したい場合は、クラスを直接インスタンス化することもできます。
synalinks.set_default_language_model("gemini/gemini-3.1-flash-lite-preview")
synalinks.set_default_embedding_model("gemini/text-embedding-004")


async def main():

    # ... プログラムの定義

    (x_train, y_train), (x_test, y_test) = synalinks.datasets.gsm8k.load_data()

    program.compile(
        reward=synalinks.rewards.ExactMatch(in_mask=["answer"]),
        optimizer="omega",
    )

    batch_size = 1
    epochs = 10

    history = await program.fit(
        x_train,
        y_train,
        validation_split=0.2,
        batch_size=batch_size,
        epochs=epochs,
    )


if __name__ == "__main__":
    asyncio.run(main())
```

## 保存と読み込み

アーキテクチャ全体と変数（プログラムの状態）を JSON ファイルに保存するには、次のようにします。

```python
program.save("my_program.json")
```

読み込むには、次のようにします。

```python
loaded_program = synalinks.Program.load("my_program.json")
```

プログラムの状態（変数）のみを JSON に保存するには、次のようにします。

```python
program.save_variables("my_program.variables.json")
```

その変数を読み込むには（同じアーキテクチャのプログラムが必要）、次のようにします。

```python
program.load_variables("my_program.variables.json")
```

## ロギング

ロギングを有効にするには、スクリプトの冒頭で次を使用します。

```python
synalinks.enable_logging()
```

## オブザーバビリティ

Synalinks は、MLflow を通じてプログラムのトレーシングとモニタリングを行う組み込みのオブザーバビリティを提供します。

> **重要**: `enable_observability()` は、モジュールを作成する**前に**呼び出してください。

```python
import synalinks

# 最初にオブザーバビリティを有効化
synalinks.enable_observability(
    tracking_uri="http://localhost:5000",  # 省略可能: MLflow サーバーの URI
    experiment_name="my_experiment",  # 省略可能: デフォルトは "synalinks_traces"
)

# その後にモジュールを作成すると、自動的にトレースされます
inputs = synalinks.Input(data_model=Query)
outputs = await synalinks.Generator(...)(inputs)
```

訓練のメトリクスとアーティファクトには、`Monitor` コールバックを使用します。

```python
monitor = synalinks.callbacks.Monitor(
    tracking_uri="http://localhost:5000",
    experiment_name="training_runs",
)

await program.fit(x=train_x, y=train_y, callbacks=[monitor])
```

高度な設定については、[オブザーバビリティガイド](https://synalinks.github.io/synalinks/guides/Observability/)を参照してください。

### さらに学ぶ

詳しくは[ドキュメント](https://synalinks.github.io/synalinks/)をご覧ください。疑問があれば、[FAQ](https://synalinks.github.io/synalinks/FAQ/) が役立つかもしれません。

### コントリビューション

追加のモジュール、メトリクス、オプティマイザの実装など、コントリビューションを歓迎します。
詳しい情報や、あなたのアイデア（あるいは論文のアイデア）の実装の手助けが必要な場合は、私たちの Discord にご参加ください。

追加のメトリクス／モジュール／オプティマイザはすべてコアチームの承認が必要である点にご注意ください。現在の主要な LM フレームワークの多くに見られるような、無秩序な肥大化による悪しきソフトウェア慣行を避けるため、ライブラリは可能な限りミニマルでクリーンに保ちたいと考えています。

具体的なフィードバックや機能リクエストがある場合は、[issue](https://github.com/SynaLinks/synalinks/issues) を開いていただくようお願いします。

### コントリビューター

あなたのコントリビューション、フィードバック、そしてサポートが、このプロジェクトを発展させています。

小さなバグ修正から大きな機能まで、オープンなコラボレーションとニューロシンボリック AI の未来を信じてくださり、ありがとうございます。

<a href="https://github.com/SynaLinks/synalinks/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=SynaLinks/synalinks"/>
</a>

### コミュニティ

ニューロシンボリックシステムと AI の未来についてさらに学ぶために、私たちのコミュニティに参加してください。多様なバックグラウンドや教育レベルの方々の参加を歓迎します。

### 私たちの研究の引用

この研究は、Keras の作者である François Chollet の監修のもとで行われました。この研究があなたの研究に役立つ場合は、以下の bibtex エントリを使用してください。

```bibtex
@misc{sallami2025synalinks,
  title={Synalinks},
  author={Sallami, Yoan and Chollet, Fran\c{c}ois},
  year={2025},
  howpublished={\url{https://github.com/SynaLinks/Synalinks}},
}
```

### クレジット

Synalinks は、以下のオープンソースプロジェクトの素晴らしい仕事なしには実現できませんでした。

- [Keras](https://keras.io/)：グラフベースの計算基盤、API、そしてコード・設計・哲学の全般に。
- [DSPy](https://dspy.ai/)：モジュール／オプティマイザの着想に。
- [Pydantic](https://docs.pydantic.dev/latest/)：バックエンドのデータレイヤーに。
- [LiteLLM](https://docs.litellm.ai/docs/)：LM の統合に。
- [DuckDB](https://duckdb.org/)、[Ladybug](https://ladybugdb.com/)、[LanceDB](https://www.lancedb.com/)：素晴らしい組み込みデータベースに。
- [MirageAI](https://www.strukto.ai/mirage)：素晴らしいサンドボックスに！
