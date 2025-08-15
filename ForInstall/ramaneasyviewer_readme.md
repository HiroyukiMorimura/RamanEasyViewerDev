# RamanEasyViewer

ラマンスペクトル分析アプリケーション - セキュリティ強化版

## 📋 概要

RamanEasyViewerは、RamanEyeで取得したラマンスペクトルデータの分析と可視化を行うStreamlitベースアプリケーションです。

## 🚀 クイックスタート

### 前提条件

- **Python 3.7以上** がインストールされていること
- **Git** がインストールされていること
- **Windows 10/11** 環境（バッチファイル使用）

### 1. 初期セットアップ

#### 自動セットアップ（推奨）
```bash
# setup.batをダウンロードして実行
setup.bat
```

#### 手動セットアップ
```bash
# リポジトリをクローン
git clone https://github.com/HiroyukiMorimura/RamanEasyViewer.git
cd RamanEasyViewer

# 仮想環境を作成
python -m venv env

# 仮想環境をアクティベート
env\Scripts\activate

# 依存関係をインストール
pip install -r requirements.txt
```

### 2. セキュリティ強化版の準備

1. **ForInstall** フォルダから以下のファイルをコピー：
   - `run_custom_encode.bat`
   - `custom_encode.py`

2. これらのファイルを **RamanEasyViewer** フォルダ内に配置

3. **セキュリティエンコーディングを実行**：
   ```
   run_custom_encode.bat
   ```

4. `custom_encoded_secure` フォルダが作成されることを確認

### 3. アプリケーションの起動

#### 方法1: 暗号化版の起動
```
cd custom_encoded_secure
RamanEasyViewer.bat
```

#### 方法2: 通常版の起動
```
# 仮想環境をアクティベート
env\Scripts\activate

# アプリケーションを起動
streamlit run main_script.py
```

### 4. デスクトップショートカットの作成

#### 手動でショートカットを作成：

1. **`custom_encoded_secure/RamanEasyViewer.bat`** を右クリック
2. **「その他のオプションを確認」** → **「ショートカットの作成」** を選択
3. ショートカット名を **`RamanEasyViewerLauncher`** に変更
4. ショートカットを右クリック → **「プロパティ」**
5. **「ショートカット」** タブ → **「アイコンの変更」**
6. 警告が表示されたら **「OK」**
7. **「参照」** をクリックして `favicon.ico` を選択
8. **「OK」** → **「適用」** → **「OK」**
9. ショートカットをデスクトップに移動

## 🔧 トラブルシューティング

### アプリケーションが起動しない場合

#### Step 1: 基本診断
```
cd custom_encoded_secure
test_secure.bat
```

**診断オプション：**
- **1. Security Validation Test**: セキュリティ機能の動作確認
- **2. Import Verification Test**: 必要なモジュールの読み込み確認
- **4. Package Dependencies Check**: 依存パッケージの確認
- **6. Full System Diagnostic**: システム全体の診断
- **7. Python Environment Debug**: Python環境の詳細分析

#### Step 2: よくある問題と解決方法

**🐍 Pythonエラー**
```
[エラー] Pythonが利用できません！
```
**解決方法:**
- Python 3.7以上がインストールされているか確認
- `python --version` でバージョン確認
- PATH環境変数にPythonが追加されているか確認

**📦 Streamlitエラー**
```
[エラー] Streamlitがインストールされていません！
```
**解決方法:**
```bash
pip install streamlit
```

**🔐 仮想環境エラー**
```
[警告] 仮想環境が見つかりません
```
**解決方法:**
- `setup.bat` を再実行
- または手動で仮想環境を作成：
  ```bash
  python -m venv env
  env\Scripts\activate
  pip install -r requirements.txt
  ```

**📁 ファイル不足エラー**
```
[エラー] main_script.py が見つかりません！
```
**解決方法:**
- `run_custom_encode.bat` が正常に実行されているか確認
- `custom_encoded_secure` フォルダ内にファイルが存在するか確認
- 必要に応じて暗号化処理を再実行

#### Step 3: ログファイルの確認

**実行ログの場所:**
- `RamanEasyViewer_debug.log`: アプリケーション実行ログ
- `security_test_log_*.txt`: セキュリティテストログ

**ログの確認方法:**
```
# テストメニューから
test_secure.bat → 8. View Log File

# または直接ファイルを開く
notepad RamanEasyViewer_debug.log
```

## 📁 ファイル構造

```
RamanEasyViewer/
├── main_script.py          # メインアプリケーション
├── requirements.txt        # 依存関係
├── favicon.ico            # アプリケーションアイコン
├── env/                   # Python仮想環境
├── custom_encode.py       # セキュリティエンコーダー
├── run_custom_encode.bat  # 暗号化実行スクリプト
└── custom_encoded_secure/ # 暗号化版アプリケーション
    ├── RamanEasyViewer.bat    # 起動スクリプト
    ├── test_secure.bat        # 診断ツール
    ├── main_script.py         # 暗号化されたメインファイル
    ├── favicon.ico            # アイコンファイル
    └── *.py                   # その他の暗号化されたファイル
```

## 🔒 セキュリティ機能

### 暗号化レベル
- **Multi-layer encryption**: 多層暗号化
- **PBKDF2 key derivation**: 動的キー生成
- **XOR + Bit manipulation**: XOR暗号化とビット操作
- **Custom Base64 alphabet**: カスタムBase64エンコーディング
- **Code obfuscation**: コード難読化

### セキュリティテスト
```
test_secure.bat
```
で以下の項目をテスト可能：
- セキュリティ検証
- モジュール読み込み確認
- パフォーマンスベンチマーク
- システム診断

## 🛠️ 開発者向け情報

### 開発環境のセットアップ
```bash
# 開発用の依存関係をインストール
pip install -r requirements-dev.txt

# アプリケーションを開発モードで起動
streamlit run main_script.py --logger.level=debug
```

### カスタマイズ

#### アイコンの変更
`favicon.ico` を置き換えることで、アプリケーションのアイコンをカスタマイズできます。

#### ポート番号の変更
`RamanEasyViewer.bat` 内の `--server.port 8501` を編集してください。

## 📞 サポート

### 問題が解決しない場合

1. **ログファイルを確認** してエラーの詳細を把握
2. **test_secure.bat** で詳細診断を実行
3. **GitHub Issues** で問題を報告
4. **必要な情報**：
   - OS バージョン
   - Python バージョン
   - エラーメッセージ
   - ログファイルの内容

### よくある質問 (FAQ)

**Q: アプリケーションがブラウザで開かない**
A: `http://localhost:8501` に手動でアクセスしてください。

**Q: 暗号化版と通常版の違いは？**
A: 暗号化版はソースコードが保護されており、不正な改変や解析を防ぎます。

**Q: 他のPCで実行できますか？**
A: `custom_encoded_secure` フォルダ全体をコピーすれば、Python環境があるPCで実行可能です。

## 📄 ライセンス

MIT License

## 🔄 更新履歴

- **v2.0**: セキュリティ強化版リリース
- **v1.0**: 初回リリース

---

**🚀 RamanEasyViewer で効率的なスペクトル分析を始めましょう！**