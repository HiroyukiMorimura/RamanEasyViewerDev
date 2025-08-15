@echo off
chcp 65001 > nul
echo ====================================
echo RamanEasyViewer セットアップスクリプト
echo ====================================
echo.

REM Gitがインストールされているかチェック
git --version >nul 2>&1
if errorlevel 1 (
    echo エラー: Gitがインストールされていません。
    echo https://git-scm.com/ からGitをインストールしてください。
    pause
    exit /b 1
)

REM Pythonがインストールされているかチェック
python --version >nul 2>&1
if errorlevel 1 (
    echo エラー: Pythonがインストールされていません。
    echo https://www.python.org/ からPythonをインストールしてください。
    pause
    exit /b 1
)

echo 1. GitHubからRamanEasyViewerをクローンしています...
git clone https://github.com/HiroyukiMorimura/RamanEasyViewer.git
if errorlevel 1 (
    echo エラー: リポジトリのクローンに失敗しました。
    pause
    exit /b 1
)

echo.
echo 2. プロジェクトディレクトリに移動しています...
cd RamanEasyViewer
if errorlevel 1 (
    echo エラー: ディレクトリの移動に失敗しました。
    pause
    exit /b 1
)

echo.
echo 3. Python仮想環境を作成しています...
python -m venv env
if errorlevel 1 (
    echo エラー: 仮想環境の作成に失敗しました。
    pause
    exit /b 1
)

echo.
echo 4. 仮想環境をアクティベートしています...
call env\Scripts\activate.bat
if errorlevel 1 (
    echo エラー: 仮想環境のアクティベートに失敗しました。
    pause
    exit /b 1
)

echo.
echo 5. 必要なパッケージをインストールしています...
pip install -r requirements.txt
if errorlevel 1 (
    echo エラー: パッケージのインストールに失敗しました。
    pause
    exit /b 1
)

echo.
echo ====================================
echo セットアップが完了しました！
echo ====================================
echo.
echo 今後このアプリケーションを実行する場合は：
echo 1. cd RamanEasyViewer
echo 2. env\Scripts\activate
echo 3. python [メインスクリプト名].py
echo.
echo 仮想環境を無効にする場合は「deactivate」と入力してください。
echo.
pause