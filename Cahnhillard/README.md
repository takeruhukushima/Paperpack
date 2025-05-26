# Free Energy Research Project

This repository contains research code for Free Energy calculations and analysis.

## Prerequisites

### Python Installation

#### For Windows
1. Download the latest Python installer from [python.org](https://www.python.org/downloads/)
2. Run the installer
3. **Important**: Check the box that says "Add Python to PATH"
4. Click "Install Now"
5. Verify installation by opening Command Prompt and running:
   ```
   python --version
   ```

#### For macOS
1. Install Homebrew (if not already installed):
   ```
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
2. Install Python:
   ```
   brew install python
   ```
3. Verify installation:
   ```
   python3 --version
   ```

#### For Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

## Setup and Running

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone [repository-url]
   cd FreeEnergy
   ```

2. **Create a virtual environment**:
   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   (If you don't have a requirements.txt, install packages individually as needed)

4. **Run the application**:
   ```bash
   python main.py
   ```
   or for Python 3 specifically:
   ```bash
   python3 main.py
   ```

## セットアップ手順

### Pythonのインストール

#### Windowsの場合
1. [python.org](https://www.python.org/downloads/) から最新のPythonインストーラーをダウンロード
2. インストーラーを実行
3. **重要**: "Add Python to PATH" にチェックを入れる
4. "Install Now" をクリック
5. コマンドプロンプトを開いてインストールを確認:
   ```
   python --version
   ```

#### macOSの場合
1. Homebrewがインストールされていない場合はインストール:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
2. Pythonをインストール:
   ```bash
   brew install python
   ```
3. インストールを確認:
   ```bash
   python3 --version
   ```

#### Linux (Ubuntu/Debian) の場合
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

### 仮想環境のセットアップと実行

1. **リポジトリをクローン** (まだの場合):
   ```bash
   git clone [repository-url]
   cd FreeEnergy
   ```

2. **仮想環境を作成**:
   ```bash
   # Windowsの場合
   python -m venv venv
   .\venv\Scripts\activate

   # macOS/Linuxの場合
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **必要なパッケージをインストール**:
   ```bash
   pip install -r requirements.txt
   ```
   (requirements.txtがない場合は、必要なパッケージを個別にインストール)

4. **アプリケーションを実行**:
   ```bash
   python main.py
   ```
   またはPython 3を明示的に指定:
   ```bash
   python3 main.py
   ```

## 注意事項
- 仮想環境を終了するには、ターミナルで `deactivate` と入力します
- 新しいターミナルを開いた場合は、再度仮想環境を有効にする必要があります
