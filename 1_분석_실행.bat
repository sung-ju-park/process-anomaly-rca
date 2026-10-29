@echo off
chcp 65001 > nul
set PYTHONUTF8=1
cd /d "%~dp0"
if not exist .venv (
  echo 처음 실행: 가상환경을 만들고 패키지를 설치합니다. 몇 분 걸릴 수 있어요.
  python -m venv .venv
  call .venv\Scripts\activate.bat
  python -m pip install --upgrade pip
  pip install -r requirements.txt
) else (
  call .venv\Scripts\activate.bat
)
python scripts\run_all.py
pause
