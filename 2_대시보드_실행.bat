@echo off
chcp 65001 > nul
set PYTHONUTF8=1
cd /d "%~dp0"
if not exist .venv (
  echo 먼저 1_분석_실행.bat 을 실행해 주세요.
  pause
  exit /b
)
call .venv\Scripts\activate.bat
echo 브라우저에서 대시보드가 열립니다. 끝내려면 이 창을 닫으세요.
python -m streamlit run app.py
