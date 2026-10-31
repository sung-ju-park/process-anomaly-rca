@echo off
chcp 65001 > nul
set PYTHONUTF8=1
cd /d "%~dp0"
where ollama > nul 2> nul
if errorlevel 1 (
  echo Ollama가 설치되어 있지 않습니다. https://ollama.com 에서 설치한 뒤 다시 실행하세요.
  pause
  exit /b 1
)
ollama list > nul 2> nul
if errorlevel 1 (
  echo Ollama를 시작합니다...
  start "" /min ollama serve
  timeout /t 8 > nul
)
ollama list | findstr /c:"qwen2.5:3b" > nul
if errorlevel 1 (
  echo qwen2.5:3b 모델을 내려받습니다. 약 2GB, 처음 한 번만 필요합니다.
  ollama pull qwen2.5:3b
)
if not exist .venv (
  echo 먼저 1_분석_실행.bat 을 실행해 주세요.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
echo [1/2] 검증기 점검 (LLM 없이, 틀린 내용을 잡아내는지)
python scripts\check_llm_guard.py
echo.
echo [2/2] 고장 21종 경보에 대해 로컬 LLM 보고서 초안 작성·검증 (CPU 노트북 기준 20~40분)
python scripts\run_llm_report_eval.py
pause
