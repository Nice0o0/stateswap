@echo off
REM stateswap one-click launcher (Windows): double-click, or run from cmd
REM Override with env vars: STATESWAP_MODEL (base model), STATESWAP_PORT (port)
cd /d %~dp0

if exist .venv312\Scripts\python.exe (
  set PY=.venv312\Scripts\python.exe
) else if exist .venv\Scripts\python.exe (
  set PY=.venv\Scripts\python.exe
) else (
  echo [stateswap] no venv found - see docs/tutorial.md section 1
  pause
  exit /b 1
)

if "%STATESWAP_MODEL%"=="" (set MODEL=models/rwkv7-1.5b-world-hf) else (set MODEL=%STATESWAP_MODEL%)
if "%STATESWAP_PORT%"=="" (set PORT=8000) else (set PORT=%STATESWAP_PORT%)

echo [stateswap] python=%PY%
echo [stateswap] model=%MODEL%  port=%PORT%
echo [stateswap] first load takes ~30-40s (Triton autotune); the status dot turns green when ready
echo [stateswap] WebUI: http://127.0.0.1:%PORT%
start "" http://127.0.0.1:%PORT%
%PY% -m stateswap.server --model %MODEL% --persona-dir personas --port %PORT%
pause
