@echo off
REM stateswap 一键启动（Windows）：双击运行，或命令行执行
REM 可用环境变量覆盖：STATESWAP_MODEL（底座路径）、STATESWAP_PORT（端口）
cd /d %~dp0

if exist .venv312\Scripts\python.exe (
  set PY=.venv312\Scripts\python.exe
) else if exist .venv\Scripts\python.exe (
  set PY=.venv\Scripts\python.exe
) else (
  echo [stateswap] 未找到虚拟环境，请先按 docs/tutorial.md 第 1 节完成安装
  pause
  exit /b 1
)

if "%STATESWAP_MODEL%"=="" (set MODEL=models/rwkv7-1.5b-world-hf) else (set MODEL=%STATESWAP_MODEL%)
if "%STATESWAP_PORT%"=="" (set PORT=8000) else (set PORT=%STATESWAP_PORT%)

echo [stateswap] python=%PY%
echo [stateswap] model=%MODEL%  port=%PORT%
echo [stateswap] 首次加载需几十秒（Triton 内核调优），就绪后状态点变绿
echo [stateswap] WebUI: http://127.0.0.1:%PORT%
start "" http://127.0.0.1:%PORT%
%PY% -m stateswap.server --model %MODEL% --persona-dir personas --port %PORT%
pause
