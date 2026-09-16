#!/usr/bin/env bash
# stateswap 一键启动（Linux）
# 可用环境变量覆盖：STATESWAP_MODEL（底座路径）、STATESWAP_PORT（端口）
cd "$(dirname "$0")"

PY=.venv/bin/python
[ -x .venv312/bin/python ] && PY=.venv312/bin/python
if [ ! -x "$PY" ]; then
  echo "[stateswap] 未找到虚拟环境，请先按 docs/tutorial.md 第 1 节完成安装" >&2
  exit 1
fi

MODEL="${STATESWAP_MODEL:-models/rwkv7-1.5b-world-hf}"
PORT="${STATESWAP_PORT:-8000}"

echo "[stateswap] python=$PY"
echo "[stateswap] model=$MODEL  port=$PORT"
echo "[stateswap] WebUI: http://127.0.0.1:$PORT"
exec "$PY" -m stateswap.server --model "$MODEL" --persona-dir personas --port "$PORT"
