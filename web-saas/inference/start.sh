#!/usr/bin/env sh
# WSL/Linux 推理服务启动脚本（pixi cpu 环境）。
#
# LD_PRELOAD 说明：WSL/Ubuntu 系统自带的 libstdc++ 版本较旧（缺
# CXXABI_1.3.15）。若 torch 先于 scipy 导入，torch 会先把系统旧版
# libstdc++ 载入进程，随后 scipy 的 C 扩展加载失败。这里预加载 pixi
# 环境内的新版 libstdc++，使任何导入顺序都安全（train.py 无此问题是
# 因为它的导入顺序恰好是 utils/scipy 在 torch 之前）。
set -e
cd "$(dirname "$0")"
exec pixi run -e cpu sh -c \
  'LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6" python -m uvicorn main:app --host 0.0.0.0 --port 8100'
