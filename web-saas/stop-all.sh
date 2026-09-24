#!/usr/bin/env sh
# 停止 start-all.sh 启动的服务（按 pid 文件，只杀自己启动的进程）。
set -e
PID_FILE=/tmp/unikt-saas.pid
if [ ! -f "$PID_FILE" ]; then
  echo "没有找到 pid 文件（服务可能没在运行，或不是 start-all.sh 启动的）。"
  exit 0
fi
for pid in $(cat "$PID_FILE"); do
  if kill "$pid" 2>/dev/null; then
    echo "已停止进程 $pid"
  else
    echo "进程 $pid 已不在运行"
  fi
done
rm -f "$PID_FILE"
