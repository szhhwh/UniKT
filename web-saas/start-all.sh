#!/usr/bin/env sh
# 一键启动本地 SaaS（Mac/开发机）：后端 8080 + 前端 5174。
# 推理服务在远程节点（WSL 等）上常驻，无需本地启动；首次使用请在
# 门户「节点」页注册并部署节点。
#
# 用法：./start-all.sh            # 前台占用一个终端（Ctrl+C 停止前端）
#       ./start-all.sh --daemon   # 后台运行，日志见 /tmp/unikt-*.log
set -e
cd "$(dirname "$0")"

# 管理令牌：设了环境变量就带上，没设则后端不开启校验（内网演示）
BACKEND_ENV=""
if [ -n "$UNIKT_ADMINTOKEN" ]; then
  BACKEND_ENV="UNIKT_ADMINTOKEN=$UNIKT_ADMINTOKEN"
fi

echo "==> 启动 SpringBoot 后端 (:8080)"
if [ -n "$BACKEND_ENV" ]; then
  env $BACKEND_ENV java -jar backend/target/unikt-saas-backend-0.1.0-SNAPSHOT.jar \
    > /tmp/unikt-backend.log 2>&1 &
else
  java -jar backend/target/unikt-saas-backend-0.1.0-SNAPSHOT.jar \
    > /tmp/unikt-backend.log 2>&1 &
fi
BACKEND_PID=$!

cleanup() {
  kill "$BACKEND_PID" 2>/dev/null || true
  [ -n "$FRONTEND_PID" ] && kill "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "==> 启动前端开发服务器 (:5174)"
cd frontend
[ -d node_modules ] || npm install --no-fund --no-audit
if [ "$1" = "--daemon" ]; then
  nohup npm run dev > /tmp/unikt-frontend.log 2>&1 &
  FRONTEND_PID=$!
  echo "后台运行中：日志 /tmp/unikt-backend.log /tmp/unikt-frontend.log"
  sleep 6
else
  npm run dev &
  FRONTEND_PID=$!
  wait "$FRONTEND_PID"
fi

echo "门户地址：http://localhost:5174"
