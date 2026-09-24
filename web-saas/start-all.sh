#!/usr/bin/env sh
# 一键启动本地 SaaS（Mac/开发机）：后端 8080 + 前端 5174。
# 推理服务在远程节点（WSL 等）上常驻，无需本地启动；首次使用请在
# 门户「节点」页注册并部署节点。
#
# 用法：./start-all.sh            # 前台运行（Ctrl+C 停止全部）
#       ./start-all.sh --daemon   # 后台运行，日志 /tmp/unikt-*.log
# 环境变量：UNIKT_ADMINTOKEN=xxx  # 开启管理接口令牌校验（可选）
set -e
cd "$(dirname "$0")"

DAEMON=0
[ "$1" = "--daemon" ] && DAEMON=1

# ---- 前置检查：端口占用（已有实例在跑就别再起一个）----
if lsof -ti :8080 > /dev/null 2>&1; then
  echo "✗ 8080 端口已被占用（可能已有一个后端在运行）。"
  echo "  直接打开 http://localhost:5174 使用；或先执行：lsof -ti :8080 | xargs kill"
  exit 1
fi
if lsof -ti :5174 > /dev/null 2>&1; then
  echo "✗ 5174 端口已被占用（可能已有一个前端在运行）。"
  echo "  直接打开 http://localhost:5174 使用；或先执行：lsof -ti :5174 | xargs kill"
  exit 1
fi

# ---- 后端 jar：缺失则尝试构建 ----
JAR=backend/target/unikt-saas-backend-0.1.0-SNAPSHOT.jar
if [ ! -f "$JAR" ]; then
  echo "==> 未找到后端 jar，尝试构建（首次约 1-2 分钟）"
  MVN=$(command -v mvn || true)
  [ -z "$MVN" ] && [ -x "/Applications/IntelliJ IDEA.app/Contents/plugins/maven-plugin/lib/maven3/bin/mvn" ] \
    && MVN="/Applications/IntelliJ IDEA.app/Contents/plugins/maven-plugin/lib/maven3/bin/mvn"
  if [ -z "$MVN" ]; then
    echo "✗ 找不到 Maven，也无法用 IDEA 自带的。请先安装 maven 或用 IDEA 打开 backend 构建一次。"
    exit 1
  fi
  (cd backend && "$MVN" -B -q -DskipTests package)
fi

# ---- 启动后端 ----
echo "==> 启动 SpringBoot 后端 (:8080)"
if [ -n "$UNIKT_ADMINTOKEN" ]; then
  UNIKT_ADMINTOKEN="$UNIKT_ADMINTOKEN" java -jar "$JAR" > /tmp/unikt-backend.log 2>&1 &
else
  java -jar "$JAR" > /tmp/unikt-backend.log 2>&1 &
fi
BACKEND_PID=$!

# ---- 启动前端 ----
echo "==> 启动前端 (:5174)"
(cd frontend && [ -d node_modules ] || npm install --no-fund --no-audit) > /tmp/unikt-frontend.log 2>&1
(cd frontend && npm run dev > /tmp/unikt-frontend.log 2>&1) &
FRONTEND_PID=$!

cleanup() {
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
}

if [ "$DAEMON" = "1" ]; then
  # 后台模式：脱离本脚本的退出陷阱，服务继续运行
  sleep 3
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo "✗ 后端启动失败，看日志：tail -30 /tmp/unikt-backend.log"
    kill "$FRONTEND_PID" 2>/dev/null || true
    exit 1
  fi
  echo "✓ 后台运行中。日志：/tmp/unikt-backend.log /tmp/unikt-frontend.log"
  echo "✓ 停止：lsof -ti :8080 -ti :5174 | xargs kill"
  echo "门户地址：http://localhost:5174"
  disown "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
else
  trap cleanup EXIT INT TERM
  echo "门户地址：http://localhost:5174 （Ctrl+C 停止全部服务）"
  wait "$FRONTEND_PID"
fi
