#!/usr/bin/env sh
# 一键启动本地 SaaS（Mac/开发机）：后端 8080 + 前端 5174。
# 推理服务在远程节点（WSL 等）上常驻，无需本地启动；首次使用请在
# 门户「节点」页注册并部署节点。
#
# 用法：./start-all.sh            # 前台运行（Ctrl+C 停止全部）
#       ./start-all.sh --daemon   # 后台运行，日志 /tmp/unikt-*.log
#       ./stop-all.sh             # 停止（按 pid 文件，只杀本脚本启动的进程）
# 环境变量：UNIKT_ADMINTOKEN=xxx  # 开启管理接口令牌校验（可选）
set -e
cd "$(dirname "$0")"

DAEMON=0
[ "$1" = "--daemon" ] && DAEMON=1
PID_FILE=/tmp/unikt-saas.pid

# 只查「正在监听」该端口的进程——不带 -sTCP:LISTEN 会把恰好连着
# 该端口的出站连接（如微信连远程 8080）也查出来，造成误判/误杀
listen_pid() {
  lsof -ti "$1" -sTCP:LISTEN 2>/dev/null || true
}

for port in 8080 5174; do
  if [ -n "$(listen_pid :$port)" ]; then
    echo "✗ $port 端口已有服务在监听（可能已启动过）。"
    echo "  直接打开 http://localhost:5174 使用；或先执行：./stop-all.sh"
    exit 1
  fi
done

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

echo "$BACKEND_PID $FRONTEND_PID" > "$PID_FILE"

cleanup() {
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
  rm -f "$PID_FILE"
}

# 后端就绪判定：轮询 /api/health（端口被占等启动失败会在这里暴露）
wait_backend() {
  i=0
  while [ $i -lt 30 ]; do
    if curl -s -m 2 http://localhost:8080/api/health > /dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
      return 1
    fi
    sleep 1
    i=$((i + 1))
  done
  return 1
}

echo "门户地址：http://localhost:5174"

if [ "$DAEMON" = "1" ]; then
  if ! wait_backend; then
    echo "✗ 后端启动失败，看日志：tail -30 /tmp/unikt-backend.log"
    kill "$FRONTEND_PID" 2>/dev/null || true
    rm -f "$PID_FILE"
    exit 1
  fi
  echo "✓ 后台运行中。日志：/tmp/unikt-backend.log /tmp/unikt-frontend.log"
  echo "✓ 停止：./stop-all.sh"
  disown "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
else
  trap cleanup EXIT INT TERM
  echo "（Ctrl+C 停止全部服务）"
  wait "$FRONTEND_PID"
fi
