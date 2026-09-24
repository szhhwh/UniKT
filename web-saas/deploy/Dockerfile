# UniKT SaaS 门户（后端+前端单镜像）
# 构建：docker build -f deploy/Dockerfile -t unikt-saas:latest .（在 web-saas/ 目录）
# 运行：见 deploy/docker-compose.yml
FROM node:20-alpine AS frontend
WORKDIR /fe
COPY frontend/package*.json ./
RUN npm ci --no-fund --no-audit
COPY frontend/ ./
RUN npm run build

FROM maven:3.9-eclipse-temurin-17 AS backend
WORKDIR /be
COPY backend/pom.xml ./
RUN mvn -B -q dependency:go-offline
COPY backend/src ./src
RUN mvn -B -q -DskipTests package

FROM eclipse-temurin:17-jre
WORKDIR /app
# 节点一键部署需要 ssh/rsync 客户端（复用挂载进来的 ~/.ssh 密钥）
RUN apt-get update && apt-get install -y --no-install-recommends openssh-client rsync \
    && rm -rf /var/lib/apt/lists/*
COPY --from=backend /be/target/unikt-saas-backend-0.1.0-SNAPSHOT.jar app.jar
COPY --from=frontend /fe/dist ./frontend-dist
# Sphinx 产物（本地或 CI 先构建到 web-saas/docs-build；空目录则文档页降级）
COPY docs-build ./docs-build
COPY inference ./inference-src
ENV UNIKT_FRONTEND_DIR=/app/frontend-dist \
    UNIKT_DOCS_DIR=/app/docs-build \
    UNIKT_INFERENCE_SRC=/app/inference-src \
    TZ=Asia/Shanghai
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s \
  CMD wget -qO- http://localhost:8080/api/health > /dev/null || exit 1
ENTRYPOINT ["java", "-jar", "app.jar"]
