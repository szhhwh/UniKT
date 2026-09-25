# UniKT SaaS

把 UniKT 的知识追踪能力包装成多用户在线服务：免登录演练、上传数据训练
自有模型、API 密钥接入、计算节点与用户管理。三层架构各层独立进程、
通过 HTTP 通信，任意一层可单独替换。

## 功能

- **演练场（免登录）**：选择已训练模型，按「一题一行」录入作答序列，
  获得逐步预测（下一题答对概率）与各知识点掌握度
- **数据与训练（注册）**：上传作答记录 CSV（流式校验、人话报错），
  选择模型与轮数提交训练，任务经 SSH 编排到计算节点执行；完成后模型
  自动上线，以 `模型@数据集` 限定名在演练场可用；不同用户的数据与
  模型互相隔离
- **开放 API（注册）**：签发 API 密钥（SHA-256 哈希存储，明文一次性
  返回），以 `X-API-Key` 调用 `/api/v1/predict`，含每日配额与每分钟
  限速
- **管理（管理员）**：注册 SSH 可达的计算节点并一键部署推理服务；
  用户角色升降与启停（禁用即时踢出会话与密钥）

## 架构

```
浏览器 ── Vite + Vue 3 前端 (5174)
             │ /api/*
             ▼
        SpringBoot 业务后端 (8080)   ← 用户/会话/编排/鉴权/配额
             │ SSH 编排 · X-Inference-Token
             ▼
        Python FastAPI 推理服务 (8100, GPU/CPU 节点)
             │
             ▼
        model/ + utils/（仓库既有代码，60+ KT 模型）
```

与 `web/`（KT 实验管理器，面向跑实验）互不影响；`web-saas/` 面向对外
服务。

## 目录

| 目录 | 技术 | 职责 |
|------|------|------|
| `backend/` | SpringBoot 3 + Java 17 + Maven | 业务后端：REST API、鉴权、会话、训练编排 |
| `frontend/` | Vite 5 + Vue 3 + TypeScript | 门户界面：首页、模型、演练场、数据、训练、API 密钥、管理 |
| `inference/` | FastAPI + Pydantic | 模型推理：checkpoint 加载、逐步掌握度预测 |
| `deploy/` | Dockerfile / compose / systemd / Caddy | 部署形态，详见 `deploy/README.md` |

## 启动（开发）

一键启动（推荐）——推理在远程节点常驻，本地只需前端+后端：

```bash
cd web-saas && ./start-all.sh            # 前台运行；--daemon 后台运行
```

打开 http://localhost:5174 。首次使用：管理员登录后到「节点」页注册
SSH 可达的服务器并一键部署推理服务，之后模型/演练场即有数据。

手动分步（等价）：

```bash
# 1. SpringBoot 后端（端口 8080，需 JDK 17+ 与 Maven）
cd web-saas/backend
mvn spring-boot:run

# 2. 前端（端口 5174，/api 与 /docs-static 自动代理到 8080）
cd web-saas/frontend
npm install && npm run dev
```

生产部署（Docker Compose / systemd / 单端口形态）见 `deploy/README.md`。

## 推理引擎说明

与 `evaluate.py` 同款加载路径：run 目录的 `run_config.yaml` 重建
RunConfig → 注册表实例化 trainer → `load_weights(best_model.pth)`。
run 目录按 mtime 自动发现（最新优先），可用环境变量
`UNIKT_RUN_DIR_<MODEL>` 精确指定。内置模型基于 ASSISTments2009 数据
集（123 个知识点）；用户上传数据训练出的模型使用其自有知识点体系。

## 约定

- 接口契约：请求/响应字段以 `backend/src/.../dto/` 的 record 与
  `inference/main.py` 的 Pydantic 模型为准，两侧保持同名同义
- 端口固定：5174（前端）/ 8080（后端）/ 8100（推理），改动需同步
  `vite.config.ts` 代理与 `application.yml`
