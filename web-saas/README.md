# UniKT SaaS（web framework 编程作业）

把 UniKT 的知识追踪能力做成在线服务（SaaS）。三段式架构，各层独立进程、
通过 HTTP 通信，任意一层可以单独替换。

```
浏览器 ── Vite 前端 (5174)
             │ /api/*
             ▼
        SpringBoot 业务后端 (8080)   ← 用户/会话/编排/鉴权
             │ /health /models /predict
             ▼
        Python 推理服务 (8100)       ← checkpoint 加载与前向计算
             │
             ▼
        model/ + utils/（仓库既有代码，60+ KT 模型）
```

与 `web/`（KT 实验管理器，FastAPI+Vue）互不影响；`web/` 面向跑实验，
`web-saas/` 面向对外服务。

## 目录

| 目录 | 技术 | 职责 | 分工 |
|------|------|------|------|
| `backend/` | SpringBoot 3 + Java 17 + Maven | 业务后端：REST API、鉴权、会话、调用推理 | 木糖、葡萄糖 |
| `inference/` | FastAPI + Pydantic | 模型推理：加载 checkpoint、逐步掌握度预测 | Python 侧负责人 |
| `frontend/` | Vite 5 + Vue 3 + TypeScript | 界面：模型列表、推理演练场 | 前端侧负责人 |

## 启动（开发）

三个终端分别执行：

```bash
# 1. 推理服务（端口 8100）
cd web-saas/inference
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --port 8100 --reload

# 2. SpringBoot 后端（端口 8080，需 JDK 17+ 与 Maven）
cd web-saas/backend
mvn spring-boot:run

# 3. 前端（端口 5174，/api 自动代理到 8080）
cd web-saas/frontend
npm install
npm run dev
```

打开 http://localhost:5174 ，首页应显示「推理服务：在线」。

## 已跑通的链路

- `GET /api/health`：聚合 SpringBoot 与推理服务的状态
- `GET /api/models`：模型注册表清单（来自 `model/` 的 `TRAINERS` 发现机制）
- `POST /api/predict`：输入作答序列，返回逐步掌握度

推理当前是**计数基线**（前 i 步答对率），用于验证三层链路；接入真实
checkpoint 时只改 `inference/engine.py` 的 `_load_model` 与 `predict`，
Java 与前端接口均不变。

## 约定

- 接口契约：请求/响应字段以 `backend/src/.../dto/` 的 record 与
  `inference/main.py` 的 Pydantic 模型为准，两侧保持同名同义
- 前端选了 Vue 3（与 `web/frontend` 一致、团队熟）。若作业要求 React，
  只需重建 `frontend/`（`npm create vite@latest -- --template react-ts`），
  `api/client.ts` 的接口定义可原样搬过去
- 端口固定：5174（前端）/ 8080（SpringBoot）/ 8100（推理），改动需同步
  `vite.config.ts` 代理与 `application.yml`
