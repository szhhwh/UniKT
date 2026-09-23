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
# 1. 推理服务（端口 8100；有 GPU 的机器或 WSL）
cd web-saas/inference
pixi install -e cpu --locked          # 仓库自带环境（首次）
pixi run -e cpu python -m pip install fastapi "uvicorn[standard]"  # 首次
./start.sh                            # Linux/WSL；macOS 直接 uvicorn main:app --port 8100

# 2. SpringBoot 后端（端口 8080，需 JDK 17+ 与 Maven）
cd web-saas/backend
mvn spring-boot:run

# 3. 前端（端口 5174，/api 自动代理到 8080）
cd web-saas/frontend
npm install
npm run dev
```

打开 http://localhost:5174 ，首页应显示「推理服务：在线」。

### WSL 常驻部署（推荐）

推理放 WSL/GPU 机器时，用 systemd 托管（`inference/unikt-inference.service`，
路径按需调整）：

```bash
scp web-saas/inference/unikt-inference.service wsl:/etc/systemd/system/
ssh wsl 'systemctl daemon-reload && systemctl enable --now unikt-inference'
```

Mac/其它机器访问 WSL 推理：**直连 WSL 的 Tailscale IP**（推荐，
uvicorn 已绑 0.0.0.0；比 SSH 隧道稳定，隧道进程易被会话回收）：

```bash
# 查 WSL 的 tailscale IP 后，以命令行参数绑定（env 变量名宽松绑定不可靠）
java -jar target/unikt-saas-backend-0.1.0-SNAPSHOT.jar \
  --unikt.inference-base-url=http://<WSL_IP>:8100
```

### 故障排查（WSL）

- **torch 先导入后 scipy 报 `CXXABI_1.3.15 not found`**：系统 libstdc++
  过旧被 torch 先载入。**凡 torch+scipy 同进程的入口**（uvicorn、
  pytest、evaluate.py、case_analysis.py、efficiency.py）都受影响，
  统一用
  `LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6 pixi run -e cpu python ...`
  方式启动；`start.sh` 已内置该修复。
- **ssh 一断后台进程就死**：直接 `nohup &` 的进程会随 WSL 会话回收，
  用 systemd（上面的单元文件）托管。长 ssh 会话也易被掐断（Tailscale
  链路），长任务放后台并轮询日志。

## 已跑通的链路

- `GET /api/health`：聚合 SpringBoot 与推理服务的状态
- `GET /api/models`：模型清单 + 是否有已训练 checkpoint（`best_model.pth`）
- `POST /api/predict`：输入作答序列，返回逐步掌握度**真实模型预测**

推理引擎与 `evaluate.py` 同款加载路径：run 目录的 `run_config.yaml`
重建 RunConfig → 注册表实例化 trainer → `load_weights(best_model.pth)`。
run 目录自动发现（按 mtime 取最新）：

- 环境变量 `UNIKT_RUN_DIR_<MODEL>` 精确指定；
- 否则扫 `UNIKT_RUNS_DIR`（默认 `runs/normal/`）下 `<MODEL>_*` 目录。

先训练一个模型（WSL/GPU 或任何能跑训练的机器）：

```bash
pixi install -e cpu --locked
pixi run -e cpu python data_process.py download -d assistments09
pixi run -e cpu python train.py -m DKT -d assistments09 --model.epochs 3
```

然后启动推理服务即可看到 DKT `available=true`。前向适配目前覆盖
DKT 族（`forward(sequence, response, mask)` → `[B, L, num_skills]`，
取 `out[t, skill[t+1]]`）；其它输出形状的模型在
`inference/engine.py` 的适配器处补充。

## 约定

- 接口契约：请求/响应字段以 `backend/src/.../dto/` 的 record 与
  `inference/main.py` 的 Pydantic 模型为准，两侧保持同名同义
- 前端选了 Vue 3（与 `web/frontend` 一致、团队熟）。若作业要求 React，
  只需重建 `frontend/`（`npm create vite@latest -- --template react-ts`），
  `api/client.ts` 的接口定义可原样搬过去
- 端口固定：5174（前端）/ 8080（SpringBoot）/ 8100（推理），改动需同步
  `vite.config.ts` 代理与 `application.yml`
