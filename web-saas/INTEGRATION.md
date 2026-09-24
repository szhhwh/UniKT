# UniKT 端到端融合方案（docs × 实验管理 × 推理服务）

目标：把三个现有部分融合成一个真正的端到端系统——从看文档、备数据、
发训练，到评估、上线推理，全在一个入口里完成。

## 现状盘点

| 部分 | 位置 | 技术 | 现状 |
|------|------|------|------|
| 文档站 | https://unikt.readthedocs.io （源码 `docs/`） | Sphinx + MyST + furo，中英双语，RTD 自动构建 | 独立站点，与系统无联动 |
| 实验管理 | `web/`（KT 实验管理器） | FastAPI 后端 + Vue3 前端，端口 5173 | 任务启动/监控、预处理、GPU 监控、optuna 搜索、日志 |
| 推理服务 | `web-saas/`（本分支） | SpringBoot 8080 → Python 推理 8100，前端 5174 | 真实 checkpoint 推理已通（DKT），API 骨架就绪 |

## 融合后的形态

```
                    ┌──────────────────────────────────────┐
                    │  统一门户（web-saas/frontend, 5174）   │
                    │  ┌────────┬────────────┬──────────┐  │
用户 ──────────────▶│  │ 文档中心 │  实验中心    │ 推理中心  │  │
                    │  └────┬───┴─────┬──────┴────┬─────┘  │
                    └───────┼─────────┼───────────┼────────┘
                            │         │           │
              /docs/** 静态托管│  /api/exp/** 反代 │ /api/** 直连
                            ▼         ▼           ▼
                    Sphinx 构建产物   web/ 管理器    SpringBoot 8080
                    （随版本构建）   （FastAPI）     → Python 推理 8100
```

- **文档中心**：不再只是外链。`sphinx-build` 产物作为静态资源由
  SpringBoot（或部署期 Nginx）挂在 `/docs/**`，门户内路由 `/docs`
  渲染；每个功能页挂对应文档深链（如任务启动页 ↔ user-guide/
  training-evaluation）。RTD 继续作为对外文档站，二者同源构建。
- **实验中心**：现有 `web/` 前端并入门户（同一 Vue3 技术栈，页面
  组件可平移），或短期由门户 `/exp` 反代到 5173。训练产出的
  run 目录（`runs/normal/<MODEL>_*/best_model.pth`）就是推理的
  checkpoint 来源。
- **推理中心**：SpringBoot 做 API 网关与业务层（鉴权/会话/配额），
  推理引擎自动发现 run 目录（`engine._find_run_dir` 已实现按 mtime
  取最新），模型训练完成即"上线"。

## 端到端主链路（用户视角）

1. 文档中心查数据准备说明 → 实验中心发起预处理（web/ 管理器 preprocess）
2. 实验中心发训练任务（TaskLaunch），GPU 监控看进度
3. 训练完成 → run 目录产出 `best_model.pth` + `run_config.yaml`
4. 推理中心模型列表自动出现该模型（available=true）
5. 演练场/开放 API 实时预测；case_analysis 结果回看

## 分阶段落地

**Phase 1（已完成）**
- [x] 三层推理链路（SpringBoot → Python → 模型注册表）
- [x] 真实 checkpoint 推理（DKT 打样，evaluate.py 同款重建路径）
- [x] SpringBoot 静态托管 Sphinx 产物（`/docs-static/**`，SPA `/docs`
  路由承载门户壳 + iframe 渲染，Sphinx 挂载前缀与 SPA 路由错开）
- [x] 门户文档中心（内嵌完整 Sphinx 站点：侧栏/搜索/主题切换可用，
      RTD 继续作为对外站点）
- [x] 实验中心状态接入（`/api/exp/health` 可达性探测 + 首页状态卡；
      `unikt.exp-base-url` 配置管理器地址）

**M1 计算节点（已完成，2026-09-24）**
- [x] 节点管理页：注册 SSH 可达服务器 → 一键自动部署（pixi/仓库/
      rsync/systemd/健康探测 8 步编排）→ 状态与部署日志
- [x] 推理路由：默认服务 + 全部在线节点聚合（模型 OR 合并、节点
      打标、5s TTL 缓存、掉线顺延）；模型页显示提供节点
- [x] 「页面在本机、计算在远端」的产品形态落地（实测：WSL 节点
      自动部署上线，纯节点供数时预测正常路由）
- 安全：输入白名单 + shellQuote 双层防注入；管理接口 X-Admin-Token
  （恒定时间比较，路径归一化防绕过）；H2 加口令去 AUTO_SERVER

**M2 开放 API（已完成，2026-09-24）**
- [x] API Key 体系：SecureRandom 160bit + SHA-256 哈希存储，明文仅
      创建时一次可见（no-store），前缀展示、用量计数、吊销即失效
- [x] 开放接口 `/api/v1/{models,predict}`（X-API-Key 鉴权）
- [x] 门户「API」页：签发/列表/吊销/调用示例

**M3 数据上传与标准化（待做）**
- 用户上传作答 CSV → 字段映射 → data_process 标准化 → 数据集管理
- 依赖：仓库侧通用 CSV DataSource 或映射层（需调研 data_source 注册）

**M4 一键训练流水线（待做）**
- 数据集 × 模型 → 节点上触发 train.py → 任务状态 → run 目录自动上线
- 复用：M1 的 SSH 编排 + run 目录自动发现

**原 Phase 2/3 条目**
- [ ] 门户 `/exp` 页面级反代（管理器 SPA 资源路径无法安全挂前缀，
      归入统一 API 时一并处理）
- [ ] web/ 前端页面并入门户（木糖、葡萄糖：SpringBoot 侧反代 + 会话打通）
- [ ] 统一 API 前缀：`/api/exp/**`（管理器）、`/api/infer/**`（推理）、
  `/api/docs/**`（文档元信息）
- [ ] checkpoint 注册表升级：SpringBoot 维护模型↔run 目录↔指标的表
- [ ] 门户一键流水线（= M4）、多用户 SaaS 化（鉴权/配额/任务隔离）
- [ ] 每页上下文文档深链；推理结果页挂 case-analysis 报告

## 决策记录

- 文档以内嵌静态托管优先于 iframe 外链 RTD：避免跨域/版本漂移，
  私有化部署（实验室服务器）也能看文档。
- 推理引擎与实验管理器解耦（不同进程/端口）：训练负载不拖垮在线
  推理；二者只通过 run 目录约定衔接。
- SpringBoot 定位为网关+业务层，不实现任何张量计算：课程作业的
  "web framework 核心技术"落在 Java 侧，模型侧全部留在 Python。
