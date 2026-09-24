# 部署指南（第 1 档：一个网址给别人用）

目标形态：**一个固定网址（HTTPS）**，别人打开即用演练场，开发者注册拿
密钥调接口；推理跑在 GPU 节点（WSL/服务器），经门户「节点」页自动部署。

```
用户 ──HTTPS──▶ Caddy/Nginx(:443) ──▶ SpringBoot(:8080, 托管前端+API)
                                        │ 经 SSH 自动部署/调用（X-Inference-Token）
                                        ▼
                                  GPU 节点 FastAPI(:8100, 只在 Tailscale/内网可达)
```

## 一、按目标选方式

| 目标 | 方式 | 步骤 |
|---|---|---|
| 云服务器（有公网 IP） | docker compose | 下文 §二 |
| 学校/实验室服务器 | systemd | 下文 §三 |
| Mac + Tailscale（无公网） | systemd 或直接 start-all.sh | §三；HTTPS 用 Tailscale 自带域名见 §四 |

## 二、云服务器（Docker Compose，推荐）

```bash
# 1. 服务器装 docker（略），克隆仓库
git clone -b feat/web-saas https://github.com/szhhwh/UniKT.git && cd UniKT/web-saas

# 2. 配置环境变量
cd deploy
cat > .env <<EOF
MYSQL_PASSWORD=换一个强密码
UNIKT_ADMIN_PASSWORD=首个管理员密码
UNIKT_INFERENCE_TOKEN=一串随机密钥（与推理节点一致）
EOF

# 3. 起服务（门户+MySQL）
docker compose up -d --build

# 4. HTTPS：装 caddy，把 deploy/Caddyfile 的域名换成你的
```

## 三、服务器（systemd）

```bash
git clone -b feat/web-saas https://github.com/szhhwh/UniKT.git /opt/unikt
cd /opt/unikt/web-saas
# 前端构建（需 Node 20+）
(cd frontend && npm ci && npm run build)
# 后端构建（需 JDK17 + Maven）
(cd backend && mvn -DskipTests package)
# 文档（可选，在装了 pixi 的机器上）
pixi run -e docs sphinx-build -b html docs/source docs-build

# systemd
sudo cp deploy/unikt-saas.service /etc/systemd/system/
# 编辑 service：改 WorkingDirectory 与两个 Environment（管理员密码/推理密钥）
sudo systemctl daemon-reload && sudo systemctl enable --now unikt-saas
```

## 四、HTTPS

- **有域名**：Caddy 一行配置自动签证书（见 deploy/Caddyfile）
- **Tailscale 内网**：开 MagicDNS + HTTPS 证书（tailscale 管理台一键），
  用 `https://机器名.tailnet-xxx.ts.net` 访问，零配置证书

## 五、推理节点（GPU 机）

门户跑起来后，管理员登录 → 「节点」页注册 GPU 机（SSH 免密 + root）→
一键部署。节点只需要 Tailscale/内网可达，**不要**把 8100 端口暴露公网；
`UNIKT_INFERENCE_TOKEN` 已由部署单元自动下发（推理服务拒绝无密钥直连）。

## 六、安全清单（上线前）

- [ ] `UNIKT_ADMIN_PASSWORD` 已设置（否则首启随机密码只在日志里一次）
- [ ] `UNIKT_INFERENCE_TOKEN` 门户与节点两侧一致
- [ ] 数据库密码非默认值；MySQL 不监听公网
- [ ] 节点 8100 仅内网/Tailscale 可达
- [ ] Caddy/Nginx 已启用（8080 不直接暴露公网）
