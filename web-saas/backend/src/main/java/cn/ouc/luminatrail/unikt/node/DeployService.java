package cn.ouc.luminatrail.unikt.node;

import cn.ouc.luminatrail.unikt.config.InferenceProperties;
import jakarta.annotation.PreDestroy;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;

/**
 * 节点部署编排：把 web-saas/inference 自动部署到远程机器并注册为推理节点。
 *
 * 步骤与手工部署完全一致（已在本机 WSL 上验证过）：连通性 → pixi →
 * 仓库 → rsync 代码 → pixi 环境 → fastapi 依赖 → systemd 服务 → 健康
 * 探测。任一步失败置 FAILED 并记录 lastError。
 *
 * 状态探测由 {@link #refreshAll} 后台定时执行（15s），HTTP 接口只读库，
 * 避免「黑洞地址节点拖死轮询请求」。部署线程每步落库前校验节点仍存在，
 * 防止「部署中删除」产生僵尸行。
 */
@Service
public class DeployService {

    private static final Logger log = LoggerFactory.getLogger(DeployService.class);

    private final NodeRepository nodes;
    private final SshExecutor ssh;
    private final InferenceProperties props;

    private final ExecutorService deployPool = Executors.newFixedThreadPool(
            2, r -> {
                Thread t = new Thread(r, "node-deploy");
                t.setDaemon(true);
                return t;
            });

    /** 每节点同时只允许一次部署。 */
    private final Map<Long, AtomicBoolean> deploying = new ConcurrentHashMap<>();

    /** 部署被中止（节点已删除等），无需再写状态。 */
    private static final class DeployAborted extends RuntimeException {
        DeployAborted(String message) {
            super(message);
        }
    }

    public DeployService(NodeRepository nodes, SshExecutor ssh, InferenceProperties props) {
        this.nodes = nodes;
        this.ssh = ssh;
        this.props = props;
    }

    /** 触发异步部署；已在部署中返回 false。 */
    public boolean deployAsync(long nodeId) {
        ComputeNode node = nodes.findById(nodeId).orElseThrow();
        AtomicBoolean flag = deploying.computeIfAbsent(nodeId, k -> new AtomicBoolean(false));
        if (!flag.compareAndSet(false, true)) {
            return false;
        }
        try {
            deployPool.submit(() -> {
                try {
                    deploySync(node);
                } finally {
                    flag.set(false);
                }
            });
        } catch (RuntimeException e) {
            // 线程池已关闭等场景：回滚标志，避免该节点永久 409
            flag.set(false);
            throw e;
        }
        return true;
    }

    private void deploySync(ComputeNode node) {
        try {
            doDeploy(node);
        } catch (DeployAborted e) {
            log.info("deploy aborted for node {}: {}", node.getName(), e.getMessage());
        } catch (Exception e) {
            // step() 已记录过失败详情时不再覆盖 lastError
            if (!ComputeNode.FAILED.equals(node.getStatus())) {
                fail(node, "部署异常：" + e.getMessage());
            }
            log.error("deploy failed for node {}", node.getName(), e);
        }
    }

    private void doDeploy(ComputeNode node) {
        node.setStatus(ComputeNode.DEPLOYING);
        node.setLastError(null);
        node.appendLog("[" + Instant.now() + "] 开始部署 " + node.getName());
        saveAlive(node);

        String t = node.getSshTarget();
        String repo = SshExecutor.shellQuote(node.getRepoPath());
        int stepTimeout = (int) Math.max(60, props.deployTimeoutMs() / 1000 / 8);

        // 1. SSH 连通性
        step(node, "SSH 连通性", () -> require(ssh.run(t, "echo __OK__", 20).ok(), "无法建立 SSH 连接"));

        // 2. pixi（没有则安装）
        step(node, "检查 pixi", () -> {
            SshExecutor.SshResult r = ssh.run(t,
                    "export PATH=$HOME/.pixi/bin:$PATH; pixi --version || "
                            + "(curl -fsSL https://pixi.sh/install.sh | PIXI_VERSION=v0.79.0 sh)",
                    Math.max(stepTimeout, 300));
            return require(r.ok(), "pixi 安装失败: " + SshExecutor.tail(r.stderr(), 3));
        });

        // 3. 仓库（已存在则 pull，否则 clone）
        step(node, "准备仓库", () -> {
            String cmd = "if [ -d " + repo + "/.git ]; then cd " + repo
                    + " && git fetch origin && git checkout -q main && git pull --ff-only -q origin main;"
                    + " else git clone -q --depth 1 https://github.com/szhhwh/UniKT.git " + repo + "; fi";
            SshExecutor.SshResult r = ssh.run(t, cmd, Math.max(stepTimeout, 300));
            return require(r.ok(), "仓库准备失败: " + SshExecutor.tail(r.stderr() + r.stdout(), 3));
        });

        // 4. rsync 本地 inference 源码（源目录按部署机进程位置解析为绝对路径）
        step(node, "同步推理服务代码", () -> {
            String src = Path.of(props.inferenceSrc()).toAbsolutePath().normalize().toString();
            SshExecutor.SshResult r = ssh.rsync(src, t,
                    node.getRepoPath() + "/web-saas/inference", stepTimeout);
            return require(r.ok(), "rsync 失败: " + SshExecutor.tail(r.stderr(), 3));
        });

        // 5. pixi cpu 环境（幂等）
        step(node, "安装 cpu 环境（首次较久）", () -> {
            String cmd = "cd " + repo + " && export PATH=$HOME/.pixi/bin:$PATH && "
                    + "pixi install -e cpu --locked";
            SshExecutor.SshResult r = ssh.run(t, cmd, Math.max(stepTimeout, 1800));
            return require(r.ok(), "pixi install 失败: " + SshExecutor.tail(r.stderr() + r.stdout(), 3));
        });

        // 6. fastapi/uvicorn（幂等）
        step(node, "安装 fastapi/uvicorn", () -> {
            String py = node.getRepoPath() + "/.pixi/envs/cpu/bin/python";
            String cmd = SshExecutor.shellQuote(py) + " -m ensurepip --upgrade >/dev/null 2>&1; "
                    + SshExecutor.shellQuote(py) + " -m pip install -q fastapi 'uvicorn[standard]'";
            SshExecutor.SshResult r = ssh.run(t, cmd, Math.max(stepTimeout, 600));
            return require(r.ok(), "pip 安装失败: " + SshExecutor.tail(r.stderr(), 3));
        });

        // 7. systemd 服务（当前实现假定 root 登录，见节点页提示）
        step(node, "安装 systemd 服务", () -> {
            String unit = systemdUnit(node);
            String cmd = "printf '%s' " + SshExecutor.shellQuote(unit)
                    + " > /etc/systemd/system/unikt-inference.service && "
                    + "systemctl daemon-reload && systemctl enable unikt-inference >/dev/null 2>&1; "
                    + "systemctl restart unikt-inference";
            SshExecutor.SshResult r = ssh.run(t, cmd, stepTimeout);
            return require(r.ok(), "systemd 配置失败: " + SshExecutor.tail(r.stderr(), 3));
        });

        // 8. 健康探测（通过对外 baseUrl，与路由同一视角）
        step(node, "健康探测", () -> {
            for (int i = 0; i < 45; i++) {
                if (ping(normalize(node.getBaseUrl()))) {
                    refreshModels(node);
                    node.appendLog("[" + Instant.now() + "] 部署完成，节点上线");
                    node.setStatus(ComputeNode.ONLINE);
                    saveAlive(node);
                    return true;
                }
                try {
                    Thread.sleep(2000);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
            return require(false, "推理服务约 2-4 分钟内未就绪（远端排查：journalctl -u unikt-inference -f）");
        });
    }

    /** 后台定时探测全部节点（15s），HTTP 列表接口只读库。 */
    @Scheduled(fixedDelay = 15000)
    public void refreshAll() {
        for (ComputeNode node : nodes.findAll()) {
            String status = node.getStatus();
            // DEPLOYING 由部署线程管理；FAILED/NEW 是事实状态，探测不改写
            if (ComputeNode.DEPLOYING.equals(status)
                    || ComputeNode.FAILED.equals(status)
                    || ComputeNode.NEW.equals(status)) {
                continue;
            }
            deployPool.submit(() -> {
                try {
                    refresh(node);
                } catch (Exception e) {
                    log.debug("refresh failed for {}: {}", node.getName(), e.getMessage());
                }
            });
        }
    }

    /** 探测单个节点并更新状态与模型数。 */
    public void refresh(ComputeNode node) {
        if (ping(normalize(node.getBaseUrl()))) {
            refreshModels(node);
            node.setStatus(ComputeNode.ONLINE);
        } else {
            node.setStatus(ComputeNode.OFFLINE);
        }
        nodes.save(node);
    }

    private void refreshModels(ComputeNode node) {
        try {
            List<Map<String, Object>> models = client(normalize(node.getBaseUrl()), 8000)
                    .get().uri("/models")
                    .retrieve()
                    .body(new ParameterizedTypeReference<List<Map<String, Object>>>() {
                    });
            node.setModelsCount(models == null ? 0 : models.size());
        } catch (Exception e) {
            log.debug("refresh models failed for {}: {}", node.getName(), e.getMessage());
        }
    }

    private static boolean ping(String baseUrl) {
        try {
            client(baseUrl, 8000).get().uri("/health").retrieve().toBodilessEntity();
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    private static RestClient client(String baseUrl, int timeoutMs) {
        SimpleClientHttpRequestFactory f = new SimpleClientHttpRequestFactory();
        f.setConnectTimeout(Duration.ofMillis(timeoutMs));
        f.setReadTimeout(Duration.ofMillis(timeoutMs));
        return RestClient.builder().baseUrl(baseUrl).requestFactory(f).build();
    }

    /** 去掉 baseUrl 末尾斜杠，避免 RestClient CONCAT 出 //path。 */
    static String normalize(String baseUrl) {
        return baseUrl == null ? "" : baseUrl.replaceAll("/+$", "");
    }

    private String systemdUnit(ComputeNode node) {
        String repo = node.getRepoPath();
        return """
                [Unit]
                Description=UniKT SaaS inference service (FastAPI, port 8100)
                After=network.target

                [Service]
                Type=simple
                WorkingDirectory=%s
                Environment=HOME=/root
                Environment=PATH=/root/.pixi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
                ExecStart=%s/web-saas/inference/start.sh
                Restart=on-failure
                RestartSec=5

                [Install]
                WantedBy=multi-user.target
                """.formatted(repo, repo);
    }

    /** 单步包装：成功/失败都记日志，失败抛异常终止；节点被删除则中止部署。 */
    private void step(ComputeNode node, String title, StepBody body) {
        node.appendLog("→ " + title);
        saveAlive(node);
        try {
            boolean ok = body.run();
            node.appendLog("  ✓ " + title + " 完成");
            saveAlive(node);
            if (!ok) {
                throw new IllegalStateException(title + " 返回失败");
            }
        } catch (DeployAborted e) {
            throw e;
        } catch (Exception e) {
            fail(node, title + "：" + e.getMessage());
            throw new IllegalStateException(title + " 失败: " + e.getMessage(), e);
        }
    }

    private boolean require(boolean cond, String message) {
        if (!cond) {
            throw new IllegalStateException(message);
        }
        return true;
    }

    private void fail(ComputeNode node, String message) {
        node.appendLog("  ✗ " + message);
        node.setLastError(message);
        node.setStatus(ComputeNode.FAILED);
        saveAlive(node);
    }

    /** 落库；节点已被删除时抛 DeployAborted 终止部署（防僵尸行复活）。 */
    private void saveAlive(ComputeNode node) {
        if (node.getId() != null && !nodes.existsById(node.getId())) {
            throw new DeployAborted("节点已被删除，中止部署");
        }
        nodes.save(node);
    }

    @FunctionalInterface
    private interface StepBody {
        boolean run() throws Exception;
    }

    @PreDestroy
    void shutdown() {
        deployPool.shutdown();
        try {
            if (!deployPool.awaitTermination(3, TimeUnit.SECONDS)) {
                deployPool.shutdownNow();
            }
        } catch (InterruptedException e) {
            deployPool.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }
}
