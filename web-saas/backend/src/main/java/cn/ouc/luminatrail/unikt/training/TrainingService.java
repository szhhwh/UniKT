package cn.ouc.luminatrail.unikt.training;

import cn.ouc.luminatrail.unikt.dataset.DatasetIngestService;
import cn.ouc.luminatrail.unikt.dataset.DatasetRepository;
import cn.ouc.luminatrail.unikt.dataset.UserDataset;
import cn.ouc.luminatrail.unikt.node.ComputeNode;
import cn.ouc.luminatrail.unikt.node.NodeRepository;
import cn.ouc.luminatrail.unikt.node.SshExecutor;
import jakarta.validation.constraints.Pattern;
import java.time.Instant;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.data.domain.Sort;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

/**
 * 训练编排：把「数据集 × 模型」送到计算节点执行（数据 rsync → 后台
 * process+train → 轮询日志识别完成标记），run 目录被节点推理服务自动
 * 发现即上线。每节点同时只跑一个任务。
 */
@Service
public class TrainingService {

    private static final Logger log = LoggerFactory.getLogger(TrainingService.class);

    private final JobRepository jobs;
    private final Object createLock = new Object();
    private final java.util.concurrent.ExecutorService orchestrator =
            java.util.concurrent.Executors.newFixedThreadPool(2, r -> {
                Thread t3 = new Thread(r, "train-orchestrator");
                t3.setDaemon(true);
                return t3;
            });
    private final DatasetRepository datasets;
    private final NodeRepository nodes;
    private final SshExecutor ssh;
    private final DatasetIngestService ingest;
    private final cn.ouc.luminatrail.unikt.node.NodeRoutingService routing;

    public TrainingService(JobRepository jobs, DatasetRepository datasets,
                           NodeRepository nodes, SshExecutor ssh,
                           DatasetIngestService ingest,
                           cn.ouc.luminatrail.unikt.node.NodeRoutingService routing) {
        this.jobs = jobs;
        this.datasets = datasets;
        this.nodes = nodes;
        this.ssh = ssh;
        this.ingest = ingest;
        this.routing = routing;
    }

    /** 创建并启动训练。 */
    public TrainingJob create(Long ownerId, Long datasetId, String modelName, int epochs) {
        UserDataset ds = datasets.findById(datasetId).orElse(null);
        if (ds == null || !ownerId.equals(ds.getOwnerId())) {
            throw new IllegalArgumentException("数据集不存在或不属于你");
        }
        if (!UserDataset.READY.equals(ds.getStatus())) {
            throw new IllegalArgumentException("数据集未就绪");
        }
        if (!modelName.matches("[A-Za-z0-9]+") || modelName.length() > 40) {
            throw new IllegalArgumentException("模型名不合法");
        }
        // 提交前校验模型名（走 NodeRoutingService 的缓存清单——同步扇出
        // 逐节点拉 /models 会占请求线程最坏 N×15s，且无缓存）
        boolean anyOnline = nodes.findAll(Sort.by("id")).stream()
                .anyMatch(n -> ComputeNode.ONLINE.equals(n.getStatus()));
        if (!anyOnline) {
            throw new IllegalStateException("当前没有在线计算节点，请联系管理员");
        }
        boolean known = routing.aggregateModels(null, false).stream()
                .anyMatch(m -> modelName.equals(m.name()));
        if (!known) {
            throw new IllegalArgumentException(
                    "未知模型 '" + modelName + "'——在训练页下拉里选择可用的模型名");
        }
        // 每用户非终态任务上限：防脚本刷满节点池
        if (jobs.countByOwnerIdAndStatusIn(ownerId,
                java.util.List.of(TrainingJob.RUNNING)) >= 3) {
            throw new IllegalStateException("你已有 3 个进行中的任务，等完成后再提交");
        }
        ComputeNode node;
        TrainingJob job;
        synchronized (createLock) {
            // pick+count+save 原子化（单实例足够），并遍历找空闲节点而非只看第一个
            node = nodes.findAll(Sort.by("id")).stream()
                    .filter(n -> ComputeNode.ONLINE.equals(n.getStatus()))
                    .filter(n -> jobs.countByNodeIdAndStatus(
                            n.getId(), TrainingJob.RUNNING) == 0)
                    .findFirst().orElse(null);
            if (node == null) {
                throw new IllegalStateException(anyOnline
                        ? "所有在线节点都有训练任务在跑（每节点同时 1 个），稍后再试"
                        : "当前没有在线计算节点，请联系管理员");
            }
            // 占位任务先落库（RUNNING），并发者随即被 count 拦住

            TrainingJob placeholder = new TrainingJob();
            placeholder.setOwnerId(ownerId);
            placeholder.setDatasetId(datasetId);
            placeholder.setModelName(modelName);
            placeholder.setNodeId(node.getId());
            placeholder.setEpochs(epochs);
            placeholder.setDatasetName(ds.getName());
            job = jobs.save(placeholder); // 直接用返回值（含 id），不回查
        }

        // rsync 与启动放后台：慢节点不再挂住 HTTP 请求（此前可挂 11 分钟）
        final ComputeNode theNode = node;
        orchestrator.submit(() -> {
            try {
                launch(job, ds, theNode);
            } catch (Exception e) {
                log.error("launch job {} failed", job.getId(), e);
                fail(job, "启动失败: " + e.getMessage());
            }
        });
        return job;
    }

    /** 在节点上执行数据同步与后台训练启动（orchestrator 线程内）。 */
    private void launch(TrainingJob job, UserDataset ds, ComputeNode node) {
        // 异步执行期间用户可能已取消（rsync 大数据集要几分钟）：每步前
        // 查库状态，非 RUNNING 即中止；落库一律走条件 UPDATE（状态守卫），
        // 从根上消灭"旧对象覆盖取消写入"的复活竞态
        if (aborted(job)) {
            return;
        }
        String rawDir = node.getRepoPath() + "/data/" + ds.getSlug() + "/raw";
        SshExecutor.SshResult mkdir = ssh.run(node.getSshTarget(),
                "mkdir -p " + SshExecutor.shellQuote(rawDir), 60);
        if (!mkdir.ok()) {
            fail(job, "创建远端目录失败: " + mkdir.stderr());
            return;
        }
        if (aborted(job)) {
            return;
        }
        SshExecutor.SshResult sync = ssh.rsync(
                ingest.stagedDir(job.getDatasetId()).toAbsolutePath().toString(),
                node.getSshTarget(), rawDir, 600);
        if (!sync.ok()) {
            fail(job, "数据同步失败: " + sync.stderr());
            return;
        }
        if (aborted(job)) {
            return;
        }

        // 节点后台执行（setsid 独立进程组；LD_PRELOAD 绝对路径）。
        // 启动后立即回读该 sh 的 PGID，取消/超时按组精确 kill。
        String logFile = "/tmp/unikt-job-" + job.getId() + ".log";
        String pidFile = "/tmp/unikt-job-" + job.getId() + ".pgid";
        String preload = node.getRepoPath() + "/.pixi/envs/cpu/lib/libstdc++.so.6";
        // inner 的 $$ 即 setsid 出的新会话组长 PID（=PGID），写入 pidfile；
        // 取消/超时读 pidfile 精确 kill -- -PGID（echo $! 外层捕获不稳）
        String inner = "echo $$ > " + SshExecutor.shellQuote(pidFile) + "; "
                + "export PATH=$HOME/.pixi/bin:$PATH LD_PRELOAD=" + preload + "; "
                + "cd " + SshExecutor.shellQuote(node.getRepoPath()) + " && "
                + "pixi run -e cpu python data_process.py process -d " + ds.getSlug()
                + " --extra=[windowlate] && "
                + "pixi run -e cpu python train.py -m " + job.getModelName()
                + " -d " + ds.getSlug() + " --model.epochs " + job.getEpochs()
                + " --early_stopping.patience 3; "
                + "echo UNIKT_EXIT=$?";
        String cmd = "setsid sh -c " + SshExecutor.shellQuote(inner)
                + " > " + SshExecutor.shellQuote(logFile) + " 2>&1 < /dev/null &";
        SshExecutor.SshResult kick = ssh.run(node.getSshTarget(), cmd, 60);
        if (!kick.ok()) {
            fail(job, "启动训练失败: " + kick.stderr());
            return;
        }
        // 条件写（仅 RUNNING 生效）：取消若发生在 ssh 往返期间，这里返回
        // 0 行——把刚启动的进程组清掉，绝不复活
        int marked = jobs.markLaunched(job.getId(),
                "训练已启动（节点 " + node.getName() + "，日志 " + logFile + "）",
                Instant.now(), TrainingJob.RUNNING);
        if (marked == 0) {
            killRemote(node, job);
        }
    }

    /** 任务已被取消/删除则 true（launch 各步骤间的中止检查）。 */
    private boolean aborted(TrainingJob job) {
        TrainingJob latest = jobs.findById(job.getId()).orElse(null);
        return latest == null || !TrainingJob.RUNNING.equals(latest.getStatus());
    }

    /** 后台轮询 RUNNING 任务。 */
    @Scheduled(fixedDelay = 10000)
    public void pollJobs() {
        for (TrainingJob job : jobs.findAll().stream()
                .filter(j -> TrainingJob.RUNNING.equals(j.getStatus())).toList()) {
            try {
                pollOne(job);
            } catch (Exception e) {
                log.debug("poll job {} failed: {}", job.getId(), e.getMessage());
            }
        }
    }

    /** 从节点 pidfile 读该任务的 PGID（读不到返回 null）。 */
    private Long readPgid(ComputeNode node, long jobId) {
        SshExecutor.SshResult r = ssh.run(node.getSshTarget(),
                "cat /tmp/unikt-job-" + jobId + ".pgid 2>/dev/null", 30);
        try {
            return Long.parseLong(r.stdout().strip());
        } catch (NumberFormatException e) {
            return null;
        }
    }

    /** 杀远端进程组（pidfile 是唯一锚点——remotePgid 已不再赋值）。 */
    private void killRemote(ComputeNode node, TrainingJob job) {
        Long pgid = readPgid(node, job.getId());
        if (pgid != null) {
            ssh.run(node.getSshTarget(),
                    "kill -- -" + pgid + " 2>/dev/null; true", 30);
        }
    }

    private void pollOne(TrainingJob job) {
        ComputeNode node = nodes.findById(job.getNodeId()).orElse(null);
        if (node == null) {
            // 节点被删：条件写终结（不覆盖用户刚写入的"已取消"）
            jobs.markFailed(job.getId(), "计算节点已被删除，任务结局未知；可重新提交",
                    Instant.now(), TrainingJob.RUNNING);
            return;
        }
        // 超时兜底：节点重启会丢 /tmp 日志（永不出现 EXIT 标记）、SSH 长期
        // 不通、进程被杀等都会让任务卡 RUNNING 并锁死该节点的训练名额
        java.time.Duration age = java.time.Duration.between(
                job.getCreatedAt(), Instant.now());
        java.time.Duration stale = java.time.Duration.between(
                job.getUpdatedAt(), Instant.now());
        if (stale.toMinutes() > 30 || age.toHours() >= 6) {
            // 超时也要杀远端进程：否则槽位释放但 python 还在跑，
            // 下一个任务与它在同节点并发写 runs/ 破坏"每节点 1 任务"
            killRemote(node, job);
            jobs.markFailed(job.getId(), "任务长时间无进展（节点失联或重启），已终止；可重新提交",
                    Instant.now(), TrainingJob.RUNNING);
            return;
        }
        String logFile = "/tmp/unikt-job-" + job.getId() + ".log";
        SshExecutor.SshResult r = ssh.run(node.getSshTarget(),
                "tail -c 4000 " + SshExecutor.shellQuote(logFile)
                        + " 2>/dev/null; echo ---MARKER---; "
                        + "grep -o 'UNIKT_EXIT=[0-9]*' "
                        + SshExecutor.shellQuote(logFile) + " 2>/dev/null | tail -1",
                60);
        if (!r.ok()) {
            return; // SSH 暂时不通：下轮再查
        }
        String out = r.stdout();
        String tail = truncateLog(out.contains("---MARKER---")
                ? out.substring(0, out.lastIndexOf("---MARKER---")) : out);
        // 终态与最终日志同一条 UPDATE（若先改状态再补日志，后者的
        // WHERE status='RUNNING' 永远 0 行——失败原因/最终指标会丢）
        if (out.contains("UNIKT_EXIT=0")) {
            String runDir = findRunDir(node, job);
            jobs.completeWithLog(job.getId(),
                    runDir == null ? null : truncate(runDir, 1000), tail.strip(),
                    Instant.now(), TrainingJob.RUNNING);
        } else if (out.contains("UNIKT_EXIT=")) {
            jobs.markFailedWithLog(job.getId(), "训练失败，看下方日志尾部定位原因",
                    tail.strip(), Instant.now(), TrainingJob.RUNNING);
        } else {
            // 仍在跑：仅当日志内容变化才刷 updatedAt（超时判定依据，
            // 卡死任务不能自己续命）
            jobs.appendLogIfRunning(job.getId(), tail.strip(),
                    Instant.now(), TrainingJob.RUNNING);
        }
    }

    /** 日志尾截断（与实体列宽一致，绕过 setter 的路径走同一规则）。 */
    private static String truncateLog(String s) {
        if (s != null && s.length() > 3600) {
            return "…（截断）…" + s.substring(s.length() - 3400);
        }
        return s;
    }

    /** 找该任务产出的 run 目录（&lt;模型&gt;_&lt;数据集&gt; 前缀取最新；两段均已白名单校验）。 */
    private String findRunDir(ComputeNode node, TrainingJob job) {
        UserDataset ds = datasets.findById(job.getDatasetId()).orElse(null);
        if (ds == null) {
            return null;
        }
        SshExecutor.SshResult r = ssh.run(node.getSshTarget(),
                "ls -1dt " + node.getRepoPath() + "/runs/normal/"
                        + job.getModelName() + "_" + ds.getSlug() + "_* 2>/dev/null | head -1",
                60);
        return r.ok() && !r.stdout().isBlank() ? r.stdout().strip() : null;
    }

    /** 取消任务（RUNNING 则尽力杀远端进程；任何状态都终结为 FAILED）。 */
    public TrainingJob cancel(Long ownerId, boolean admin, Long jobId) {
        TrainingJob job = jobs.findById(jobId).orElse(null);
        if (job == null) {
            throw new IllegalArgumentException("任务不存在");
        }
        if (!admin && !ownerId.equals(job.getOwnerId())) {
            throw new IllegalArgumentException("任务不属于你");
        }
        if (TrainingJob.RUNNING.equals(job.getStatus())) {
            // 按进程组精确杀（此前 pkill -f 'train.py -m X' 会误杀别人
            // 同模型名的任务；pidfile 记录的 PGID 是唯一可靠锚点）
            nodes.findById(job.getNodeId()).ifPresent(node -> killRemote(node, job));
        }
        jobs.markFailed(job.getId(), "已取消", Instant.now(), TrainingJob.RUNNING);
        return jobs.findById(job.getId()).orElse(job);
    }

    /** 标记失败（仅 RUNNING 时生效：不覆盖用户取消写入的"已取消"）。 */
    private TrainingJob fail(TrainingJob job, String message) {
        jobs.markFailed(job.getId(), truncate(message, 3800),
                Instant.now(), TrainingJob.RUNNING);
        return jobs.findById(job.getId()).orElse(job);
    }

    /** lastError 列宽 3900 的统一截断（JPQL 绕过实体 setter）。 */
    private static String truncate(String s, int max) {
        if (s != null && s.length() > max) {
            return s.substring(0, max);
        }
        return s;
    }

    @jakarta.annotation.PreDestroy
    void shutdown() {
        orchestrator.shutdown();
    }

}
