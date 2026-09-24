package cn.ouc.luminatrail.unikt.training;

import cn.ouc.luminatrail.unikt.dataset.DatasetIngestService;
import cn.ouc.luminatrail.unikt.dataset.DatasetRepository;
import cn.ouc.luminatrail.unikt.dataset.UserDataset;
import cn.ouc.luminatrail.unikt.node.ComputeNode;
import cn.ouc.luminatrail.unikt.node.NodeRepository;
import cn.ouc.luminatrail.unikt.node.SshExecutor;
import jakarta.validation.constraints.Pattern;
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
    private final DatasetRepository datasets;
    private final NodeRepository nodes;
    private final SshExecutor ssh;
    private final DatasetIngestService ingest;

    public TrainingService(JobRepository jobs, DatasetRepository datasets,
                           NodeRepository nodes, SshExecutor ssh,
                           DatasetIngestService ingest) {
        this.jobs = jobs;
        this.datasets = datasets;
        this.nodes = nodes;
        this.ssh = ssh;
        this.ingest = ingest;
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
        ComputeNode node = pickNode();
        if (node == null) {
            throw new IllegalStateException("当前没有在线计算节点，请联系管理员");
        }
        if (jobs.countByNodeIdAndStatus(node.getId(), TrainingJob.RUNNING) > 0) {
            throw new IllegalStateException("该节点已有训练任务在跑（每节点同时 1 个），稍后再试");
        }

        TrainingJob job = new TrainingJob();
        job.setOwnerId(ownerId);
        job.setDatasetId(datasetId);
        job.setModelName(modelName);
        job.setNodeId(node.getId());
        job.setEpochs(epochs);
        job = jobs.save(job);

        // 1. 数据同步到节点 raw/（mkdir + rsync）
        String rawDir = node.getRepoPath() + "/data/" + ds.getSlug() + "/raw";
        SshExecutor.SshResult mkdir = ssh.run(node.getSshTarget(),
                "mkdir -p " + SshExecutor.shellQuote(rawDir), 60);
        if (!mkdir.ok()) {
            return fail(job, "创建远端目录失败: " + mkdir.stderr());
        }
        SshExecutor.SshResult sync = ssh.rsync(
                ingest.stagedDir(datasetId).toAbsolutePath().toString(),
                node.getSshTarget(), rawDir, 600);
        if (!sync.ok()) {
            return fail(job, "数据同步失败: " + sync.stderr());
        }

        // 2. 节点后台执行（setsid 防 SSH 会话回收；LD_PRELOAD 绝对路径）
        String logFile = "/tmp/unikt-job-" + job.getId() + ".log";
        String preload = node.getRepoPath() + "/.pixi/envs/cpu/lib/libstdc++.so.6";
        String inner = "export PATH=$HOME/.pixi/bin:$PATH LD_PRELOAD=" + preload + "; "
                + "cd " + SshExecutor.shellQuote(node.getRepoPath()) + " && "
                + "pixi run -e cpu python data_process.py process -d " + ds.getSlug()
                + " --extra=[windowlate] && "
                + "pixi run -e cpu python train.py -m " + modelName
                + " -d " + ds.getSlug() + " --model.epochs " + epochs
                + " --early_stopping.patience 3; "
                + "echo UNIKT_EXIT=$?";
        String cmd = "setsid sh -c " + SshExecutor.shellQuote(inner)
                + " > " + SshExecutor.shellQuote(logFile) + " 2>&1 < /dev/null &";
        SshExecutor.SshResult kick = ssh.run(node.getSshTarget(), cmd, 60);
        if (!kick.ok()) {
            return fail(job, "启动训练失败: " + kick.stderr());
        }
        job.setLogTail("训练已启动（节点 " + node.getName() + "，日志 " + logFile + "）");
        return jobs.save(job);
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

    private void pollOne(TrainingJob job) {
        ComputeNode node = nodes.findById(job.getNodeId()).orElse(null);
        if (node == null) {
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
        String tail = out.contains("---MARKER---")
                ? out.substring(0, out.lastIndexOf("---MARKER---")) : out;
        job.setLogTail(tail.strip());
        if (out.contains("UNIKT_EXIT=0")) {
            job.setRunDir(findRunDir(node, job));
            job.setStatus(TrainingJob.DONE);
        } else if (out.contains("UNIKT_EXIT=")) {
            job.setStatus(TrainingJob.FAILED);
            job.setLastError("训练失败，看日志尾部定位原因");
        }
        jobs.save(job);
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

    private TrainingJob fail(TrainingJob job, String message) {
        job.setStatus(TrainingJob.FAILED);
        job.setLastError(message);
        return jobs.save(job);
    }

    private ComputeNode pickNode() {
        return nodes.findAll(Sort.by("id")).stream()
                .filter(n -> ComputeNode.ONLINE.equals(n.getStatus()))
                .findFirst().orElse(null);
    }
}
