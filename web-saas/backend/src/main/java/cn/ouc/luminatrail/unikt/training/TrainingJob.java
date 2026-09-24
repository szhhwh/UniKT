package cn.ouc.luminatrail.unikt.training;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import java.time.Instant;

/** 训练任务：数据集 × 模型 × 节点的一次编排执行。 */
@Entity
@Table(name = "training_job")
public class TrainingJob {

    public static final String RUNNING = "RUNNING";
    public static final String DONE = "DONE";
    public static final String FAILED = "FAILED";

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private Long ownerId;

    @Column(nullable = false)
    private Long datasetId;

    @Column(nullable = false, length = 40)
    private String modelName;

    @Column(nullable = false)
    private Long nodeId;

    private int epochs;

    /** 数据集展示名快照（数据集删除后任务仍可读）。 */
    @Column(length = 60)
    private String datasetName;

    private String status = RUNNING;

    private String runDir;

    @Column(length = 4000)
    private String logTail = "";

    @Column(length = 3900)
    private String lastError;

    private Instant createdAt = Instant.now();

    private Instant updatedAt = Instant.now();

    public Long getId() {
        return id;
    }

    public Long getOwnerId() {
        return ownerId;
    }

    public void setOwnerId(Long ownerId) {
        this.ownerId = ownerId;
    }

    public Long getDatasetId() {
        return datasetId;
    }

    public void setDatasetId(Long datasetId) {
        this.datasetId = datasetId;
    }

    public String getModelName() {
        return modelName;
    }

    public void setModelName(String modelName) {
        this.modelName = modelName;
    }

    public Long getNodeId() {
        return nodeId;
    }

    public void setNodeId(Long nodeId) {
        this.nodeId = nodeId;
    }

    public String getDatasetName() {
        return datasetName;
    }

    public void setDatasetName(String datasetName) {
        this.datasetName = datasetName;
    }

    public int getEpochs() {
        return epochs;
    }

    public void setEpochs(int epochs) {
        this.epochs = epochs;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
        this.updatedAt = Instant.now();
    }

    public String getRunDir() {
        return runDir;
    }

    public void setRunDir(String runDir) {
        this.runDir = runDir;
    }

    public String getLogTail() {
        return logTail == null ? "" : logTail;
    }

    public void setLogTail(String logTail) {
        if (logTail != null && logTail.length() > 3800) {
            logTail = "…（截断）…" + logTail.substring(logTail.length() - 3600);
        }
        // 只有内容真正变化才 touch updatedAt——它是超时判断的依据，
        // 日志停更时不能自己给自己续命
        if (!java.util.Objects.equals(this.logTail, logTail)) {
            this.logTail = logTail;
            this.updatedAt = Instant.now();
        } else {
            this.logTail = logTail;
        }
    }

    public String getLastError() {
        return lastError;
    }

    public void setLastError(String lastError) {
        if (lastError != null && lastError.length() > 3800) {
            lastError = lastError.substring(0, 3800);
        }
        this.lastError = lastError;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public Instant getUpdatedAt() {
        return updatedAt;
    }
}
