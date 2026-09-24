package cn.ouc.luminatrail.unikt.node;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import java.time.Instant;

/** 计算节点：一台部署了 web-saas/inference 的远程机器（SSH 可达）。 */
@Entity
@Table(name = "compute_node")
public class ComputeNode {

    /** 部署状态机：NEW → DEPLOYING → ONLINE / FAILED；ONLINE 探测失败 → OFFLINE。 */
    public static final String NEW = "NEW";
    public static final String DEPLOYING = "DEPLOYING";
    public static final String ONLINE = "ONLINE";
    public static final String OFFLINE = "OFFLINE";
    public static final String FAILED = "FAILED";

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    /** 展示名，如 "wsl-4060"。 */
    @Column(nullable = false)
    private String name;

    /** SSH 目标（~/.ssh/config 别名或 user@host），限 [A-Za-z0-9@._-]。 */
    @Column(nullable = false)
    private String sshTarget;

    /** 推理服务对外地址（http://host:8100），限 URL 安全字符。 */
    @Column(nullable = false)
    private String baseUrl;

    /** 远端仓库路径（绝对路径，限 [/A-Za-z0-9._-]）。 */
    @Column(nullable = false)
    private String repoPath;

    private String status = NEW;

    private int modelsCount;

    @Column(length = 4000)
    private String lastError;

    /** 最近一次部署日志（追加，保留尾部）。 */
    @Column(length = 16000)
    private String deployLog = "";

    private Instant createdAt = Instant.now();

    private Instant updatedAt = Instant.now();

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getSshTarget() {
        return sshTarget;
    }

    public void setSshTarget(String sshTarget) {
        this.sshTarget = sshTarget;
    }

    public String getBaseUrl() {
        return baseUrl;
    }

    public void setBaseUrl(String baseUrl) {
        this.baseUrl = baseUrl;
    }

    public String getRepoPath() {
        return repoPath;
    }

    public void setRepoPath(String repoPath) {
        this.repoPath = repoPath;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
        touch();
    }

    public int getModelsCount() {
        return modelsCount;
    }

    public void setModelsCount(int modelsCount) {
        this.modelsCount = modelsCount;
        touch();
    }

    public String getLastError() {
        return lastError;
    }

    public void setLastError(String lastError) {
        if (lastError != null && lastError.length() > 3900) {
            lastError = lastError.substring(0, 3900);
        }
        this.lastError = lastError;
    }

    public String getDeployLog() {
        return deployLog == null ? "" : deployLog;
    }

    /** 追加一行部署日志，超出上限时保留尾部。 */
    public void appendLog(String line) {
        String current = getDeployLog();
        String next = current.isEmpty() ? line : current + "\n" + line;
        if (next.length() > 15000) {
            next = "…（截断）…" + next.substring(next.length() - 14000);
        }
        this.deployLog = next;
        touch();
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public Instant getUpdatedAt() {
        return updatedAt;
    }

    private void touch() {
        this.updatedAt = Instant.now();
    }
}
