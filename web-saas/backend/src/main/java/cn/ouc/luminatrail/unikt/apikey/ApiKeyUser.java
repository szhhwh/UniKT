package cn.ouc.luminatrail.unikt.apikey;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import java.time.Instant;

/** 开放 API 的调用方：按 API Key 鉴权（密钥只存哈希）。 */
@Entity
@Table(name = "api_key_user")
public class ApiKeyUser {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    /** 调用方名称，如 "刷题 App 后端"。 */
    @Column(nullable = false)
    private String name;

    /** SHA-256(key) 的十六进制，全键唯一。 */
    @Column(nullable = false, unique = true, length = 64)
    private String keyHash;

    /** key 前缀（展示用，如 unikt_a1b2c3d4…），不可用于鉴权。 */
    @Column(nullable = false)
    private String keyPrefix;

    /** 归属用户；null = 历史遗留（启动时归并给管理员）。 */
    private Long ownerId;

    private boolean active = true;

    private Instant lastUsedAt;

    private long requestCount;

    /** 每日配额计数（跨天自动重置）。default 0 兼容旧表迁移。 */
    @Column(nullable = false, columnDefinition = "bigint default 0")
    private long dailyCount;

    /** 计数所属日期（yyyy-MM-dd，UTC）。default '' 兼容旧表迁移。 */
    @Column(length = 10, columnDefinition = "varchar(10) default ''")
    private String dailyDate;

    private Instant createdAt = Instant.now();

    public Long getId() {
        return id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getKeyHash() {
        return keyHash;
    }

    public void setKeyHash(String keyHash) {
        this.keyHash = keyHash;
    }

    public String getKeyPrefix() {
        return keyPrefix;
    }

    public void setKeyPrefix(String keyPrefix) {
        this.keyPrefix = keyPrefix;
    }

    public Long getOwnerId() {
        return ownerId;
    }

    public void setOwnerId(Long ownerId) {
        this.ownerId = ownerId;
    }

    public boolean isActive() {
        return active;
    }

    public void setActive(boolean active) {
        this.active = active;
    }

    public Instant getLastUsedAt() {
        return lastUsedAt;
    }

    public void setLastUsedAt(Instant lastUsedAt) {
        this.lastUsedAt = lastUsedAt;
    }

    public long getRequestCount() {
        return requestCount;
    }

    public void setRequestCount(long requestCount) {
        this.requestCount = requestCount;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public long getDailyCount() {
        return dailyCount;
    }

    public void setDailyCount(long dailyCount) {
        this.dailyCount = dailyCount;
    }

    public String getDailyDate() {
        return dailyDate;
    }

    public void setDailyDate(String dailyDate) {
        this.dailyDate = dailyDate;
    }
}
