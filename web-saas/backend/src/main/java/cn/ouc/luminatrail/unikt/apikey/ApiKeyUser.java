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

    private boolean active = true;

    private Instant lastUsedAt;

    private long requestCount;

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
}
