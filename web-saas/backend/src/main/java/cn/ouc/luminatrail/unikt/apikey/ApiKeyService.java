package cn.ouc.luminatrail.unikt.apikey;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.time.LocalDate;
import java.time.ZoneOffset;
import java.util.HexFormat;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/** API Key 的生成、校验与每日配额（SecureRandom 生成，仅存 SHA-256 哈希）。 */
@Service
public class ApiKeyService {

    private static final String PREFIX = "unikt_";
    private static final int RANDOM_HEX_LEN = 40;

    private final SecureRandom random = new SecureRandom();
    private final ApiKeyRepository keys;

    public ApiKeyService(ApiKeyRepository keys) {
        this.keys = keys;
    }

    /** 生成形如 unikt_&lt;40位hex&gt; 的密钥，归属 owner；明文仅此一次可见。 */
    public CreatedKey create(String name, Long ownerId) {
        StringBuilder sb = new StringBuilder(RANDOM_HEX_LEN);
        for (int i = 0; i < RANDOM_HEX_LEN / 2; i++) {
            sb.append(String.format("%02x", random.nextInt(256)));
        }
        String raw = PREFIX + sb;
        ApiKeyUser user = new ApiKeyUser();
        user.setName(name);
        user.setOwnerId(ownerId);
        user.setKeyHash(sha256(raw));
        user.setKeyPrefix(raw.substring(0, PREFIX.length() + 8) + "…");
        return new CreatedKey(keys.save(user), raw);
    }

    /** 校验密钥：存在且启用则返回实体（不写库）；否则 null。 */
    public ApiKeyUser verify(String rawKey) {
        if (rawKey == null || rawKey.isBlank()) {
            return null;
        }
        ApiKeyUser user = keys.findByKeyHash(sha256(rawKey)).orElse(null);
        if (user == null || !user.isActive()) {
            return null;
        }
        return user;
    }

    /**
     * 消耗一次当日配额（原子自增用量 + 跨天重置）。
     *
     * @return false 表示配额已满（本次未消耗）
     */
    @Transactional
    public boolean tryConsumeQuota(ApiKeyUser keyUser, long dailyLimit) {
        ApiKeyUser fresh = keys.findById(keyUser.getId()).orElse(null);
        if (fresh == null) {
            return false;
        }
        String today = LocalDate.now(ZoneOffset.UTC).toString();
        if (!today.equals(fresh.getDailyDate())) {
            fresh.setDailyDate(today);
            fresh.setDailyCount(0);
        }
        if (fresh.getDailyCount() >= dailyLimit) {
            return false;
        }
        int updated = keys.consume(fresh.getId(), today, fresh.getDailyCount() + 1,
                java.time.Instant.now());
        return updated > 0;
    }

    /** record 与实体一并返回（明文 key 不落库；toString 掩码防误打印）。 */
    public record CreatedKey(ApiKeyUser user, String rawKey) {

        @Override
        public String toString() {
            return "CreatedKey[user=" + user.getId() + ", rawKey=****]";
        }
    }

    static String sha256(String s) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            return HexFormat.of().formatHex(md.digest(s.getBytes(StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 不可用", e);
        }
    }
}
