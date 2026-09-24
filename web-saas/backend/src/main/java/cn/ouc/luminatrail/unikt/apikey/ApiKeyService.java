package cn.ouc.luminatrail.unikt.apikey;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.util.HexFormat;
import org.springframework.stereotype.Service;

/** API Key 的生成与校验（SecureRandom 生成，仅存 SHA-256 哈希）。 */
@Service
public class ApiKeyService {

    private static final String PREFIX = "unikt_";
    private static final int RANDOM_HEX_LEN = 40;

    private final SecureRandom random = new SecureRandom();
    private final ApiKeyRepository keys;

    public ApiKeyService(ApiKeyRepository keys) {
        this.keys = keys;
    }

    /** 生成形如 unikt_&lt;40位hex&gt; 的密钥；返回实体与明文（仅此一次可见）。 */
    public CreatedKey create(String name) {
        StringBuilder sb = new StringBuilder(RANDOM_HEX_LEN);
        for (int i = 0; i < RANDOM_HEX_LEN / 2; i++) {
            sb.append(String.format("%02x", random.nextInt(256)));
        }
        String raw = PREFIX + sb;
        ApiKeyUser user = new ApiKeyUser();
        user.setName(name);
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

    /** 记录一次调用（原子自增，与 verify 分离避免 detached merge）。 */
    @org.springframework.transaction.annotation.Transactional
    public void touch(Long id) {
        keys.touch(id, java.time.Instant.now());
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
