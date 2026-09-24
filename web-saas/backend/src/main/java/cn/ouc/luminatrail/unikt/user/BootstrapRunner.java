package cn.ouc.luminatrail.unikt.user;

import cn.ouc.luminatrail.unikt.apikey.ApiKeyRepository;
import cn.ouc.luminatrail.unikt.apikey.ApiKeyUser;
import java.security.SecureRandom;
import java.util.HexFormat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Component;

/**
 * 启动引导：无任何用户时创建管理员（密码取 UNIKT_ADMIN_PASSWORD，
 * 未设则随机生成并打印一次），并把历史无主密钥归属给管理员。
 */
@Component
public class BootstrapRunner implements ApplicationRunner {

    private static final Logger log = LoggerFactory.getLogger(BootstrapRunner.class);

    private final UserRepository users;
    private final ApiKeyRepository keys;
    private final PasswordEncoder encoder;

    public BootstrapRunner(UserRepository users, ApiKeyRepository keys, PasswordEncoder encoder) {
        this.users = users;
        this.keys = keys;
        this.encoder = encoder;
    }

    @Override
    public void run(ApplicationArguments args) {
        // 无主密钥归并（幂等，每次启动执行——不止首次引导）
        long adopted = 0;
        for (ApiKeyUser k : keys.findAll()) {
            if (k.getOwnerId() == null) {
                k.setOwnerId(adminId());
                keys.save(k);
                adopted++;
            }
        }
        if (adopted > 0) {
            log.info("已把 {} 枚无主密钥归并给管理员", adopted);
        }
        if (users.count() > 0) {
            return;
        }
        String password = System.getenv("UNIKT_ADMIN_PASSWORD");
        boolean generated = false;
        if (password == null || password.isBlank()) {
            SecureRandom r = new SecureRandom();
            byte[] buf = new byte[6];
            r.nextBytes(buf);
            password = HexFormat.of().formatHex(buf);
            generated = true;
        }
        User admin = new User();
        admin.setUsername("admin");
        admin.setPasswordHash(encoder.encode(password));
        admin.setRole(User.ROLE_ADMIN);
        users.save(admin);
        // 历史无主密钥归并给管理员
        for (ApiKeyUser k : keys.findAll()) {
            if (k.getOwnerId() == null) {
                k.setOwnerId(admin.getId());
                keys.save(k);
            }
        }
        if (generated) {
            log.warn("已创建管理员 admin，随机密码（仅显示这一次，请立即保存）：{}", password);
        } else {
            log.info("已创建管理员 admin（密码来自 UNIKT_ADMIN_PASSWORD）");
        }
    }

    private Long adminId() {
        return users.findByUsername("admin").map(u -> u.getId()).orElse(null);
    }
}
