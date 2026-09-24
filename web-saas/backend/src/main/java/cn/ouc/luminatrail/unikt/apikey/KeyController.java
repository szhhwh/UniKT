package cn.ouc.luminatrail.unikt.apikey;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import java.time.Instant;
import java.util.List;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/** API Key 管理（门户内部路由，暂无管理员权限——Phase 3 收口）。 */
@RestController
@RequestMapping("/api/keys")
public class KeyController {

    /** 创建请求。 */
    public record CreateKeyRequest(@NotBlank @Size(max = 60) String name) {
    }

    /** 密钥视图（不含明文）。 */
    public record KeyDto(
            Long id, String name, String keyPrefix, boolean active,
            Instant lastUsedAt, long requestCount, Instant createdAt) {

        static KeyDto from(ApiKeyUser u) {
            return new KeyDto(u.getId(), u.getName(), u.getKeyPrefix(), u.isActive(),
                    u.getLastUsedAt(), u.getRequestCount(), u.getCreatedAt());
        }
    }

    /** 创建成功响应：明文 key 仅此一次返回。 */
    public record CreatedKeyDto(Long id, String name, String key) {
    }

    private final ApiKeyService apiKeys;
    private final ApiKeyRepository keys;

    public KeyController(ApiKeyService apiKeys, ApiKeyRepository keys) {
        this.apiKeys = apiKeys;
        this.keys = keys;
    }

    @GetMapping
    public List<KeyDto> list() {
        return keys.findAll().stream().map(KeyDto::from).toList();
    }

    @PostMapping
    public ResponseEntity<CreatedKeyDto> create(@Valid @RequestBody CreateKeyRequest req) {
        ApiKeyService.CreatedKey created = apiKeys.create(req.name().trim());
        return ResponseEntity.status(HttpStatus.CREATED)
                .cacheControl(org.springframework.http.CacheControl.noStore())
                .body(new CreatedKeyDto(created.user().getId(), created.user().getName(),
                        created.rawKey()));
    }

    /** 吊销（软删除：active=false，历史保留）。 */
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> revoke(@PathVariable Long id) {
        ApiKeyUser user = keys.findById(id).orElse(null);
        if (user == null) {
            return ResponseEntity.notFound().build();
        }
        user.setActive(false);
        keys.save(user);
        return ResponseEntity.noContent().build();
    }
}
