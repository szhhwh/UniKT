package cn.ouc.luminatrail.unikt.apikey;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import java.time.Instant;
import java.util.List;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import cn.ouc.luminatrail.unikt.user.User;

/**
 * API Key 管理：登录用户管理**自己的**密钥；管理员可用 ?scope=all 看全部。
 * 吊销只能针对自己的密钥（他人的返回 403）。
 */
@RestController
@RequestMapping("/api/keys")
public class KeyController {

    /** 创建请求。 */
    public record CreateKeyRequest(@NotBlank @Size(max = 60) String name) {
    }

    /** 密钥视图（不含明文）。 */
    public record KeyDto(
            Long id, String name, String keyPrefix, boolean active,
            Instant lastUsedAt, long requestCount, long dailyCount, Instant createdAt) {

        static KeyDto from(ApiKeyUser u) {
            return new KeyDto(u.getId(), u.getName(), u.getKeyPrefix(), u.isActive(),
                    u.getLastUsedAt(), u.getRequestCount(), u.getDailyCount(), u.getCreatedAt());
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
    public List<KeyDto> list(@RequestParam(required = false) String scope) {
        if (isAdmin() && "all".equals(scope)) {
            return keys.findAll().stream()
                    .sorted((a, b) -> Long.compare(b.getId(), a.getId()))
                    .map(KeyDto::from)
                    .toList();
        }
        return keys.findByOwnerIdOrderByIdDesc(currentUserId()).stream()
                .map(KeyDto::from)
                .toList();
    }

    @PostMapping
    public ResponseEntity<CreatedKeyDto> create(@Valid @RequestBody CreateKeyRequest req) {
        ApiKeyService.CreatedKey created = apiKeys.create(req.name().trim(), currentUserId());
        return ResponseEntity.status(HttpStatus.CREATED)
                .cacheControl(org.springframework.http.CacheControl.noStore())
                .body(new CreatedKeyDto(created.user().getId(), created.user().getName(),
                        created.rawKey()));
    }

    /** 吊销（软删除）；只能吊销自己的。 */
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> revoke(@PathVariable Long id) {
        ApiKeyUser user = keys.findById(id).orElse(null);
        if (user == null) {
            return ResponseEntity.notFound().build();
        }
        if (!java.util.Objects.equals(user.getOwnerId(), currentUserId()) && !isAdmin()) {
            return ResponseEntity.status(HttpStatus.FORBIDDEN).build();
        }
        user.setActive(false);
        keys.save(user);
        return ResponseEntity.noContent().build();
    }

    private static Long currentUserId() {
        Authentication auth = currentAuth();
        if (auth != null && auth.getPrincipal() instanceof
                cn.ouc.luminatrail.unikt.user.PortalPrincipal p) {
            return p.getId();
        }
        return null;
    }

    private static Authentication currentAuth() {
        return SecurityContextHolder.getContext().getAuthentication();
    }

    private static boolean isAdmin() {
        Authentication auth = currentAuth();
        return auth != null
                && auth.getAuthorities().stream()
                        .anyMatch(a -> "ROLE_ADMIN".equals(a.getAuthority()));
    }
}
