package cn.ouc.luminatrail.unikt.user;

import cn.ouc.luminatrail.unikt.apikey.ApiKeyRepository;
import jakarta.validation.Valid;
import jakarta.validation.constraints.Pattern;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PatchMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/** 管理员：用户列表与角色/启停管理。 */
@RestController
@RequestMapping("/api/admin/users")
public class AdminUserController {

    /** 用户视图。 */
    public record UserDto(Long id, String username, String role, boolean active,
                          long keyCount, Instant createdAt) {
    }

    /** 更新请求（字段可选）。 */
    public record UpdateUserRequest(
            @Pattern(regexp = "USER|ADMIN", message = "角色只能是 USER 或 ADMIN") String role,
            Boolean active) {
    }

    private final UserRepository users;
    private final ApiKeyRepository keys;

    public AdminUserController(UserRepository users, ApiKeyRepository keys) {
        this.users = users;
        this.keys = keys;
    }

    @GetMapping
    public List<UserDto> list() {
        return users.findAll().stream()
                .map(u -> new UserDto(u.getId(), u.getUsername(), u.getRole(), u.isActive(),
                        keys.findByOwnerIdOrderByIdDesc(u.getId()).size(), u.getCreatedAt()))
                .toList();
    }

    @PatchMapping("/{id}")
    public ResponseEntity<?> update(@PathVariable Long id,
                                    @Valid @RequestBody UpdateUserRequest req) {
        User user = users.findById(id).orElse(null);
        if (user == null) {
            return ResponseEntity.notFound().build();
        }
        if (id.equals(currentUserId()) && (req.active() != null && !req.active()
                || "USER".equals(req.role()))) {
            return ResponseEntity.status(409)
                    .body(Map.of("message", "不能降级或禁用自己的管理员账号"));
        }
        if (req.role() != null) {
            user.setRole(req.role());
        }
        if (req.active() != null) {
            user.setActive(req.active());
        }
        users.save(user);
        return ResponseEntity.ok().build();
    }

    private static Long currentUserId() {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        if (auth != null && auth.getPrincipal() instanceof PortalPrincipal p) {
            return p.getId();
        }
        return null;
    }
}
