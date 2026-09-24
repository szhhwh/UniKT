package cn.ouc.luminatrail.unikt.user;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import java.util.Map;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.AuthenticationException;
import org.springframework.security.core.context.SecurityContext;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.web.context.SecurityContextRepository;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

/** 注册 / 登录 / 登出 / 当前用户。 */
@RestController
@RequestMapping("/api/auth")
public class AuthController {

    /** 注册请求。 */
    public record RegisterRequest(
            @NotBlank @Size(min = 3, max = 30)
            @Pattern(regexp = "[a-zA-Z0-9_]+", message = "用户名只能含字母数字下划线")
                    String username,
            @NotBlank @Size(min = 8, max = 64, message = "密码至少 8 位") String password) {
    }

    public record LoginRequest(@NotBlank String username, @NotBlank String password) {
    }

    public record MeResponse(String username, String role) {
    }

    private final UserRepository users;
    private final PasswordEncoder encoder;
    private final AuthenticationManager authManager;
    private final SecurityContextRepository contextRepo;
    private final cn.ouc.luminatrail.unikt.common.RateLimiter limiter;

    public AuthController(UserRepository users, PasswordEncoder encoder,
                          AuthenticationManager authManager,
                          SecurityContextRepository contextRepo,
                          cn.ouc.luminatrail.unikt.common.RateLimiter limiter) {
        this.users = users;
        this.encoder = encoder;
        this.authManager = authManager;
        this.contextRepo = contextRepo;
        this.limiter = limiter;
    }

    /** 登录失败锁定：同用户名连续失败 5 次锁 10 分钟（成功登录即清零）。 */
    private record FailState(int count, long lockUntil, long lastFailAt) {
    }

    private static final java.util.Map<String, FailState> loginFails =
            new java.util.concurrent.ConcurrentHashMap<>();
    private static final int MAX_FAILS = 5;
    private static final long LOCK_MS = 10 * 60_000L;

    @PostMapping("/register")
    public ResponseEntity<?> register(@Valid @RequestBody RegisterRequest req,
                                      HttpServletRequest request,
                                      HttpServletResponse response) {
        // 防 script 批量刷注册：同 IP 每小时最多 5 个
        if (!limiter.tryAcquire("register:"
                + cn.ouc.luminatrail.unikt.common.Ips.clientIp(request), 5, 3600_000)) {
            return ResponseEntity.status(429)
                    .body(Map.of("message", "注册太频繁，一小时后再试"));
        }
        if (users.existsByUsername(req.username())) {
            return ResponseEntity.status(HttpStatus.CONFLICT)
                    .body(Map.of("message", "用户名已被占用"));
        }
        User user = new User();
        user.setUsername(req.username());
        user.setPasswordHash(encoder.encode(req.password()));
        users.save(user);
        // 注册即登录
        loginSession(req.username(), req.password(), request, response);
        loginFails.remove(req.username());
        return ResponseEntity.status(HttpStatus.CREATED)
                .body(new MeResponse(user.getUsername(), user.getRole()));
    }

    @PostMapping("/login")
    public ResponseEntity<?> login(@Valid @RequestBody LoginRequest req,
                                   HttpServletRequest request,
                                   HttpServletResponse response) {
        // IP 维度限流补位用户名锁定的盲区（换用户名的密码枚举）
        if (!limiter.tryAcquire("login:"
                + cn.ouc.luminatrail.unikt.common.Ips.clientIp(request), 20, 60_000)) {
            return ResponseEntity.status(429)
                    .body(Map.of("message", "登录尝试太频繁，稍后再试"));
        }
        long now = System.currentTimeMillis();
        FailState fail = loginFails.get(req.username());
        if (fail != null && now < fail.lockUntil()) {
            long minutes = (fail.lockUntil() - now) / 60_000 + 1;
            return ResponseEntity.status(429)
                    .body(Map.of("message", "失败次数过多，锁定中（约 " + minutes + " 分钟）"));
        }
        User user = users.findByUsername(req.username()).orElse(null);
        if (user == null || !user.isActive()) {
            recordFailure(req.username(), now, fail);
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .body(Map.of("message", "用户名或密码不正确"));
        }
        try {
            loginSession(req.username(), req.password(), request, response);
        } catch (AuthenticationException e) {
            recordFailure(req.username(), now, fail);
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .body(Map.of("message", "用户名或密码不正确"));
        }
        loginFails.remove(req.username());
        return ResponseEntity.ok(new MeResponse(user.getUsername(), user.getRole()));
    }

    private static void recordFailure(String username, long now, FailState prev) {
        // 原子 merge 防并发绕过；距上次失败超过 LOCK_MS 时计数衰减，
        // 避免"第 11 分钟错一次就再锁"的永久锁死
        loginFails.merge(username, new FailState(1, 0, now), (old, v) -> {
            int count = now - old.lastFailAt() > LOCK_MS ? 1 : old.count() + 1;
            return new FailState(count, count >= MAX_FAILS ? now + LOCK_MS : 0, now);
        });
        if (loginFails.size() > 5000) {
            // 上限防御：丢弃已过锁定期且久未活动的条目
            loginFails.entrySet().removeIf(e -> e.getValue().lockUntil() < now - LOCK_MS
                    && e.getValue().lastFailAt() < now - LOCK_MS);
        }
    }

    @PostMapping("/logout")
    public Map<String, String> logout(HttpServletRequest request) {
        request.getSession().invalidate();
        SecurityContextHolder.clearContext();
        return Map.of("message", "已退出");
    }

    @GetMapping("/me")
    public ResponseEntity<MeResponse> me() {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        if (auth == null || !auth.isAuthenticated()
                || "anonymousUser".equals(auth.getPrincipal())) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(null);
        }
        return ResponseEntity.ok(new MeResponse(auth.getName(), roleOf(auth)));
    }

    static String roleOf(Authentication auth) {
        return auth.getAuthorities().stream()
                .map(a -> a.getAuthority())
                .filter(r -> r.startsWith("ROLE_"))
                .map(r -> r.substring(5))
                .findFirst().orElse(User.ROLE_USER);
    }

    /** 校验凭据并把认证写入会话（先轮换会话 ID，防会话固定攻击）。 */
    private void loginSession(String username, String password,
                              HttpServletRequest request, HttpServletResponse response) {
        Authentication auth = authManager.authenticate(
                new UsernamePasswordAuthenticationToken(username, password));
        // Spring Security 的 changeSessionId 只在 filter 阶段认证时触发，
        // 手动认证流程必须自己轮换，否则攻击者可预置 JSESSIONID 劫持会话
        request.getSession();
        try {
            request.changeSessionId();
        } catch (IllegalStateException ignored) {
            // 会话失效的极端时序下放弃轮换（认证仍写入新会话）
        }
        SecurityContext context = SecurityContextHolder.createEmptyContext();
        context.setAuthentication(auth);
        SecurityContextHolder.setContext(context);
        contextRepo.saveContext(context, request, response);
    }
}
