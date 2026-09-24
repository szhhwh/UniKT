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

    public AuthController(UserRepository users, PasswordEncoder encoder,
                          AuthenticationManager authManager,
                          SecurityContextRepository contextRepo) {
        this.users = users;
        this.encoder = encoder;
        this.authManager = authManager;
        this.contextRepo = contextRepo;
    }

    @PostMapping("/register")
    public ResponseEntity<?> register(@Valid @RequestBody RegisterRequest req,
                                      HttpServletRequest request,
                                      HttpServletResponse response) {
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
        return ResponseEntity.status(HttpStatus.CREATED)
                .body(new MeResponse(user.getUsername(), user.getRole()));
    }

    @PostMapping("/login")
    public ResponseEntity<MeResponse> login(@Valid @RequestBody LoginRequest req,
                                            HttpServletRequest request,
                                            HttpServletResponse response) {
        User user = users.findByUsername(req.username()).orElse(null);
        if (user == null || !user.isActive()) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .body(null);
        }
        try {
            loginSession(req.username(), req.password(), request, response);
        } catch (AuthenticationException e) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(null);
        }
        return ResponseEntity.ok(new MeResponse(user.getUsername(), user.getRole()));
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

    /** 校验凭据并把认证写入会话。 */
    private void loginSession(String username, String password,
                              HttpServletRequest request, HttpServletResponse response) {
        Authentication auth = authManager.authenticate(
                new UsernamePasswordAuthenticationToken(username, password));
        SecurityContext context = SecurityContextHolder.createEmptyContext();
        context.setAuthentication(auth);
        SecurityContextHolder.setContext(context);
        contextRepo.saveContext(context, request, response);
    }
}
