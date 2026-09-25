package cn.ouc.luminatrail.unikt.user;

import jakarta.servlet.http.HttpServletResponse;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.MediaType;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.config.annotation.authentication.configuration.AuthenticationConfiguration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.context.HttpSessionSecurityContextRepository;
import org.springframework.security.web.context.SecurityContextRepository;

/**
 * 安全配置：会话 Cookie 认证 + 按角色划分访问面。
 *
 * 公开（评委打开即玩）：首页/文档/模型/演练场/开放 API（API Key 层）。
 * 登录用户：密钥管理（只见自己的）。管理员：节点、用户管理。
 *
 * CSRF 由 SameSite=Strict 会话 Cookie 覆盖（前后端同源）；关闭 csrf
 * token 是有意为之。
 */
@Configuration
@EnableWebSecurity
public class SecurityConfig {

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http.csrf(csrf -> csrf.disable());
        http.sessionManagement(s -> s.sessionCreationPolicy(SessionCreationPolicy.IF_REQUIRED));
        http.authorizeHttpRequests(auth -> auth
                .requestMatchers("/api/auth/**").permitAll()
                .requestMatchers("/api/health", "/api/models", "/api/predict",
                        "/api/skills/**", "/api/exp/health").permitAll()
                .requestMatchers("/api/v1/**").permitAll() // 由 ApiKeyFilter 鉴权
                .requestMatchers("/api/nodes/**").hasRole("ADMIN")
                .requestMatchers("/api/admin/**").hasRole("ADMIN")
                .requestMatchers("/api/keys/**", "/api/datasets/**",
                        "/api/training-jobs/**").authenticated()
                .anyRequest().permitAll()); // 静态资源/SPA/文档等
        http.exceptionHandling(eh -> eh
                .authenticationEntryPoint((req, res, ex) ->
                        writeJson(res, 401, "未登录或会话已过期"))
                .accessDeniedHandler((req, res, ex) ->
                        writeJson(res, 403, "没有权限执行此操作")));
        return http.build();
    }

    /** API Key 过滤器在 Spring Security 排序里注册为全局组件即可生效。 */
    @Bean
    public AuthenticationManager authenticationManager(
            AuthenticationConfiguration config) throws Exception {
        return config.getAuthenticationManager();
    }

    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }

    @Bean
    public SecurityContextRepository securityContextRepository() {
        return new HttpSessionSecurityContextRepository();
    }

    private static void writeJson(HttpServletResponse res, int status, String message)
            throws java.io.IOException {
        res.setStatus(status);
        res.setContentType(MediaType.APPLICATION_JSON_VALUE);
        res.setCharacterEncoding("UTF-8");
        res.getWriter().write("{\"status\":" + status + ",\"message\":\"" + message + "\"}");
    }
}
