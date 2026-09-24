package cn.ouc.luminatrail.unikt.user;

import cn.ouc.luminatrail.unikt.apikey.PathGuard;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import java.io.IOException;
import org.springframework.http.MediaType;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

/**
 * 会话账号实时校验：登录后的每个 /api 请求都会按用户名回查一次账号
 * 状态——被禁用/删除、或角色与登录时不一致（被降权/提权）立即失效
 * 会话并要求重新登录。否则管理员"禁用"要等对方会话自然过期才生效。
 *
 * 公开路径与开放 API 不经过本检查（无认证上下文）。
 */
@Component
public class SessionAccountFilter extends OncePerRequestFilter {

    private final UserRepository users;

    public SessionAccountFilter(UserRepository users) {
        this.users = users;
    }

    @Override
    protected boolean shouldNotFilter(HttpServletRequest request) {
        String path = PathGuard.normalize(request);
        if (path == null) {
            return true; // 可疑路径交给安全链拒绝
        }
        return !(path.startsWith("/api/keys") || path.startsWith("/api/nodes")
                || path.startsWith("/api/admin") || path.equals("/api/auth/me")
                || path.startsWith("/api/datasets") || path.startsWith("/api/training-jobs"));
    }

    @Override
    protected void doFilterInternal(
            HttpServletRequest request, HttpServletResponse response, FilterChain chain)
            throws ServletException, IOException {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        if (auth != null && auth.isAuthenticated()
                && auth.getPrincipal() instanceof PortalPrincipal principal) {
            User current = users.findByUsername(principal.getUsername()).orElse(null);
            boolean invalid = current == null
                    || !current.isActive()
                    || !current.getRole().equals(principal.getRole());
            if (invalid) {
                request.getSession().invalidate();
                SecurityContextHolder.clearContext();
                response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
                response.setContentType(MediaType.APPLICATION_JSON_VALUE);
                response.setCharacterEncoding("UTF-8");
                response.getWriter()
                        .write("{\"status\":401,\"message\":\"账号状态已变更，请重新登录\"}");
                return;
            }
        }
        chain.doFilter(request, response);
    }
}
