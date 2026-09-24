package cn.ouc.luminatrail.unikt.apikey;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import java.io.IOException;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

/**
 * 开放 API（/api/v1/**）的 API Key 鉴权过滤器。
 *
 * 请求头 X-API-Key；无效/缺失返回 401 JSON（与门户统一错误体一致）。
 * CORS 预检（OPTIONS）放行。门户自身路由（/api/models 等）不走此
 * 过滤器——它们由浏览器会话使用，多用户化后统一收口到 Phase 3。
 */
@Component
public class ApiKeyFilter extends OncePerRequestFilter {

    private final ApiKeyService apiKeys;

    public ApiKeyFilter(ApiKeyService apiKeys) {
        this.apiKeys = apiKeys;
    }

    @Override
    protected boolean shouldNotFilter(HttpServletRequest request) {
        if ("OPTIONS".equals(request.getMethod())) {
            return true;
        }
        // 覆盖整个 /api 命名空间：可疑形态（normalize 为 null）也要进
        // doFilterInternal 拒绝，不能在 shouldNotFilter 里放行
        return !underApi(request);
    }

    private static boolean underApi(HttpServletRequest request) {
        String raw = request.getRequestURI();
        String ctx = request.getContextPath();
        if (ctx != null && !ctx.isEmpty()) {
            return raw.startsWith(ctx + "/api");
        }
        return raw.startsWith("/api");
    }

    @Override
    protected void doFilterInternal(
            HttpServletRequest request, HttpServletResponse response, FilterChain chain)
            throws ServletException, IOException {
        String path = PathGuard.normalize(request);
        if (path == null) {
            reject(response, HttpServletResponse.SC_BAD_REQUEST, "请求路径形态非法");
            return;
        }
        if (!path.startsWith("/api/v1/")) {
            chain.doFilter(request, response);
            return;
        }
        ApiKeyUser user = apiKeys.verify(request.getHeader("X-API-Key"));
        if (user == null) {
            reject(response, HttpServletResponse.SC_UNAUTHORIZED,
                    "API Key 缺失或无效（请求头 X-API-Key，密钥在门户「API」页创建）");
            return;
        }
        // 供后续按调用方限流/审计使用（当前无消费者）
        request.setAttribute("apiKeyUser", user);
        apiKeys.touch(user.getId());
        chain.doFilter(request, response);
    }

    private static void reject(HttpServletResponse response, int status, String message)
            throws IOException {
        response.setStatus(status);
        response.setContentType(MediaType.APPLICATION_JSON_VALUE);
        response.setCharacterEncoding("UTF-8");
        response.getWriter().write("{\"status\":" + status + ",\"message\":\"" + message + "\"}");
    }
}
