package cn.ouc.luminatrail.unikt.apikey;

import cn.ouc.luminatrail.unikt.config.InferenceProperties;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import java.io.IOException;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

/**
 * 开放 API（/api/v1/**）的 API Key 鉴权过滤器：验证密钥 + 每日配额。
 *
 * 请求头 X-API-Key；无效/缺失 401，超出当日配额 429。CORS 预检放行。
 */
@Component
public class ApiKeyFilter extends OncePerRequestFilter {

    private final ApiKeyService apiKeys;
    private final InferenceProperties props;

    public ApiKeyFilter(ApiKeyService apiKeys, InferenceProperties props) {
        this.apiKeys = apiKeys;
        this.props = props;
    }

    @Override
    protected boolean shouldNotFilter(HttpServletRequest request) {
        if ("OPTIONS".equals(request.getMethod())) {
            return true;
        }
        // 解码后视角判定 /api 命名空间（原始 URI 的 //api、/api/%76%31
        // 等变形由 normalize 拒绝为 null → 进 doFilterInternal 400，
        // 绝不放行）。normalize==null 的任何请求都进过滤器拒绝。
        String path = PathGuard.normalize(request);
        if (path == null) {
            return false;
        }
        return !path.startsWith("/api");
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
        if (!apiKeys.tryConsumeQuota(user, props.quotaDaily())) {
            reject(response, 429, "今日调用配额已用完（" + props.quotaDaily() + " 次/天），明天再来");
            return;
        }
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
