package cn.ouc.luminatrail.unikt.apikey;

import cn.ouc.luminatrail.unikt.config.InferenceProperties;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

/**
 * 管理令牌过滤器：保护门户的管理接口（节点 /api/nodes、密钥 /api/keys，
 * 含只读查询——节点信息含内网拓扑，属管理面数据）。
 *
 * 配置了 ``unikt.admin-token``（或环境变量 UNIKT_ADMINTOKEN）时，请求
 * 必须携带一致的 ``X-Admin-Token`` 头（恒定时间比较）；未配置时放行并
 * 在启动日志 ERROR——内网演示可接受，对外部署必须配置。
 *
 * 推理开放接口（/api/v1，走 API Key）不在保护范围。
 */
@Component
public class AdminTokenFilter extends OncePerRequestFilter {

    private static final Logger log = LoggerFactory.getLogger(AdminTokenFilter.class);

    private final InferenceProperties props;

    public AdminTokenFilter(InferenceProperties props) {
        this.props = props;
    }

    @Override
    protected boolean shouldNotFilter(HttpServletRequest request) {
        if ("OPTIONS".equals(request.getMethod())) {
            return true;
        }
        // 覆盖整个 /api 命名空间：可疑形态也要进 doFilterInternal 拒绝
        String raw = request.getRequestURI();
        String ctx = request.getContextPath();
        if (ctx != null && !ctx.isEmpty()) {
            return !raw.startsWith(ctx + "/api");
        }
        return !raw.startsWith("/api");
    }

    @Override
    protected void doFilterInternal(
            HttpServletRequest request, HttpServletResponse response, FilterChain chain)
            throws ServletException, IOException {
        String path = PathGuard.normalize(request);
        if (path == null) {
            response.setStatus(HttpServletResponse.SC_BAD_REQUEST);
            writeJson(response, 400, "请求路径形态非法");
            return;
        }
        boolean managed = path.startsWith("/api/nodes") || path.startsWith("/api/keys");
        if (!managed) {
            chain.doFilter(request, response);
            return;
        }
        String token = props.adminToken();
        if (token == null || token.isBlank()) {
            // 未配置：放行（启动时已 ERROR），仅限内网演示场景
            chain.doFilter(request, response);
            return;
        }
        String provided = request.getHeader("X-Admin-Token");
        boolean ok = provided != null
                && MessageDigest.isEqual(
                        token.getBytes(StandardCharsets.UTF_8),
                        provided.getBytes(StandardCharsets.UTF_8));
        if (ok) {
            chain.doFilter(request, response);
            return;
        }
        response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
        writeJson(response, 401, "需要管理令牌（X-Admin-Token）");
    }

    private static void writeJson(HttpServletResponse response, int status, String message)
            throws IOException {
        response.setContentType(MediaType.APPLICATION_JSON_VALUE);
        response.setCharacterEncoding("UTF-8");
        response.getWriter().write("{\"status\":" + status + ",\"message\":\"" + message + "\"}");
    }

    @Override
    protected void initFilterBean() {
        if (props.adminToken() == null || props.adminToken().isBlank()) {
            log.error("unikt.admin-token 未配置：节点/密钥管理接口处于无保护状态"
                    + "（内网演示可接受，对外部署必须设置 UNIKT_ADMINTOKEN）");
        }
    }
}
