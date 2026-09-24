package cn.ouc.luminatrail.unikt.apikey;

import jakarta.servlet.http.HttpServletRequest;

/**
 * 过滤器路径守卫：以与 servlet 映射一致的视角判断受保护路径。
 *
 * {@link HttpServletRequest#getRequestURI()} 返回未解码、未归一化的
 * 原始串，直接 startsWith 会漏掉 ``/api/x/../v1/...``、``%2e%2e``、
 * ``;`` 路径参数等变体（映射层折叠后命中受保护 handler，过滤器却
 * 跳过）。这里先剥 context path，再显式拒绝可疑形态。
 */
public final class PathGuard {

    private PathGuard() {
    }

    /** 解析出剥掉 context path 的路径；形态可疑时返回 null（调用方应 400）。 */
    public static String normalize(HttpServletRequest request) {
        String uri = request.getRequestURI();
        String ctx = request.getContextPath();
        if (ctx != null && !ctx.isEmpty() && uri.startsWith(ctx)) {
            uri = uri.substring(ctx.length());
        }
        if (!uri.startsWith("/")) {
            return null;
        }
        String lower = uri.toLowerCase();
        if (lower.contains("/../") || lower.contains("/./")
                || uri.contains(";") || lower.contains("%2e")
                || lower.contains("%2f") || lower.contains("%5c")
                || uri.contains("\\")) {
            return null;
        }
        return uri;
    }
}
