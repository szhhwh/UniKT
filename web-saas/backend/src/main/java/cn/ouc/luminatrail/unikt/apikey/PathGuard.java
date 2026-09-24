package cn.ouc.luminatrail.unikt.apikey;

import jakarta.servlet.http.HttpServletRequest;

/**
 * 过滤器路径守卫：以与 servlet 映射一致的视角判断受保护路径。
 *
 * 前缀判定基于 {@link HttpServletRequest#getServletPath()}（Boot 默认
 * dispatcher 映射 / 时即**解码+归一化后**的路径）——此前用原始
 * {@link HttpServletRequest#getRequestURI()} 做 startsWith，会被
 * {@code //api/v1/x}、{@code /api//v1/x}、{@code /api/%76%31/x} 等
 * 变形绕过（映射层归一化解码后命中 handler，过滤器却跳过/放行）。
 *
 * 原始 URI 仍做显式拒绝（纵深防御）：{@code /../}、{@code ;}、
 * 编码点斜杠等可疑形态一律 400。
 */
public final class PathGuard {

    private PathGuard() {
    }

    /** 解析出解码后的 servlet 路径；原始 URI 形态可疑时返回 null（调用方应 400）。 */
    public static String normalize(HttpServletRequest request) {
        String raw = request.getRequestURI();
        String ctx = request.getContextPath();
        String rawNoCtx = raw;
        if (ctx != null && !ctx.isEmpty() && raw.startsWith(ctx)) {
            rawNoCtx = raw.substring(ctx.length());
        }
        if (!rawNoCtx.startsWith("/")) {
            return null;
        }
        String lower = rawNoCtx.toLowerCase();
        if (lower.contains("/../") || lower.contains("/./")
                || rawNoCtx.contains(";") || lower.contains("%2e")
                || lower.contains("%2f") || lower.contains("%5c")
                || rawNoCtx.contains("\\")) {
            return null;
        }
        // 解码+归一化视角：getServletPath 在默认 / 映射下返回去上下文前缀
        // 的解码路径（已折叠 // 与 ..）；与 getRequestURI 不一致时以它为准
        String servletPath = request.getServletPath();
        if (servletPath != null && !servletPath.isBlank()) {
            return servletPath;
        }
        return rawNoCtx;
    }
}
