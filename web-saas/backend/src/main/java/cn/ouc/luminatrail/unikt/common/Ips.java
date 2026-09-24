package cn.ouc.luminatrail.unikt.common;

import jakarta.servlet.http.HttpServletRequest;

/**
 * 客户端 IP 提取（限流/注册配额共用）。
 *
 * 默认只信 socket 地址；仅当直连方是本机回环（即我们自己的 Caddy/
 * Nginx 反代同机部署）时才采信 X-Forwarded-For 的**最右**值（最近一跳
 * 由受信反代写入，客户端伪造的最左值不可信）。
 */
public final class Ips {

    private Ips() {
    }

    public static String clientIp(HttpServletRequest request) {
        String remote = request.getRemoteAddr();
        if (remote != null && (remote.equals("127.0.0.1") || remote.equals("0:0:0:0:0:0:0:1")
                || remote.equals("::1"))) {
            String xff = request.getHeader("X-Forwarded-For");
            if (xff != null && !xff.isBlank()) {
                String[] parts = xff.split(",");
                for (int i = parts.length - 1; i >= 0; i--) {
                    String v = parts[i].trim();
                    if (!v.isEmpty()) {
                        return v;
                    }
                }
            }
        }
        return remote;
    }
}
