package cn.ouc.luminatrail.unikt.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

/** 服务连接与路径配置（application.yml: unikt.*）。 */
@ConfigurationProperties(prefix = "unikt")
public record InferenceProperties(
        String inferenceBaseUrl,
        int inferenceTimeoutMs,
        String docsDir,
        String expBaseUrl,
        String inferenceSrc,
        long deployTimeoutMs,
        long quotaDaily,
        String inferenceToken,
        String frontendDir) {
}
