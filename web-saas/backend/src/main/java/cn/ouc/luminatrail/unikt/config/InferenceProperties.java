package cn.ouc.luminatrail.unikt.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

/** Python 推理服务连接配置（application.yml: unikt.*）。 */
@ConfigurationProperties(prefix = "unikt")
public record InferenceProperties(String inferenceBaseUrl, int inferenceTimeoutMs) {
}
