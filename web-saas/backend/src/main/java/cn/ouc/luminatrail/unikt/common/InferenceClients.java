package cn.ouc.luminatrail.unikt.common;

import cn.ouc.luminatrail.unikt.config.InferenceProperties;
import java.time.Duration;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;

/**
 * 推理服务 HTTP 客户端工厂：统一注入共享密钥头（unikt.inference-token
 * 配置时）。所有对推理服务（默认或节点）的调用都必须经此创建客户端，
 * 否则会被推理服务的 token 校验拒绝。
 */
@Component
public class InferenceClients {

    private final InferenceProperties props;

    public InferenceClients(InferenceProperties props) {
        this.props = props;
    }

    public RestClient client(String baseUrl, int timeoutMs) {
        SimpleClientHttpRequestFactory f = new SimpleClientHttpRequestFactory();
        f.setConnectTimeout(Duration.ofMillis(timeoutMs));
        f.setReadTimeout(Duration.ofMillis(timeoutMs));
        RestClient.Builder b = RestClient.builder()
                .baseUrl(baseUrl)
                .requestFactory(f);
        String token = props.inferenceToken();
        if (token != null && !token.isBlank()) {
            b.defaultHeader("X-Inference-Token", token);
        }
        return b.build();
    }
}
