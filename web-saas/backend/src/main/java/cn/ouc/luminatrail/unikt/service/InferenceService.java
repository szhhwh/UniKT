package cn.ouc.luminatrail.unikt.service;

import cn.ouc.luminatrail.unikt.config.InferenceProperties;
import cn.ouc.luminatrail.unikt.dto.ModelInfo;
import cn.ouc.luminatrail.unikt.dto.PredictRequest;
import cn.ouc.luminatrail.unikt.dto.PredictResponse;
import java.time.Duration;
import java.util.List;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.stereotype.Service;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;

/**
 * 对 Python 推理服务（web-saas/inference）的转发层。
 *
 * SpringBoot 只做业务编排（鉴权/会话/限流等），张量计算一律下沉到 Python 侧，
 * 避免在 Java 里重写 60+ 个 KT 模型。
 */
@Service
public class InferenceService {

    private final RestClient client;
    private final Duration timeout;

    public InferenceService(InferenceProperties props) {
        this.timeout = Duration.ofMillis(props.inferenceTimeoutMs());
        this.client = build(props.inferenceBaseUrl(), timeout);
    }

    private static RestClient build(String baseUrl, Duration timeout) {
        SimpleClientHttpRequestFactory factory = new SimpleClientHttpRequestFactory();
        factory.setConnectTimeout(timeout);
        factory.setReadTimeout(timeout);
        return RestClient.builder().baseUrl(baseUrl).requestFactory(factory).build();
    }

    /** 拉取可用模型清单（默认推理服务）；不可达时返回空列表。 */
    public List<ModelInfo> listModels() {
        try {
            return client.get()
                    .uri("/models")
                    .retrieve()
                    .body(new ParameterizedTypeReference<List<ModelInfo>>() {
                    });
        } catch (RestClientException e) {
            return List.of();
        }
    }

    /** 面向指定节点地址的模型清单（节点路由用）。 */
    public List<ModelInfo> listModelsFrom(String baseUrl) {
        try {
            return build(baseUrl, timeout).get()
                    .uri("/models")
                    .retrieve()
                    .body(new ParameterizedTypeReference<List<ModelInfo>>() {
                    });
        } catch (RestClientException e) {
            return List.of();
        }
    }

    public PredictResponse predict(PredictRequest request) {
        return predictAt(null, request);
    }

    /** 在指定节点上推理；baseUrl 为 null 时用默认推理服务。 */
    public PredictResponse predictAt(String baseUrl, PredictRequest request) {
        RestClient c = baseUrl == null ? client : build(baseUrl, timeout);
        try {
            return c.post()
                    .uri("/predict")
                    .body(request)
                    .retrieve()
                    .body(PredictResponse.class);
        } catch (HttpClientErrorException e) {
            // 上游 4xx（如参数校验失败）原样透传，不吞成 5xx
            throw e;
        } catch (ResourceAccessException e) {
            throw new InferenceUnavailableException("推理服务不可达，请先启动 web-saas/inference", e);
        } catch (RestClientException e) {
            throw new InferenceUnavailableException("推理服务返回错误：" + e.getMessage(), e);
        }
    }

    /** 推理服务连通性探测。 */
    public boolean ping() {
        try {
            client.get().uri("/health").retrieve().toBodilessEntity();
            return true;
        } catch (RestClientException e) {
            return false;
        }
    }

    /** 推理服务不可用（未启动/超时），对应 HTTP 503。 */
    public static class InferenceUnavailableException extends RuntimeException {

        public InferenceUnavailableException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
