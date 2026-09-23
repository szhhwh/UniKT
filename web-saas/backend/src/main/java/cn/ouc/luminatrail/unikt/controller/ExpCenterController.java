package cn.ouc.luminatrail.unikt.controller;

import cn.ouc.luminatrail.unikt.config.InferenceProperties;
import java.time.Duration;
import java.util.Map;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;

/**
 * 实验中心（web/ 管理器）状态代理。
 *
 * 管理器是独立进程（Vite 5173 + FastAPI），门户不做反代（其 SPA 资源与
 * API 路径无法安全挂在 /exp 前缀下），这里只做可达性探测 + 跳转地址下发。
 */
@RestController
@RequestMapping("/api/exp")
public class ExpCenterController {

    private final InferenceProperties props;

    public ExpCenterController(InferenceProperties props) {
        this.props = props;
    }

    @GetMapping("/health")
    public Map<String, Object> health() {
        String base = props.expBaseUrl();
        if (base == null || base.isBlank()) {
            return Map.of("status", "not-configured", "reachable", false, "url", "");
        }
        boolean reachable = ping(base);
        return Map.of(
                "status", reachable ? "ok" : "unreachable",
                "reachable", reachable,
                "url", base);
    }

    private static boolean ping(String base) {
        try {
            SimpleClientHttpRequestFactory factory = new SimpleClientHttpRequestFactory();
            factory.setConnectTimeout(Duration.ofSeconds(2));
            factory.setReadTimeout(Duration.ofSeconds(2));
            RestClient.builder()
                    .baseUrl(base)
                    .requestFactory(factory)
                    .build()
                    .get()
                    .uri("/")
                    .retrieve()
                    .toBodilessEntity();
            return true;
        } catch (RestClientException e) {
            // 4xx 也算"进程在跑"（如根路径 404）；连接失败才算不可达
            return e instanceof HttpClientErrorException;
        }
    }
}
