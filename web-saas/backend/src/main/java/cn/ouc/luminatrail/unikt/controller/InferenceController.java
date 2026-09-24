package cn.ouc.luminatrail.unikt.controller;

import cn.ouc.luminatrail.unikt.dto.ModelInfo;
import cn.ouc.luminatrail.unikt.dto.PredictRequest;
import cn.ouc.luminatrail.unikt.dto.PredictResponse;
import cn.ouc.luminatrail.unikt.node.NodeRoutingService;
import cn.ouc.luminatrail.unikt.service.InferenceService;
import jakarta.validation.Valid;
import java.util.List;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api")
public class InferenceController {

    private final InferenceService inference;
    private final NodeRoutingService routing;
    private final cn.ouc.luminatrail.unikt.common.RateLimiter limiter;

    public InferenceController(InferenceService inference, NodeRoutingService routing,
                               cn.ouc.luminatrail.unikt.common.RateLimiter limiter) {
        this.inference = inference;
        this.routing = routing;
        this.limiter = limiter;
    }

    @GetMapping("/models")
    public List<ModelInfo> models() {
        return routing.aggregateModels();
    }

    @GetMapping("/skills/{model}")
    public ResponseEntity<List<cn.ouc.luminatrail.unikt.dto.SkillDto>> skills(
            @PathVariable String model) {
        List<cn.ouc.luminatrail.unikt.dto.SkillDto> catalog = routing.skillsFor(model);
        if (catalog == null) {
            return ResponseEntity.status(404).body(null);
        }
        return ResponseEntity.ok(catalog);
    }

    /**
     * 演练场预测：公开接口（评委开箱即玩），但按 IP 限流——
     * 否则绕过 /api/v1 的 API Key 配额直接打满 GPU。
     */
    @PostMapping("/predict")
    public ResponseEntity<?> predict(
            @Valid @RequestBody PredictRequest request,
            jakarta.servlet.http.HttpServletRequest http) {
        String ip = clientIp(http);
        if (!limiter.tryAcquire("predict:" + ip, 20, 60_000)) {
            return ResponseEntity.status(429)
                    .body(java.util.Map.of("status", 429,
                            "message", "预测请求太频繁（每分钟 20 次），稍后再试；"
                                    + "批量调用请用 API 密钥走 /api/v1/predict"));
        }
        return ResponseEntity.ok(routing.route(request));
    }

    private static String clientIp(jakarta.servlet.http.HttpServletRequest req) {
        // 反代场景取 X-Forwarded-For 首个（部署在 Caddy/Nginx 后时正确）
        String fwd = req.getHeader("X-Forwarded-For");
        if (fwd != null && !fwd.isBlank()) {
            return fwd.split(",")[0].trim();
        }
        return req.getRemoteAddr();
    }
}
