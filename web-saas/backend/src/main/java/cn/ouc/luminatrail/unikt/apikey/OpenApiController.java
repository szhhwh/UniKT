package cn.ouc.luminatrail.unikt.apikey;

import cn.ouc.luminatrail.unikt.dto.ModelInfo;
import cn.ouc.luminatrail.unikt.dto.PredictRequest;
import cn.ouc.luminatrail.unikt.dto.PredictResponse;
import cn.ouc.luminatrail.unikt.node.NodeRoutingService;
import jakarta.validation.Valid;
import java.util.List;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * 开放 API（需 X-API-Key，见 {@link ApiKeyFilter}）：给用户自己的应用调用。
 *
 * 与门户 /api/* 同一套推理路由，但对第三方暴露稳定契约（v1 版本化）。
 */
@RestController
@RequestMapping("/api/v1")
public class OpenApiController {

    private final NodeRoutingService routing;
    private final cn.ouc.luminatrail.unikt.common.RateLimiter limiter;

    public OpenApiController(NodeRoutingService routing,
                             cn.ouc.luminatrail.unikt.common.RateLimiter limiter) {
        this.routing = routing;
        this.limiter = limiter;
    }

    @GetMapping("/models")
    public List<ModelInfo> models(jakarta.servlet.http.HttpServletRequest http) {
        Object u = http.getAttribute("apiKeyUser");
        Long ownerId = u instanceof cn.ouc.luminatrail.unikt.apikey.ApiKeyUser au
                ? au.getOwnerId() : null;
        return routing.aggregateModels(ownerId, false);
    }

    @GetMapping("/skills/{model}")
    public ResponseEntity<List<cn.ouc.luminatrail.unikt.dto.SkillDto>> skills(
            @PathVariable String model) {
        List<cn.ouc.luminatrail.unikt.dto.SkillDto> catalog = routing.skillsFor(model);
        if (catalog == null) {
            return ResponseEntity.notFound().build();
        }
        return ResponseEntity.ok(catalog);
    }

    @PostMapping("/predict")
    public PredictResponse predict(@Valid @RequestBody PredictRequest request,
                                   jakarta.servlet.http.HttpServletRequest http) {
        // 日配额只限总量；这里加每密钥分钟级限速防瞬时打满 GPU
        Object u = http.getAttribute("apiKeyUser");
        long keyId = u instanceof cn.ouc.luminatrail.unikt.apikey.ApiKeyUser au ? au.getId() : -1;
        if (!limiter.tryAcquire("v1:" + keyId, 60, 60_000)) {
            throw new cn.ouc.luminatrail.unikt.apikey.RateLimitedException(
                    "该密钥每分钟最多 60 次调用");
        }
        Long ownerId = u instanceof cn.ouc.luminatrail.unikt.apikey.ApiKeyUser au2
                ? au2.getOwnerId() : null;
        var denied = routing.checkModelAccess(request.model(), ownerId, false);
        if (denied != null) {
            throw new cn.ouc.luminatrail.unikt.apikey.ModelAccessDeniedException(
                    (String) denied.get("message"));
        }
        return routing.route(request);
    }
}
