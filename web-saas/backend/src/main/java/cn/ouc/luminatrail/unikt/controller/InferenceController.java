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
import org.springframework.security.core.context.SecurityContextHolder;
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
        var auth = SecurityContextHolder.getContext().getAuthentication();
        boolean admin = auth != null && auth.getAuthorities().stream()
                .anyMatch(a -> "ROLE_ADMIN".equals(a.getAuthority()));
        Long viewerId = auth != null
                && auth.getPrincipal() instanceof cn.ouc.luminatrail.unikt.user.PortalPrincipal p
                ? p.getId() : null;
        return routing.aggregateModels(viewerId, admin);
    }

    @GetMapping("/skills/{model}")
    public ResponseEntity<?> skills(@PathVariable String model) {
        // 限定名（@generic）的技能目录是用户上传数据的派生物，与 predict
        // 同样的归属校验（此前漏了这处，匿名可读他人目录）
        var denied = routing.checkModelAccess(model, currentViewerId(), isAdmin());
        if (denied != null) {
            return ResponseEntity.status(404).body(denied);
        }
        List<cn.ouc.luminatrail.unikt.dto.SkillDto> catalog = routing.skillsFor(model);
        if (catalog == null) {
            return ResponseEntity.status(404)
                    .body(java.util.Map.of("message", "模型未训练或无目录"));
        }
        return ResponseEntity.ok(catalog);
    }

    private static boolean isAdmin() {
        var auth = SecurityContextHolder.getContext().getAuthentication();
        return auth != null && auth.getAuthorities().stream()
                .anyMatch(a -> "ROLE_ADMIN".equals(a.getAuthority()));
    }

    private static Long currentViewerId() {
        var auth = org.springframework.security.core.context.SecurityContextHolder
                .getContext().getAuthentication();
        return auth != null
                && auth.getPrincipal() instanceof cn.ouc.luminatrail.unikt.user.PortalPrincipal p
                ? p.getId() : null;
    }

    /**
     * 演练场预测：公开接口（评委开箱即玩），但按 IP 限流——
     * 否则绕过 /api/v1 的 API Key 配额直接打满 GPU。
     */
    @PostMapping("/predict")
    public ResponseEntity<?> predict(
            @Valid @RequestBody PredictRequest request,
            jakarta.servlet.http.HttpServletRequest http) {
        String ip = cn.ouc.luminatrail.unikt.common.Ips.clientIp(http);
        if (!limiter.tryAcquire("predict:" + ip, 20, 60_000)) {
            return ResponseEntity.status(429)
                    .body(java.util.Map.of("status", 429,
                            "message", "预测请求太频繁（每分钟 20 次），稍后再试；"
                                    + "批量调用请用 API 密钥走 /api/v1/predict"));
        }
        // 与 skills/models 同口径：admin 可预测任意归属的模型（管理排障场景）
        var denied = routing.checkModelAccess(request.model(), currentViewerId(), isAdmin());
        if (denied != null) {
            return ResponseEntity.status(404).body(denied);
        }
        return ResponseEntity.ok(routing.route(request));
    }
}
