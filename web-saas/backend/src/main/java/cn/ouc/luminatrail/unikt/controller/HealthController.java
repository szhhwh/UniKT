package cn.ouc.luminatrail.unikt.controller;

import cn.ouc.luminatrail.unikt.config.WebConfig;
import cn.ouc.luminatrail.unikt.dto.ModelInfo;
import cn.ouc.luminatrail.unikt.service.InferenceService;
import java.util.List;
import java.util.Map;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api")
public class HealthController {

    private final InferenceService inference;
    private final WebConfig webConfig;

    public HealthController(InferenceService inference, WebConfig webConfig) {
        this.inference = inference;
        this.webConfig = webConfig;
    }

    @GetMapping("/health")
    public Map<String, Object> health() {
        boolean inferenceUp = inference.ping();
        List<ModelInfo> models = inferenceUp ? inference.listModels() : List.of();
        return Map.of(
                "status", "ok",
                "service", "unikt-saas-backend",
                "inferenceUp", inferenceUp,
                "modelCount", models.size(),
                "docsAvailable", webConfig.docsAvailable());
    }
}
