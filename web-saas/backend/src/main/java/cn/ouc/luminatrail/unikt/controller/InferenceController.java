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

    public InferenceController(InferenceService inference, NodeRoutingService routing) {
        this.inference = inference;
        this.routing = routing;
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

    @PostMapping("/predict")
    public ResponseEntity<PredictResponse> predict(@Valid @RequestBody PredictRequest request) {
        return ResponseEntity.ok(routing.route(request));
    }
}
