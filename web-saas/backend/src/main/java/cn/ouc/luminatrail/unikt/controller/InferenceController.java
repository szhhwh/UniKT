package cn.ouc.luminatrail.unikt.controller;

import cn.ouc.luminatrail.unikt.dto.ModelInfo;
import cn.ouc.luminatrail.unikt.dto.PredictRequest;
import cn.ouc.luminatrail.unikt.dto.PredictResponse;
import cn.ouc.luminatrail.unikt.service.InferenceService;
import jakarta.validation.Valid;
import java.util.List;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api")
public class InferenceController {

    private final InferenceService inference;

    public InferenceController(InferenceService inference) {
        this.inference = inference;
    }

    @GetMapping("/models")
    public List<ModelInfo> models() {
        return inference.listModels();
    }

    @PostMapping("/predict")
    public ResponseEntity<PredictResponse> predict(@Valid @RequestBody PredictRequest request) {
        return ResponseEntity.ok(inference.predict(request));
    }
}
