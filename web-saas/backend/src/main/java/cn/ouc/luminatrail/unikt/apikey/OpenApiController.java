package cn.ouc.luminatrail.unikt.apikey;

import cn.ouc.luminatrail.unikt.dto.ModelInfo;
import cn.ouc.luminatrail.unikt.dto.PredictRequest;
import cn.ouc.luminatrail.unikt.dto.PredictResponse;
import cn.ouc.luminatrail.unikt.node.NodeRoutingService;
import jakarta.validation.Valid;
import java.util.List;
import org.springframework.web.bind.annotation.GetMapping;
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

    public OpenApiController(NodeRoutingService routing) {
        this.routing = routing;
    }

    @GetMapping("/models")
    public List<ModelInfo> models() {
        return routing.aggregateModels();
    }

    @PostMapping("/predict")
    public PredictResponse predict(@Valid @RequestBody PredictRequest request) {
        return routing.route(request);
    }
}
