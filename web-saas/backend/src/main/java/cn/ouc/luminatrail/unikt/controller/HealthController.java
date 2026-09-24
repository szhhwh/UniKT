package cn.ouc.luminatrail.unikt.controller;

import cn.ouc.luminatrail.unikt.config.WebConfig;
import cn.ouc.luminatrail.unikt.dto.ModelInfo;
import cn.ouc.luminatrail.unikt.node.ComputeNode;
import cn.ouc.luminatrail.unikt.node.NodeRepository;
import cn.ouc.luminatrail.unikt.service.InferenceService;
import java.util.HashMap;
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
    private final NodeRepository nodes;

    public HealthController(InferenceService inference, WebConfig webConfig, NodeRepository nodes) {
        this.inference = inference;
        this.webConfig = webConfig;
        this.nodes = nodes;
    }

    @GetMapping("/health")
    public Map<String, Object> health() {
        boolean inferenceUp = inference.ping();
        List<ModelInfo> models = inferenceUp ? inference.listModels() : List.of();
        List<ComputeNode> all = nodes.findAll();
        long online = all.stream().filter(n -> ComputeNode.ONLINE.equals(n.getStatus())).count();

        Map<String, Object> out = new HashMap<>();
        out.put("status", "ok");
        out.put("service", "unikt-saas-backend");
        out.put("inferenceUp", inferenceUp);
        out.put("modelCount", models.size());
        out.put("docsAvailable", webConfig.docsAvailable());
        out.put("nodesOnline", online);
        out.put("nodesTotal", all.size());
        return out;
    }
}
