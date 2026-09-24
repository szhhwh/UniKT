package cn.ouc.luminatrail.unikt.node;

import cn.ouc.luminatrail.unikt.dto.ModelInfo;
import cn.ouc.luminatrail.unikt.dto.PredictRequest;
import cn.ouc.luminatrail.unikt.dto.PredictResponse;
import cn.ouc.luminatrail.unikt.service.InferenceService;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.data.domain.Sort;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;

/**
 * 推理路由：聚合「默认推理服务 + 全部在线节点」的模型视图，预测请求
 * 路由到第一个持有该可用模型的执行体（默认服务优先，其次节点注册序）。
 *
 * 模型清单按执行体做 5 秒 TTL 缓存并用短超时（2.5s）拉取，避免每次
 * 预测都串行等全部节点、以及掉线节点给请求加 15s 延迟。节点暂时不可
 * 达时自动顺延到下一个候选，全部失败才抛给上层。
 */
@Service
public class NodeRoutingService {

    private static final long CACHE_TTL_MS = 10000;
    private static final int NODE_TIMEOUT_MS = 8000;

    /** 一个推理执行体（默认服务或某节点）。 */
    private record Endpoint(String name, String baseUrl) {
    }

    /** 执行体级模型缓存。 */
    private record ModelsCache(long at, List<ModelInfo> models) {
    }

    private final NodeRepository nodes;
    private final InferenceService inference;
    private final Map<String, ModelsCache> cache = new ConcurrentHashMap<>();

    public NodeRoutingService(NodeRepository nodes, InferenceService inference) {
        this.nodes = nodes;
        this.inference = inference;
    }

    /** 聚合模型清单：同名模型合并 available（OR）与首个提供者。 */
    public List<ModelInfo> aggregateModels() {
        Map<String, ModelInfo> merged = new HashMap<>();
        for (Endpoint ep : endpoints()) {
            for (ModelInfo m : modelsOf(ep)) {
                // 上游不感知节点名，这里按执行体打标（默认服务保持 null）
                ModelInfo tagged = m.node() == null
                        ? new ModelInfo(m.name(), m.available(), m.numSkills(), ep.name())
                        : m;
                merged.merge(tagged.name(), tagged, (a, b) -> new ModelInfo(
                        a.name(), a.available() || b.available(),
                        a.numSkills() != null ? a.numSkills() : b.numSkills(),
                        a.available() ? a.node() : b.node()));
            }
        }
        return new ArrayList<>(merged.values());
    }

    /** 路由一次预测：按候选顺序找第一个「有该可用模型」的执行体。 */
    public PredictResponse route(PredictRequest request) {
        for (Endpoint ep : endpoints()) {
            List<ModelInfo> models = modelsOf(ep);
            if (models.stream().anyMatch(m -> request.model().equals(m.name()) && m.available())) {
                try {
                    return inference.predictAt(ep.baseUrl(), request);
                } catch (cn.ouc.luminatrail.unikt.service.InferenceService
                        .InferenceUnavailableException e) {
                    // 该执行体刚掉线，顺延下一个候选
                }
            }
        }
        // 没有任何执行体声明该模型可用：仍交给默认服务，让它返回标准错误
        return inference.predict(request);
    }

    /** 该模型的技能目录：路由到持有它的执行体拉取；找不到返回 null。 */
    public List<cn.ouc.luminatrail.unikt.dto.SkillDto> skillsFor(String model) {
        for (Endpoint ep : endpoints()) {
            List<ModelInfo> models = modelsOf(ep);
            if (models.stream().anyMatch(m -> model.equals(m.name()) && m.available())) {
                List<cn.ouc.luminatrail.unikt.dto.SkillDto> body =
                        inference.listSkills(ep.baseUrl(), model);
                if (body != null) {
                    return body;
                }
            }
        }
        return null;
    }

    private List<ModelInfo> modelsOf(Endpoint ep) {
        String key = ep.baseUrl() == null ? "<default>" : ep.baseUrl();
        ModelsCache c = cache.get(key);
        long now = System.currentTimeMillis();
        if (c != null && now - c.at() < CACHE_TTL_MS) {
            return c.models();
        }
        List<ModelInfo> models = ep.baseUrl() == null
                ? inference.listModels()
                : fetchModels(ep.baseUrl());
        cache.put(key, new ModelsCache(now, models));
        return models;
    }

    private static List<ModelInfo> fetchModels(String baseUrl) {
        try {
            SimpleClientHttpRequestFactory f = new SimpleClientHttpRequestFactory();
            f.setConnectTimeout(Duration.ofMillis(NODE_TIMEOUT_MS));
            f.setReadTimeout(Duration.ofMillis(NODE_TIMEOUT_MS));
            List<ModelInfo> body = RestClient.builder()
                    .baseUrl(DeployService.normalize(baseUrl))
                    .requestFactory(f)
                    .build()
                    .get()
                    .uri("/models")
                    .retrieve()
                    .body(new ParameterizedTypeReference<List<ModelInfo>>() {
                    });
            return body == null ? List.of() : body;
        } catch (Exception e) {
            return List.of();
        }
    }

    private List<Endpoint> endpoints() {
        List<Endpoint> eps = new ArrayList<>();
        eps.add(new Endpoint(null, null));
        for (ComputeNode n : nodes.findAll(Sort.by("id"))) {
            if (ComputeNode.ONLINE.equals(n.getStatus())) {
                eps.add(new Endpoint(n.getName(), DeployService.normalize(n.getBaseUrl())));
            }
        }
        return eps;
    }
}
