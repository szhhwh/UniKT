package cn.ouc.luminatrail.unikt.node;

import cn.ouc.luminatrail.unikt.dto.ModelInfo;
import cn.ouc.luminatrail.unikt.dto.PredictRequest;
import cn.ouc.luminatrail.unikt.dto.PredictResponse;
import cn.ouc.luminatrail.unikt.dataset.UserDataset;
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
    private static final long SKILLS_TTL_MS = 300000;
    private static final int NODE_TIMEOUT_MS = 8000;

    /** 一个推理执行体（默认服务或某节点）。 */
    private record Endpoint(String name, String baseUrl) {
    }

    /** 执行体级模型缓存。 */
    private record ModelsCache(long at, List<ModelInfo> models) {
    }

    /** 模型级技能目录缓存（目录只在重跑预处理后变化，长 TTL 足够）。 */
    private record SkillsCache(long at, List<cn.ouc.luminatrail.unikt.dto.SkillDto> skills) {
    }

    private final NodeRepository nodes;
    private final InferenceService inference;
    private final cn.ouc.luminatrail.unikt.common.InferenceClients clients;
    private final cn.ouc.luminatrail.unikt.dataset.DatasetRepository datasets;
    private final Map<String, ModelsCache> cache = new ConcurrentHashMap<>();
    private final Map<String, SkillsCache> skillsCache = new ConcurrentHashMap<>();

    public NodeRoutingService(NodeRepository nodes, InferenceService inference,
                              cn.ouc.luminatrail.unikt.common.InferenceClients clients,
                              cn.ouc.luminatrail.unikt.dataset.DatasetRepository datasets) {
        this.nodes = nodes;
        this.inference = inference;
        this.clients = clients;
        this.datasets = datasets;
    }

    /**
     * 聚合模型清单：同名模型合并 available（OR）与首个提供者。
     *
     * @param viewerId 当前用户 id（匿名 null）
     * @param viewerAdmin 是否管理员
     */
    public List<ModelInfo> aggregateModels(Long viewerId, boolean viewerAdmin) {
        Map<String, ModelInfo> merged = new HashMap<>();
        for (Endpoint ep : endpoints()) {
            for (ModelInfo m : modelsOf(ep)) {
                // 归属过滤：generic 限定模型（用户自有数据训练）只对
                // 其 owner 与管理员可见；裸名内置模型对所有人可见
                if (!visibleTo(m, viewerId, viewerAdmin)) {
                    continue;
                }
                // 上游不感知节点名，这里按执行体打标（默认服务保持 null）
                ModelInfo tagged = m.node() == null
                        ? new ModelInfo(m.name(), m.available(), m.numSkills(), ep.name(),
                                m.dataset())
                        : m;
                merged.merge(tagged.name(), tagged, (a, b) -> new ModelInfo(
                        a.name(), a.available() || b.available(),
                        a.numSkills() != null ? a.numSkills() : b.numSkills(),
                        a.available() ? a.node() : b.node(),
                        a.dataset() != null ? a.dataset() : b.dataset()));
            }
        }
        return new ArrayList<>(merged.values());
    }

    /** 裸名与非 generic 数据集 → 公开；generic 数据集 → owner 或管理员。 */
    private boolean visibleTo(ModelInfo m, Long viewerId, boolean viewerAdmin) {
        String ds = m.dataset();
        if (ds == null || !ds.startsWith("generic_")) {
            return true;
        }
        if (viewerAdmin) {
            return true;
        }
        if (viewerId == null) {
            return false;
        }
        return datasets.findBySlug(ds)
                .map(UserDataset::getOwnerId)
                .map(viewerId::equals)
                .orElse(false);
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

    /**
     * 预测前的模型访问校验：限定名（含 @generic）要求 viewer 是 owner
     * 或管理员。返回 null 表示放行；否则返回错误体 Map。
     */
    public java.util.Map<String, Object> checkModelAccess(
            String model, Long viewerId, boolean viewerAdmin) {
        if (model == null || !model.contains("@")) {
            return null; // 裸名走引擎的非 generic 解析，公共模型
        }
        String ds = model.substring(model.indexOf('@') + 1);
        if (!ds.startsWith("generic_")) {
            return null;
        }
        if (viewerAdmin) {
            return null;
        }
        boolean allowed = viewerId != null && datasets.findBySlug(ds)
                .map(UserDataset::getOwnerId)
                .map(viewerId::equals)
                .orElse(false);
        return allowed ? null
                : java.util.Map.of("status", 404,
                        "message", "模型不存在或无权访问");
    }

    /** 该模型的技能目录：路由到持有它的执行体拉取（按模型缓存 5 分钟）；找不到返回 null。 */
    public List<cn.ouc.luminatrail.unikt.dto.SkillDto> skillsFor(String model) {
        SkillsCache c = skillsCache.get(model);
        long now = System.currentTimeMillis();
        if (c != null && now - c.at() < SKILLS_TTL_MS) {
            return c.skills();
        }
        for (Endpoint ep : endpoints()) {
            List<ModelInfo> models = modelsOf(ep);
            if (models.stream().anyMatch(m -> model.equals(m.name()) && m.available())) {
                List<cn.ouc.luminatrail.unikt.dto.SkillDto> body =
                        inference.listSkills(ep.baseUrl(), model);
                if (body != null) {
                    skillsCache.put(model, new SkillsCache(now, body));
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

    private List<ModelInfo> fetchModels(String baseUrl) {
        try {
            List<ModelInfo> body = clients
                    .client(DeployService.normalize(baseUrl), NODE_TIMEOUT_MS)
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
