package cn.ouc.luminatrail.unikt.dto;

/**
 * 模型条目：来自 Python 推理服务（默认或某计算节点）的模型注册表。
 *
 * @param name 模型注册名
 * @param available 是否已加载 checkpoint 可直接推理
 * @param numSkills 训练数据知识点总数（技能 id 上界），元数据缺失时为 null
 * @param node 提供该模型的节点名；null 表示默认推理服务
 */
public record ModelInfo(
        String name, boolean available, Integer numSkills, String node, String dataset) {

    /** 四参构造：无 dataset 的场景（直连默认服务反序列化用）。 */
    public ModelInfo(String name, boolean available, Integer numSkills, String node) {
        this(name, available, numSkills, node, null);
    }
}
