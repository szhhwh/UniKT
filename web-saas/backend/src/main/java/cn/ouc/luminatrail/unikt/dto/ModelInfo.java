package cn.ouc.luminatrail.unikt.dto;

/**
 * 模型条目：来自 Python 推理服务的模型注册表。
 *
 * @param name 模型注册名
 * @param available 是否已加载 checkpoint 可直接推理
 * @param numSkills 训练数据知识点总数（技能 id 上界），元数据缺失时为 null
 */
public record ModelInfo(String name, boolean available, Integer numSkills) {
}
