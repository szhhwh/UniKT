package cn.ouc.luminatrail.unikt.dto;

/**
 * 技能目录条目：模型训练数据里的一个知识点.
 *
 * @param id 训练用稠密 id（预测请求里的 skills 取值）
 * @param name 数据集原始名称（如 "Box and Whisker"），缺失时为 null
 * @param questions 关联题目数（了解该技能覆盖范围的参考）
 */
public record SkillDto(int id, String name, int questions) {
}
