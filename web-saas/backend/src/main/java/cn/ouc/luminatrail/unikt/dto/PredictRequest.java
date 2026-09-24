package cn.ouc.luminatrail.unikt.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import java.util.List;

/**
 * 知识追踪推理请求：一条学习者作答序列。
 *
 * @param model    模型名，对应 model/ 目录下的注册名（如 "DKT"、"AKT"）
 * @param questions 兼容字段（旧占位，与 skills 同义），可省略
 * @param skills   知识点(技能) id 序列，与 responses 对齐；目录见 /api/skills
 * @param responses 作答对错序列（1 对 0 错），与 skills 对齐
 */
public record PredictRequest(
        @NotBlank String model,
        List<Integer> questions,
        @NotEmpty List<Integer> skills,
        @NotEmpty List<Integer> responses) {

    public PredictRequest {
        if (questions != null && questions.size() != skills.size()) {
            throw new IllegalArgumentException("questions/skills 长度必须一致");
        }
        if (skills.size() != responses.size()) {
            throw new IllegalArgumentException("skills/responses 长度必须一致");
        }
        if (questions == null) {
            // Python 侧同名字段为可选；缺省时用 skills 补齐保持契约一致
            questions = skills;
        }
    }
}
