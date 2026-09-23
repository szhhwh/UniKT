package cn.ouc.luminatrail.unikt.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import java.util.List;

/**
 * 知识追踪推理请求：一条学习者作答序列。
 *
 * @param model    模型名，对应 model/ 目录下的注册名（如 "DKT"、"AKT"）
 * @param questions 题目 id 序列
 * @param skills   知识点(技能) id 序列，与 questions 对齐
 * @param responses 作答对错序列（1 对 0 错），与 questions 对齐
 */
public record PredictRequest(
        @NotBlank String model,
        @NotEmpty List<Integer> questions,
        @NotEmpty List<Integer> skills,
        @NotEmpty List<Integer> responses) {

    public PredictRequest {
        if (questions.size() != skills.size() || questions.size() != responses.size()) {
            throw new IllegalArgumentException("questions/skills/responses 长度必须一致");
        }
    }
}
