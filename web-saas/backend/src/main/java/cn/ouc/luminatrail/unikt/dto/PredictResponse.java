package cn.ouc.luminatrail.unikt.dto;

import java.util.List;

/**
 * 推理结果：每个时间步预测下一题答对的概率。
 *
 * @param model 模型名
 * @param predictions 逐步掌握度预测（0~1），长度 = 输入序列长度 - 1
 */
public record PredictResponse(String model, List<Double> predictions) {
}
