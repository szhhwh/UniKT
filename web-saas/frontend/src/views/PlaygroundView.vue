<script setup lang="ts">
import { ref } from "vue";
import { api, type PredictResponse } from "../api/client";

const model = ref("DKT");
const sequenceText = ref("1,1,0,1,1,1,0,1");
const result = ref<PredictResponse | null>(null);
const error = ref("");
const busy = ref(false);

async function submit() {
  error.value = "";
  result.value = null;
  const responses = sequenceText.value
    .split(/[,，\s]+/)
    .filter(Boolean)
    .map(Number);
  if (responses.some((r) => Number.isNaN(r) || (r !== 0 && r !== 1))) {
    error.value = "作答序列只能是 0/1，逗号或空格分隔";
    return;
  }
  if (responses.length < 2) {
    error.value = "至少输入 2 步作答";
    return;
  }
  // 脚手架阶段：题目/知识点用占位 id 递增，接入真实数据集后替换
  const questions = responses.map((_, i) => i + 1);
  const skills = responses.map(() => 1);
  busy.value = true;
  try {
    result.value = await api.predict({
      model: model.value,
      questions,
      skills,
      responses,
    });
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
  } finally {
    busy.value = false;
  }
}
</script>

<template>
  <section>
    <h1>推理演练场</h1>
    <p>输入一段作答序列，观察掌握度预测的变化（当前为基线实现）。</p>

    <form class="form" @submit.prevent="submit">
      <label>
        模型
        <input v-model="model" placeholder="DKT / AKT / …" />
      </label>
      <label>
        作答序列（1 对 0 错）
        <input v-model="sequenceText" placeholder="例如 1,1,0,1" />
      </label>
      <button type="submit" :disabled="busy">
        {{ busy ? "预测中…" : "预测" }}
      </button>
    </form>

    <p v-if="error" class="error">{{ error }}</p>

    <div v-if="result" class="result">
      <h2>{{ result.model }} 预测结果</h2>
      <div class="bars">
        <div
          v-for="(p, i) in result.predictions"
          :key="i"
          class="bar-row"
          :title="`第 ${i + 1} 步后：${p}`"
        >
          <span class="step">t{{ i + 1 }}</span>
          <div class="bar">
            <div class="fill" :style="{ width: `${p * 100}%` }"></div>
          </div>
          <span class="val">{{ p }}</span>
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.form {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  max-width: 420px;
}
.form label {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  font-size: 0.9rem;
}
.form input {
  padding: 0.5rem;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
}
.form button {
  align-self: flex-start;
  padding: 0.5rem 1.5rem;
  background: #2563eb;
  color: #fff;
  border: none;
  border-radius: 6px;
}
.bars {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  max-width: 560px;
}
.bar-row {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.step {
  width: 3rem;
  color: #64748b;
  font-size: 0.85rem;
}
.bar {
  flex: 1;
  height: 14px;
  background: #e2e8f0;
  border-radius: 4px;
  overflow: hidden;
}
.fill {
  height: 100%;
  background: #2563eb;
}
.val {
  width: 3.5rem;
  font-variant-numeric: tabular-nums;
  font-size: 0.85rem;
}
.error {
  color: #dc2626;
}
</style>
