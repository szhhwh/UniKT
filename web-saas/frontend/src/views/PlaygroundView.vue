<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { api, type ModelInfo, type PredictResponse } from "../api/client";

const models = ref<ModelInfo[]>([]);
const modelsError = ref("");
const modelName = ref("DKT");
const skillsText = ref("5,5,5,12,12,7,7,7");
const responsesText = ref("1,1,0,0,1,1,1,0");
const result = ref<PredictResponse | null>(null);
const error = ref("");
const busy = ref(false);

onMounted(async () => {
  try {
    models.value = await api.models();
  } catch (e) {
    modelsError.value = e instanceof Error ? e.message : String(e);
  }
});

const availableModels = computed(() => models.value.filter((m) => m.available));
const selectedModel = computed(
  () => models.value.find((m) => m.name === modelName.value.trim()) ?? null,
);

function parseList(text: string): number[] | null {
  const parts = text.split(/[,，\s]+/).filter(Boolean);
  if (parts.length === 0) return null;
  const nums = parts.map(Number);
  if (nums.some((n) => !Number.isInteger(n))) return null;
  return nums;
}

const skills = computed(() => parseList(skillsText.value));
const responses = computed(() => {
  const list = parseList(responsesText.value);
  if (list === null) return null;
  return list.every((r) => r === 0 || r === 1) ? list : null;
});

const lengthMismatch = computed(
  () =>
    skills.value !== null &&
    responses.value !== null &&
    skills.value.length !== responses.value.length,
);

const skillRangeError = computed(() => {
  const sel = selectedModel.value;
  if (!sel || !sel.available || sel.numSkills === null) return null;
  if (!skills.value) return null;
  const bad = skills.value.filter((s) => s < 0 || s >= (sel.numSkills ?? 0));
  return bad.length > 0
    ? `知识点 id ${[...new Set(bad)].join(", ")} 超出范围 [0, ${sel.numSkills})`
    : null;
});

const formError = computed(() => {
  if (skills.value === null) return "知识点序列格式不对，应为逗号分隔的整数";
  if (skills.value.length < 2) return "至少 2 个知识点";
  if (responses.value === null)
    return "作答序列格式不对，应为逗号分隔的 0/1";
  if (lengthMismatch.value)
    return `两个序列长度不一致（知识点 ${skills.value.length} vs 作答 ${responses.value.length}）`;
  if (selectedModel.value && !selectedModel.value.available)
    return `模型 ${modelName.value} 尚未训练（先在 WSL 运行 train.py）`;
  return skillRangeError.value;
});

const canSubmit = computed(
  () => !busy.value && formError.value === null && modelName.value.trim() !== "",
);

const examples: { label: string; skills: string; responses: string }[] = [
  { label: "先对后错", skills: "5,5,5,12,12,7,7,7", responses: "1,1,0,0,1,1,1,0" },
  { label: "持续答对", skills: "3,3,4,4,9,9,9,9", responses: "1,1,1,1,1,1,1,1" },
  { label: "先错后学", skills: "8,8,8,8,2,2,2", responses: "0,0,0,1,1,1,1" },
];

function fillExample(ex: (typeof examples)[number]) {
  skillsText.value = ex.skills;
  responsesText.value = ex.responses;
  result.value = null;
  error.value = "";
}

async function submit() {
  error.value = "";
  result.value = null;
  if (!canSubmit.value || !skills.value || !responses.value) return;
  busy.value = true;
  try {
    result.value = await api.predict({
      model: modelName.value.trim(),
      questions: skills.value,
      skills: skills.value,
      responses: responses.value,
    });
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
  } finally {
    busy.value = false;
  }
}

const avgPrediction = computed(() => {
  if (!result.value || result.value.predictions.length === 0) return null;
  const mean =
    result.value.predictions.reduce((a, b) => a + b, 0) /
    result.value.predictions.length;
  return (mean * 100).toFixed(1);
});
</script>

<template>
  <section>
    <h1>推理演练场</h1>
    <p class="sub">
      输入一段作答序列，真实模型逐步预测「下一题答对概率」。首次调用会加载
      checkpoint，稍慢；之后同模型走缓存。
    </p>

    <div class="card form-card">
      <div class="form-grid">
        <div class="field">
          <label for="model">模型</label>
          <input
            id="model"
            v-model="modelName"
            list="model-options"
            placeholder="DKT / AKT / …"
          />
          <datalist id="model-options">
            <option v-for="m in availableModels" :key="m.name" :value="m.name">
              {{ m.numSkills !== null ? `${m.numSkills} 个知识点` : "" }}
            </option>
          </datalist>
          <span v-if="selectedModel" class="hint">
            {{
              selectedModel.available
                ? selectedModel.numSkills !== null
                  ? `已训练 · 知识点 id 范围 [0, ${selectedModel.numSkills})`
                  : "已训练"
                : "未训练"
            }}
          </span>
          <span v-else-if="modelsError" class="error-text">{{ modelsError }}</span>
        </div>

        <div class="field">
          <label for="skills">知识点序列</label>
          <input id="skills" v-model="skillsText" placeholder="例如 5,5,5,12,12" />
          <span class="hint">逗号分隔的概念 id，与作答一一对应</span>
        </div>

        <div class="field">
          <label for="responses">作答序列（1 对 0 错）</label>
          <input id="responses" v-model="responsesText" placeholder="例如 1,1,0,0,1" />
          <span class="hint">与知识点序列等长</span>
        </div>
      </div>

      <p v-if="formError" class="banner-error">{{ formError }}</p>

      <div class="actions">
        <button class="btn" :disabled="!canSubmit" @click="submit">
          {{ busy ? "预测中…" : "开始预测" }}
        </button>
        <span class="examples">
          示例：
          <button
            v-for="ex in examples"
            :key="ex.label"
            class="btn ghost small"
            type="button"
            @click="fillExample(ex)"
          >
            {{ ex.label }}
          </button>
        </span>
      </div>
    </div>

    <p v-if="error" class="banner-error">{{ error }}</p>

    <div v-if="result" class="card result">
      <div class="result-head">
        <h2 style="margin: 0">{{ result.model }} 预测结果</h2>
        <span v-if="avgPrediction !== null" class="badge ok">
          平均答对概率 {{ avgPrediction }}%
        </span>
      </div>
      <p class="legend">
        第 j 步柱形 = 看完前 j 步作答后，模型对第 j+1 题的答对概率
      </p>
      <div class="bars">
        <div
          v-for="(p, i) in result.predictions"
          :key="i"
          class="bar-row"
          :title="`第 ${i + 1} 步后 → 下一题答对概率 ${p}`"
        >
          <span class="step">t{{ i + 1 }}</span>
          <div class="bar">
            <div
              class="fill"
              :class="p >= 0.6 ? 'high' : p >= 0.4 ? 'mid' : 'low'"
              :style="{ width: `${Math.max(p * 100, 2)}%` }"
            ></div>
          </div>
          <span class="val">{{ p.toFixed(4) }}</span>
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.sub {
  color: var(--muted);
  max-width: 42rem;
}

.form-card {
  margin-top: 1rem;
}

.form-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 1rem;
}

.actions {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-top: 1.1rem;
  flex-wrap: wrap;
}

.examples {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  color: var(--muted);
  font-size: 0.88rem;
  flex-wrap: wrap;
}

.result {
  margin-top: 1.4rem;
}

.result-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  flex-wrap: wrap;
}

.legend {
  color: var(--muted);
  font-size: 0.84rem;
}

.bars {
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
  margin-top: 0.6rem;
}

.bar-row {
  display: flex;
  align-items: center;
  gap: 0.6rem;
}

.step {
  width: 2.6rem;
  color: var(--muted);
  font-size: 0.82rem;
  font-variant-numeric: tabular-nums;
}

.bar {
  flex: 1;
  height: 16px;
  background: #eef2f7;
  border-radius: 5px;
  overflow: hidden;
}

.fill {
  height: 100%;
  border-radius: 5px;
  transition: width 0.4s ease;
}

.fill.high {
  background: linear-gradient(90deg, #2563eb, #3b82f6);
}

.fill.mid {
  background: linear-gradient(90deg, #f59e0b, #fbbf24);
}

.fill.low {
  background: linear-gradient(90deg, #dc2626, #f87171);
}

.val {
  width: 3.6rem;
  font-variant-numeric: tabular-nums;
  font-size: 0.85rem;
  color: var(--muted);
}
</style>
