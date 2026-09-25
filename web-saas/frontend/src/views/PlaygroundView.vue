<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { useRoute } from "vue-router";
import { api, type ModelInfo, type PredictResponse, type SkillInfo } from "../api/client";

/** 一题一行：知识点 + 做对/做错。 */
interface Row {
  skill: number;
  correct: 0 | 1;
}

const route = useRoute();
const models = ref<ModelInfo[]>([]);
const catalog = ref<SkillInfo[]>([]);
const catalogLoading = ref(false);
const modelName = ref("");
const rows = ref<Row[]>([]);
/** 作答行来源的数据集：跨数据集切换模型时旧知识点 id 语义已失效，必须重填。 */
const rowsDataset = ref<string | null>(null);
const result = ref<PredictResponse | null>(null);
const error = ref("");
const busy = ref(false);
/** 首次预测要加载 checkpoint，给用户一个明确的心理预期。 */
const firstCall = ref(true);

onMounted(async () => {
  try {
    models.value = await api.models();
    const available = models.value.filter((m) => m.available);
    const fromQuery = route.query.model;
    modelName.value =
      (typeof fromQuery === "string" && available.some((m) => m.name === fromQuery)
        ? fromQuery
        : (available[0]?.name ?? ""));
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
  }
});

const availableModels = computed(() => models.value.filter((m) => m.available));
const selectedModel = computed(
  () => models.value.find((m) => m.name === modelName.value) ?? null,
);

/** 目录请求防竞态：快速切换模型时，晚返回的旧请求直接丢弃。 */
let catalogReqId = 0;

watch(modelName, async (m) => {
  catalog.value = [];
  result.value = null;
  error.value = "";
  if (!m) return;
  const reqId = ++catalogReqId;
  catalogLoading.value = true;
  try {
    const c = await api.skills(m);
    if (reqId !== catalogReqId) {
      return;
    }
    catalog.value = c;
  } catch {
    // 目录拉不到（如数据集缺映射）不阻塞预测，走 id 降级模式
  } finally {
    if (reqId === catalogReqId) {
      catalogLoading.value = false;
      // 目录就绪（或确认拉不到）后自动填示例，避免用户面对空表单；
      // 跨数据集切换时旧行的知识点 id 语义已变，必须重填（同数据集换模型则保留）
      const ds = selectedModel.value?.dataset ?? null;
      if (rows.value.length === 0 || rowsDataset.value !== ds) {
        fillDefaultRows();
      }
    }
  }
});

/** 默认行：优先用带名称的技能示例；目录不可用时退化为 id 示例。 */
function fillDefaultRows(): void {
  rowsDataset.value = selectedModel.value?.dataset ?? null;
  const ex = examples.value[0];
  if (ex) {
    rows.value = ex.rows.map((r) => ({ ...r }));
    return;
  }
  const n = Math.max(2, Math.min(4, selectedModel.value?.numSkills ?? 2));
  const ids = [0, 0, Math.min(1, n - 1), Math.min(1, n - 1), 0].slice(0, Math.max(2, n));
  rows.value = ids.map((skill, i) => ({ skill, correct: (i % 3 === 2 ? 0 : 1) as 0 | 1 }));
}

function skillLabel(id: number): string {
  const s = catalog.value.find((c) => c.id === id);
  return s?.name ? s.name.trim() : `知识点 #${id}`;
}

function optionLabel(s: SkillInfo): string {
  const name = s.name ? s.name.trim() : `知识点 #${s.id}`;
  return `${name}（${s.questions} 题）`;
}

/** 有名称的排前面（按名称排序），无名称的按 id 靠后，便于下拉查找。 */
const catalogOptions = computed(() => [
  ...catalog.value
    .filter((s) => s.name)
    .sort((a, b) => (a.name ?? "").localeCompare(b.name ?? "")),
  ...catalog.value.filter((s) => !s.name).sort((a, b) => a.id - b.id),
]);

function addRow(skill?: number): void {
  const fallback =
    rows.value.length > 0
      ? rows.value[rows.value.length - 1].skill
      : (catalog.value.find((c) => c.name)?.id ?? 0);
  rows.value.push({ skill: skill ?? fallback, correct: 1 });
  result.value = null;
}

function removeRow(i: number): void {
  rows.value.splice(i, 1);
  result.value = null;
}

function toggle(row: Row): void {
  row.correct = row.correct === 1 ? 0 : 1;
  result.value = null;
}

const namedSkills = computed(() => catalog.value.filter((c) => c.name));

/** 示例序列：从当前模型目录里取有名称的技能，避免无意义的裸 id。 */
const examples = computed<{ label: string; rows: Row[] }[]>(() => {
  const picks = namedSkills.value.slice(0, 3);
  if (picks.length < 2) return [];
  const [a, b, c] = picks;
  return [
    { label: "先对后错", rows: [
      { skill: a.id, correct: 1 }, { skill: a.id, correct: 1 },
      { skill: b.id, correct: 0 }, { skill: b.id, correct: 0 },
      { skill: a.id, correct: 1 },
    ] },
    { label: "持续答对", rows: picks.map((p) => ({ skill: p.id, correct: 1 as 0 | 1 })) },
    { label: "先错后学", rows: [
      { skill: (c ?? b).id, correct: 0 }, { skill: (c ?? b).id, correct: 0 },
      { skill: (c ?? b).id, correct: 1 }, { skill: a.id, correct: 1 }, { skill: a.id, correct: 1 },
    ] },
  ];
});

function fillExample(ex: { label: string; rows: Row[] }): void {
  rows.value = ex.rows.map((r) => ({ ...r }));
  result.value = null;
  error.value = "";
}

const canSubmit = computed(
  () => !busy.value && modelName.value !== "" && rows.value.length >= 2,
);

async function submit(): Promise<void> {
  error.value = "";
  result.value = null;
  if (!canSubmit.value) return;
  busy.value = true;
  try {
    const skills = rows.value.map((r) => r.skill);
    const responses = rows.value.map((r) => r.correct);
    result.value = await api.predict({
      model: modelName.value,
      questions: skills,
      skills,
      responses,
    });
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
  } finally {
    busy.value = false;
    firstCall.value = false;
  }
}

/** 每步人话描述：做完第 j 题（技能 X ✓/✗）→ 预测第 j+1 题（技能 Y）答对概率。 */
const steps = computed(() => {
  if (!result.value) return [];
  return result.value.predictions.map((p, j) => ({
    j,
    doneSkill: skillLabel(rows.value[j]?.skill ?? 0),
    doneOk: rows.value[j]?.correct === 1,
    nextSkill: skillLabel(rows.value[j + 1]?.skill ?? 0),
    p,
  }));
});

/** 各知识点掌握度：该技能作为"下一题"的最后一次预测（附判断时间点）。 */
const mastery = computed(() => {
  if (!result.value) return [];
  const last = new Map<number, { p: number; j: number }>();
  result.value.predictions.forEach((p, j) => {
    const nextSkill = rows.value[j + 1]?.skill;
    if (nextSkill !== undefined) {
      last.set(nextSkill, { p, j });
    }
  });
  return [...last.entries()]
    .map(([skill, { p, j }]) => ({ skill, label: skillLabel(skill), p, step: j + 1 }))
    .sort((x, y) => y.p - x.p);
});
</script>

<template>
  <section>
    <h1>推理演练场</h1>
    <p class="sub">
      按学生实际的作答过程录入：每题选一个知识点、标记对错，模型逐步预测
      「下一题答对概率」，并汇总各知识点当前掌握度。
    </p>

    <div v-if="availableModels.length === 0" class="banner-error">
      当前没有已训练的模型，无法演练。请联系管理员在「节点」页确认推理服务。
    </div>

    <div class="card form-card">
      <div class="field">
        <label for="pg-model">模型</label>
        <select id="pg-model" v-model="modelName" :disabled="availableModels.length === 0">
          <option v-for="m in availableModels" :key="m.name" :value="m.name">
            {{ m.name }}（{{ m.numSkills ?? "?" }} 个知识点{{ m.dataset ? ` · ${m.dataset}` : "" }}）
          </option>
        </select>
        <span v-if="catalogLoading" class="hint">知识点目录加载中…</span>
        <span v-else-if="catalog.length > 0" class="hint">
          共 {{ catalog.length }} 个知识点{{ namedSkills.length > 0 ? `，${namedSkills.length} 个有名称` : "" }}
        </span>
      </div>

      <div class="field">
        <label>作答记录（第 1 题在最上面）</label>
        <p v-if="!catalogLoading && catalog.length === 0" class="banner-error">
          知识点目录不可用（推理节点可能未更新或元数据缺失）。已切换为
          手动输入知识点 id（0 ~ {{ selectedModel?.numSkills ?? "?" }}），可照常预测。
        </p>
        <div class="rows" :class="{ disabled: catalogLoading }">
          <div v-for="(row, i) in rows" :key="i" class="row">
            <span class="idx">{{ i + 1 }}</span>
            <select
              v-if="catalogOptions.length > 0"
              v-model.number="row.skill"
              class="skill"
              :disabled="catalogLoading"
              @change="result = null"
            >
              <option v-for="s in catalogOptions" :key="s.id" :value="s.id">
                {{ optionLabel(s) }}
              </option>
            </select>
            <input
              v-else
              v-model.number="row.skill"
              type="number"
              min="0"
              class="skill"
              @change="result = null"
            />
            <button
              class="result-btn"
              :class="row.correct === 1 ? 'right' : 'wrong'"
              type="button"
              @click="toggle(row)"
            >
              {{ row.correct === 1 ? "✓ 对" : "✗ 错" }}
            </button>
            <button class="del" type="button" title="删除这一题" @click="removeRow(i)">×</button>
          </div>
        </div>
        <div class="row-actions">
          <button
            class="btn ghost small"
            type="button"
            :disabled="catalogLoading"
            @click="addRow()"
          >
            + 添加一题
          </button>
          <span v-if="catalogLoading" class="hint">知识点目录加载中（远程节点，需数秒）…</span>
          <span v-else-if="examples.length > 0" class="examples">
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

      <div class="submit-line">
        <button class="btn" :disabled="!canSubmit" @click="submit">
          {{ busy ? "预测中…" : "开始预测" }}
        </button>
        <span v-if="busy && firstCall" class="hint">
          首次调用需要加载模型（约 1 分钟），之后会很快…
        </span>
        <span v-else-if="rows.length > 0 && rows.length < 2" class="hint">至少录入 2 题</span>
      </div>
    </div>

    <p v-if="error" class="banner-error">{{ error }}</p>

    <div v-if="mastery.length > 0" class="card result">
      <h2 style="margin: 0 0 0.3rem">各知识点掌握度（最近判断）</h2>
      <p class="legend">
        模型对该生每个技能的最新判断；括号标注判断时学生已完成的题数
        （此后未再考到的技能沿用当时的判断）。
      </p>
      <div class="bars">
        <div v-for="m in mastery" :key="m.skill" class="bar-row">
          <span class="bar-label">
            {{ m.label }}<span class="step-note">（第 {{ m.step }} 题后）</span>
          </span>
          <div class="bar">
            <div
              class="fill"
              :class="m.p >= 0.6 ? 'high' : m.p >= 0.4 ? 'mid' : 'low'"
              :style="{ width: `${Math.max(m.p * 100, 2)}%` }"
            ></div>
          </div>
          <span class="val">{{ (m.p * 100).toFixed(1) }}%</span>
        </div>
      </div>
    </div>

    <div v-if="steps.length > 0" class="card result">
      <h2 style="margin: 0 0 0.3rem">逐步预测过程</h2>
      <ol class="steps">
        <li v-for="s in steps" :key="s.j">
          做完第 {{ s.j + 1 }} 题（{{ s.doneSkill }}
          <span :class="s.doneOk ? 'ok-text' : 'bad-text'">{{ s.doneOk ? "✓ 对" : "✗ 错" }}</span
          >）后
          → 预测第 {{ s.j + 2 }} 题（{{ s.nextSkill }}）答对概率
          <strong :class="s.p >= 0.6 ? 'ok-text' : s.p >= 0.4 ? '' : 'bad-text'">
            {{ (s.p * 100).toFixed(1) }}%
          </strong>
        </li>
      </ol>
    </div>
  </section>
</template>

<style scoped>
.sub {
  color: var(--muted);
  max-width: 44rem;
}

.form-card {
  margin-top: 1rem;
  display: flex;
  flex-direction: column;
  gap: 1.2rem;
}

select {
  font: inherit;
  padding: 0.5rem 0.6rem;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: #fff;
  max-width: 100%;
}

.rows {
  display: flex;
  flex-direction: column;
  gap: 0.45rem;
}

.rows.disabled {
  opacity: 0.55;
}

.step-note {
  color: var(--muted);
  font-size: 0.78rem;
  font-weight: 400;
}

.row {
  display: flex;
  align-items: center;
  gap: 0.6rem;
}

.idx {
  width: 1.6rem;
  color: var(--muted);
  font-size: 0.85rem;
  font-variant-numeric: tabular-nums;
  text-align: right;
}

.skill {
  flex: 1;
  min-width: 0;
}

.result-btn {
  width: 4.2rem;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 0.45rem 0;
  font-weight: 600;
}

.result-btn.right {
  background: var(--success-weak);
  color: var(--success);
  border-color: #bbf7d0;
}

.result-btn.wrong {
  background: var(--danger-weak);
  color: var(--danger);
  border-color: #fecaca;
}

.del {
  border: none;
  background: none;
  color: #94a3b8;
  font-size: 1.1rem;
  padding: 0 0.3rem;
}

.del:hover {
  color: var(--danger);
}

.row-actions {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-top: 0.5rem;
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

.submit-line {
  display: flex;
  align-items: center;
  gap: 1rem;
  flex-wrap: wrap;
}

.result {
  margin-top: 1.4rem;
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

.bar-label {
  width: 13rem;
  min-width: 8rem;
  font-size: 0.88rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
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
  width: 4rem;
  font-variant-numeric: tabular-nums;
  font-size: 0.85rem;
  color: var(--muted);
}

.steps {
  margin: 0.4rem 0 0;
  padding-left: 1.4rem;
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  line-height: 1.7;
}

.ok-text {
  color: var(--success);
  font-weight: 600;
}

.bad-text {
  color: var(--danger);
  font-weight: 600;
}
</style>
