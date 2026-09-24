<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from "vue";
import { api, type DatasetInfo, type ModelInfo, type TrainingJobInfo } from "../api/client";

const jobs = ref<TrainingJobInfo[]>([]);
const datasets = ref<DatasetInfo[]>([]);
const models = ref<ModelInfo[]>([]);
const loadError = ref("");
const opError = ref("");
const loading = ref(true);

const datasetId = ref<number | null>(null);
const modelName = ref("");
const epochs = ref(3);
const submitting = ref(false);
const expandedId = ref<number | null>(null);

let timer: ReturnType<typeof setInterval> | null = null;

async function refresh(): Promise<void> {
  // 三路各自容错：models 慢（远程节点）不能拖垮任务/数据集的展示
  try {
    jobs.value = await api.training.list();
  } catch (e) {
    loadError.value = e instanceof Error ? e.message : String(e);
  }
  try {
    datasets.value = await api.datasets.list();
  } catch (e) {
    loadError.value = e instanceof Error ? e.message : String(e);
  }
  try {
    models.value = await api.models();
  } catch {
    // 模型清单可选（datalist 建议）；失败不阻塞训练表单
  }
  loading.value = false;
}

onMounted(async () => {
  await refresh();
  timer = setInterval(() => void refresh(), 5000);
});
onUnmounted(() => {
  if (timer !== null) clearInterval(timer);
});

const readyDatasets = computed(() =>
  datasets.value.filter((d) => d.status === "READY"),
);

const canSubmit = computed(
  () =>
    !submitting.value &&
    datasetId.value !== null &&
    /^[A-Za-z0-9]+$/.test(modelName.value.trim()) &&
    epochs.value >= 1 &&
    epochs.value <= 30,
);

const statusMeta: Record<string, { cls: string; text: string }> = {
  RUNNING: { cls: "warn", text: "训练中…" },
  DONE: { cls: "ok", text: "完成" },
  FAILED: { cls: "fail", text: "失败" },
};

async function submit(): Promise<void> {
  opError.value = "";
  submitting.value = true;
  try {
    const job = await api.training.create({
      datasetId: datasetId.value!,
      modelName: modelName.value.trim(),
      epochs: epochs.value,
    });
    expandedId.value = job.id;
    await refresh();
  } catch (e) {
    opError.value = e instanceof Error ? e.message : String(e);
  } finally {
    submitting.value = false;
  }
}

function fmtTime(iso: string): string {
  return new Date(iso).toLocaleString();
}
</script>

<template>
  <section>
    <h1>训练模型</h1>
    <p class="sub">
      用你上传的数据集在计算节点上训练模型；完成后模型自动上线，演练场即刻可用。
      训练几分钟到几十分钟（取决于数据量与轮数），可离开此页。
    </p>

    <div class="card form-card">
      <div v-if="readyDatasets.length === 0" class="banner-error">
        还没有就绪的数据集——先到「数据」页上传作答记录。
      </div>
      <div v-else class="form-grid">
        <div class="field">
          <label for="tj-ds">数据集</label>
          <select id="tj-ds" v-model.number="datasetId">
            <option v-for="d in readyDatasets" :key="d.id" :value="d.id">
              {{ d.name }}（{{ d.users }} 学生 / {{ d.skills }} 知识点 / {{ d.interactions }} 条）
            </option>
          </select>
        </div>
        <div class="field">
          <label for="tj-model">模型（注册名，如 DKT / SAKT / ATKT）</label>
          <input
            id="tj-model"
            v-model="modelName"
            list="model-names"
            maxlength="40"
            placeholder="DKT"
          />
          <datalist id="model-names">
            <option v-for="m in models" :key="m.name" :value="m.name" />
          </datalist>
          <span class="hint">
            {{ models.length }} 个可选；同名模型训练后以新数据集版本提供服务
          </span>
        </div>
        <div class="field">
          <label for="tj-epochs">训练轮数（epochs，1-30）</label>
          <input id="tj-epochs" v-model.number="epochs" type="number" min="1" max="30" />
          <span class="hint">数据量小建议 3-10 轮；过多会过拟合</span>
        </div>
      </div>
      <p v-if="opError" class="banner-error">{{ opError }}</p>
      <button class="btn" :disabled="!canSubmit" @click="submit">
        {{ submitting ? "启动中…" : "开始训练" }}
      </button>
    </div>

    <p v-if="loadError" class="banner-error">加载失败：{{ loadError }}</p>
    <p v-else-if="loading">加载中…</p>
    <template v-else>
      <p v-if="jobs.length === 0" class="empty">还没有训练任务。</p>
      <div v-for="j in jobs" :key="j.id" class="card job">
        <div class="row">
          <div class="ident">
            <span class="name">#{{ j.id }} {{ j.modelName }}</span>
            <span class="badge" :class="(statusMeta[j.status] ?? { cls: 'muted' }).cls">
              {{ (statusMeta[j.status] ?? { text: j.status }).text }}
            </span>
            <span class="meta">{{ j.datasetName }} · {{ j.epochs }} 轮</span>
          </div>
          <div class="actions">
            <button v-if="j.logTail" class="btn ghost small" @click="expandedId = expandedId === j.id ? null : j.id">
              {{ expandedId === j.id ? "收起日志" : "日志" }}
            </button>
            <span v-if="j.status === 'DONE'" class="hint">模型已上线 → 去「演练场」试用</span>
          </div>
        </div>
        <p class="meta-line">启动于 {{ fmtTime(j.createdAt) }}</p>
        <p v-if="j.lastError" class="banner-error">{{ j.lastError }}</p>
        <pre v-if="expandedId === j.id && j.logTail" class="log">{{ j.logTail }}</pre>
      </div>
    </template>
  </section>
</template>

<style scoped>
.sub {
  color: var(--muted);
  max-width: 44rem;
}

.form-card {
  margin: 1rem 0 1.4rem;
}

.form-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
  gap: 1rem;
  margin-bottom: 1rem;
}

select,
input {
  font: inherit;
  padding: 0.5rem 0.6rem;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: #fff;
}

.job {
  margin-bottom: 0.9rem;
}

.row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.8rem;
  flex-wrap: wrap;
}

.ident {
  display: flex;
  align-items: center;
  gap: 0.7rem;
  flex-wrap: wrap;
}

.name {
  font-weight: 650;
}

.meta {
  color: var(--muted);
  font-size: 0.85rem;
}

.meta-line {
  margin-top: 0.4rem;
  color: var(--muted);
  font-size: 0.82rem;
}

.actions {
  display: flex;
  gap: 0.8rem;
  align-items: center;
  flex-wrap: wrap;
}

.badge.fail {
  background: var(--danger-weak);
  color: var(--danger);
}

.log {
  margin: 0.8rem 0 0;
  background: #0f172a;
  color: #cbd5e1;
  border-radius: var(--radius-sm);
  padding: 0.9rem 1rem;
  font-size: 0.78rem;
  line-height: 1.5;
  overflow-x: auto;
  white-space: pre-wrap;
  max-height: 320px;
}

.hint {
  color: var(--muted);
  font-size: 0.83rem;
}

.empty {
  color: var(--muted);
}
</style>
