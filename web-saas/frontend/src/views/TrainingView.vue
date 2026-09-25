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

let refreshing = false;

/** 乐观插入但尚未被服务端 list 确认的任务 id：先于 create 发出的陈旧 list
 *  响应不含新任务，直接覆盖会把新卡抹掉并误判全终态停掉轮询。 */
const optimisticIds = new Set<number>();

async function refresh(): Promise<void> {
  if (refreshing) return; // 防重入
  refreshing = true;
  try {
    await doRefresh();
  } finally {
    refreshing = false;
  }
}

async function doRefresh(): Promise<void> {
  // opError 不在轮询路径清：提交/取消的错误要留得住（此前 5 秒被抹掉，
  // 用户来不及读）。loadError 仍每轮清（轮询自愈）。
  loadError.value = "";
  // 三路并发、各自容错：models 走远程节点可慢至十余秒，不能拖住
  // 任务/数据集的展示（此前串行 await 让"加载中"挂 10 秒+）
  const tasks = [
    api.training.list().then((v) => {
      for (const j of v) optimisticIds.delete(j.id);
      const pending = jobs.value.filter(
        (j) => optimisticIds.has(j.id) && !v.some((x) => x.id === j.id),
      );
      jobs.value = pending.length > 0 ? [...pending, ...v] : v;
    }).catch((e) => {
      loadError.value = e instanceof Error ? e.message : String(e);
    }),
    api.datasets.list().then((v) => (datasets.value = v)).catch((e) => {
      loadError.value = e instanceof Error ? e.message : String(e);
    }),
    api.models().then((v) => (models.value = v)).catch(() => {
      // 模型清单可选（datalist 建议）；失败不阻塞训练表单
    }),
  ];
  // 核心两路（任务+数据集）就绪即解除加载态；models 继续后台加载
  await Promise.race([
    Promise.allSettled([tasks[0], tasks[1]]),
    new Promise((r) => setTimeout(r, 8000)),
  ]);
  loading.value = false;
  await Promise.allSettled(tasks);
  // 任务拉取成功后记录全终态快照（乐观插入的 RUNNING 不会被它覆盖掉
  // ——快照只反映服务端确认的状态）
  lastRefreshAllTerminal = jobs.value.every((j) => j.status !== "RUNNING");
}

/** 最近一次 refresh 确认的全终态快照（自停依据，不读可能被乐观插入污染的 jobs.value）。 */
let lastRefreshAllTerminal = false;

/** 启动轮询（幂等）：有进行中任务才轮询；最近一次 refresh 确认全终态后自停。 */
function startPolling(): void {
  if (timer !== null) clearInterval(timer);
  lastRefreshAllTerminal = false;
  timer = setInterval(() => {
    if (lastRefreshAllTerminal) {
      if (timer !== null) clearInterval(timer);
      timer = null;
      return;
    }
    void refresh();
  }, 5000);
}

onMounted(async () => {
  await refresh();
  startPolling();
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
    // 乐观插入：防重入守卫可能吞掉紧随的 refresh（慢节点下 create 与
    // 轮询竞争，陈旧列表会让自停逻辑立刻杀掉新起的计时器）
    optimisticIds.add(job.id);
    jobs.value.unshift(job);
    expandedId.value = job.id;
    startPolling();
  } catch (e) {
    opError.value = e instanceof Error ? e.message : String(e);
  } finally {
    submitting.value = false;
  }
}

async function cancelJob(j: { id: number; status: string }): Promise<void> {
  if (j.status !== "RUNNING") return;
  if (!window.confirm("取消这个训练任务？（节点上的进程会被终止）")) return;
  try {
    await api.training.cancel(j.id);
    await refresh();
    startPolling(); // 取消后若有其他 RUNNING 任务继续跟踪
  } catch (e) {
    opError.value = e instanceof Error ? e.message : String(e);
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
            <option
              v-for="m in models.filter((x) => /^[A-Za-z0-9]+$/.test(x.name))"
              :key="m.name"
              :value="m.name"
            />
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
            <button
              v-if="j.status === 'RUNNING'"
              class="btn ghost small danger"
              @click="cancelJob(j)"
            >
              取消
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
