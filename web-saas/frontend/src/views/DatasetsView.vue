<script setup lang="ts">
import { onMounted, onUnmounted, ref } from "vue";
import { api, type DatasetInfo } from "../api/client";

const datasets = ref<DatasetInfo[]>([]);
const loadError = ref("");
const opError = ref("");
const loading = ref(true);

const name = ref("");
const interactionsFile = ref<File | null>(null);
const skillsFile = ref<File | null>(null);
const uploading = ref(false);
/** 上传成功后的统计预览。 */
const fresh = ref<DatasetInfo | null>(null);

let timer: ReturnType<typeof setInterval> | null = null;

async function refresh(): Promise<void> {
  try {
    datasets.value = await api.datasets.list();
    loadError.value = "";
  } catch (e) {
    loadError.value = e instanceof Error ? e.message : String(e);
  } finally {
    loading.value = false;
  }
}

onMounted(async () => {
  await refresh();
  timer = setInterval(() => void refresh(), 10000);
});
onUnmounted(() => {
  if (timer !== null) clearInterval(timer);
});

function onInteractions(e: Event): void {
  const f = (e.target as HTMLInputElement).files?.[0] ?? null;
  interactionsFile.value = f;
  if (f && !name.value) {
    name.value = f.name.replace(/\.csv$/i, "").slice(0, 60);
  }
}

function onSkills(e: Event): void {
  skillsFile.value = (e.target as HTMLInputElement).files?.[0] ?? null;
}

async function upload(): Promise<void> {
  opError.value = "";
  fresh.value = null;
  uploading.value = true;
  try {
    fresh.value = await api.datasets.upload(
      name.value.trim(),
      interactionsFile.value!,
      skillsFile.value,
    );
    interactionsFile.value = null;
    skillsFile.value = null;
    name.value = "";
    await refresh();
  } catch (e) {
    opError.value = e instanceof Error ? e.message : String(e);
  } finally {
    uploading.value = false;
  }
}

async function remove(d: DatasetInfo): Promise<void> {
  if (!window.confirm(`删除数据集「${d.name}」？（已训练出的模型不受影响）`)) {
    return;
  }
  try {
    await api.datasets.remove(d.id);
    await refresh();
  } catch (e) {
    opError.value = e instanceof Error ? e.message : String(e);
  }
}
</script>

<template>
  <section>
    <h1>我的数据</h1>
    <p class="sub">
      上传你课程的作答记录（CSV），用于训练自己的模型。格式要求见下方说明。
    </p>

    <div class="card form-card">
      <div class="form-grid">
        <div class="field">
          <label for="ds-name">数据集名称</label>
          <input
            id="ds-name"
            v-model="name"
            maxlength="60"
            placeholder="例如：初一下学期数学"
          />
        </div>
        <div class="field">
          <label for="ds-inter">作答记录 interactions.csv（必需）</label>
          <input
            id="ds-inter"
            type="file"
            accept=".csv,text/csv"
            @change="onInteractions"
          />
        </div>
        <div class="field">
          <label for="ds-skills">知识点名称 skills.csv（可选）</label>
          <input id="ds-skills" type="file" accept=".csv,text/csv" @change="onSkills" />
        </div>
      </div>
      <p v-if="opError" class="banner-error">{{ opError }}</p>
      <button
        class="btn"
        :disabled="uploading || !name.trim() || !interactionsFile"
        @click="upload"
      >
        {{ uploading ? "校验中…" : "上传并校验" }}
      </button>
      <details class="fmt">
        <summary>格式要求</summary>
        <pre class="example">interactions.csv（表头固定，correct 只能 0/1，timestamp 可省略）：
user_id,item_id,skill_id,correct,timestamp
s001,q_12,k3,1,1710000001
s001,q_15,k3,0,1710000030

skills.csv（可选，提供后训练出的模型在演练场显示知识点名称）：
skill_id,skill_name
k3,一元一次方程</pre>
        <p class="hint">
          至少 2 个学生、2 个知识点、100 条作答；建议几千条以上（过少模型学不到东西）。
        </p>
      </details>
    </div>

    <div v-if="fresh" class="card fresh">
      <p><strong>「{{ fresh.name }}」校验通过：</strong></p>
      <p class="stats">
        {{ fresh.interactions }} 条作答 · {{ fresh.users }} 学生 ·
        {{ fresh.skills }} 知识点 · {{ fresh.questions }} 题
        {{ fresh.hasSkillNames ? " · 含知识点名称" : "" }}
      </p>
      <p class="hint">到「训练」页即可用这个数据集训练模型。</p>
    </div>

    <p v-if="loadError" class="banner-error">加载失败：{{ loadError }}</p>
    <p v-else-if="loading">加载中…</p>
    <template v-else>
      <p v-if="datasets.length === 0" class="empty">还没有数据集。</p>
      <div v-else class="card list">
        <table class="table">
          <thead>
            <tr>
              <th>名称</th>
              <th>作答数</th>
              <th>学生</th>
              <th>知识点</th>
              <th>题数</th>
              <th>状态</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="d in datasets" :key="d.id">
              <td>{{ d.name }}</td>
              <td class="num">{{ d.interactions }}</td>
              <td class="num">{{ d.users }}</td>
              <td class="num">{{ d.skills }}</td>
              <td class="num">{{ d.questions }}</td>
              <td>
                <span :class="d.status === 'READY' ? 'badge ok' : 'badge muted'">
                  {{ d.status === "READY" ? "就绪" : "异常" }}
                </span>
              </td>
              <td>
                <button class="btn ghost small danger" @click="remove(d)">删除</button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </template>
  </section>
</template>

<style scoped>
.sub {
  color: var(--muted);
  max-width: 42rem;
}

.form-card {
  margin: 1rem 0;
}

.form-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
  gap: 1rem;
  margin-bottom: 1rem;
}

.form-grid input[type="file"] {
  padding: 0.35rem;
  border: 1px dashed var(--border);
  border-radius: var(--radius-sm);
  background: #fff;
}

.fmt {
  margin-top: 0.9rem;
}

.fmt summary {
  cursor: pointer;
  color: var(--muted);
  font-size: 0.86rem;
}

.example {
  background: #0f172a;
  color: #cbd5e1;
  border-radius: var(--radius-sm);
  padding: 0.9rem 1rem;
  font-size: 0.78rem;
  line-height: 1.6;
  overflow-x: auto;
}

.fresh {
  border-color: #bbf7d0;
  background: #f0fdf4;
}

.stats {
  font-variant-numeric: tabular-nums;
}

.hint {
  color: var(--muted);
  font-size: 0.83rem;
}

.empty {
  color: var(--muted);
}

.table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.92rem;
}

.table th,
.table td {
  text-align: left;
  padding: 0.5rem 0.7rem;
  border-bottom: 1px solid var(--border);
}

.table th {
  color: var(--muted);
  font-weight: 600;
  font-size: 0.82rem;
}

.num {
  font-variant-numeric: tabular-nums;
}

.btn.danger:hover {
  border-color: var(--danger);
  color: var(--danger);
  background: transparent;
}
</style>
