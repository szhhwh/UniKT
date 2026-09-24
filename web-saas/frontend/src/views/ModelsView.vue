<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { useRouter } from "vue-router";
import { api, type ModelInfo } from "../api/client";

const router = useRouter();
const models = ref<ModelInfo[]>([]);
const error = ref("");
const loading = ref(true);
const search = ref("");
// 默认只看已训练：体验者关心的是"现在能用什么"
const onlyReady = ref(true);

async function reload(): Promise<void> {
  error.value = "";
  try {
    models.value = await api.models();
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
  } finally {
    loading.value = false;
  }
}
onMounted(reload);

const filtered = computed(() => {
  const q = search.value.trim().toLowerCase();
  return models.value.filter((m) => {
    if (onlyReady.value && !m.available) return false;
    if (q && !m.name.toLowerCase().includes(q)) return false;
    return true;
  });
});
const readyCount = computed(() => models.value.filter((m) => m.available).length);

/** 可用模型：点击卡片直接去演练场试它。 */
function tryModel(m: ModelInfo): void {
  if (m.available) {
    void router.push({ path: "/playground", query: { model: m.name } });
  }
}
</script>

<template>
  <section>
    <div class="head">
      <h1>模型</h1>
      <span v-if="!loading" class="badge ok">可推理 {{ readyCount }}</span>
      <span v-if="!loading" class="badge muted">注册 {{ models.length }}</span>
    </div>

    <div class="toolbar">
      <input
        v-model="search"
        class="search"
        type="search"
        placeholder="搜索模型，如 DKT / AKT …"
        aria-label="搜索模型"
      />
      <label class="toggle">
        <input v-model="onlyReady" type="checkbox" />
        只看已训练
      </label>
    </div>

    <div v-if="error" class="banner-error">获取失败：{{ error }} <button class="btn ghost small" type="button" @click="reload">重试</button></div>
    <p v-else-if="loading">加载中…</p>

    <template v-else>
      <p v-if="filtered.length === 0" class="empty">
        {{ onlyReady ? "当前没有已训练的模型——取消勾选可查看全部注册模型。" : "没有匹配的模型。" }}
      </p>
      <p v-if="readyCount > 0 && filtered.some((m) => m.available)" class="hint">
        点击可用模型卡片即可到演练场试用。
      </p>
      <ul v-if="filtered.length > 0" class="model-grid">
        <li
          v-for="m in filtered"
          :key="m.name"
          class="card model"
          :class="{ ready: m.available, clickable: m.available }"
          :title="m.available ? `点击试用 ${m.name}` : undefined"
          @click="tryModel(m)"
        >
          <div class="row">
            <span class="name">{{ m.name }}</span>
            <span :class="m.available ? 'badge ok' : 'badge muted'">
              {{ m.available ? "可推理" : "未训练" }}
            </span>
          </div>
          <div class="meta">
            <template v-if="m.available">
              <span v-if="m.numSkills !== null">知识点 {{ m.numSkills }}</span>
              <span class="node-tag" :title="'由 ' + (m.node ?? '默认推理服务') + ' 提供'">
                @{{ m.node ?? "默认" }}
              </span>
              <span class="go">试用 →</span>
            </template>
            <template v-else>
              <span>管理员训练后自动上线</span>
            </template>
          </div>
        </li>
      </ul>
    </template>
  </section>
</template>

<style scoped>
.head {
  display: flex;
  align-items: center;
  gap: 0.7rem;
}

.toolbar {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin: 1rem 0 1.2rem;
}

.search {
  flex: 1;
  max-width: 340px;
  padding: 0.5rem 0.8rem;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
}

.search:focus {
  outline: none;
  border-color: var(--primary);
  box-shadow: 0 0 0 3px rgb(37 99 235 / 0.12);
}

.toggle {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  color: var(--muted);
  font-size: 0.9rem;
  user-select: none;
}

.model-grid {
  list-style: none;
  padding: 0;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(215px, 1fr));
  gap: 0.8rem;
}

.model {
  padding: 0.85rem 1rem;
  transition: border-color 0.15s ease;
}

.model.ready {
  border-color: #bbf7d0;
}

.model.clickable {
  cursor: pointer;
}

.model.clickable:hover {
  border-color: var(--primary);
  box-shadow: 0 4px 14px rgb(37 99 235 / 0.12);
}

.go {
  color: var(--primary);
  font-size: 0.8rem;
  margin-left: auto;
}

.hint {
  color: var(--muted);
  font-size: 0.86rem;
  margin: 0.2rem 0 0.9rem;
}

.model .row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.5rem;
}

.model .name {
  font-weight: 650;
  font-variant-numeric: tabular-nums;
}

.model .meta {
  margin-top: 0.3rem;
  font-size: 0.8rem;
  color: var(--muted);
  display: flex;
  flex-wrap: wrap;
  gap: 0.3rem 0.9rem;
}

.node-tag {
  color: var(--primary);
  font-weight: 600;
}

.empty {
  color: var(--muted);
}
</style>
