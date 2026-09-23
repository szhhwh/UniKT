<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api, type ModelInfo } from "../api/client";

const models = ref<ModelInfo[]>([]);
const error = ref("");
const loading = ref(true);

onMounted(async () => {
  try {
    models.value = await api.models();
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
  } finally {
    loading.value = false;
  }
});
</script>

<template>
  <section>
    <h1>模型列表</h1>
    <p v-if="loading">加载中…</p>
    <p v-else-if="error" class="error">
      获取失败：{{ error }}（请确认 SpringBoot 8080 与推理服务 8100 已启动）
    </p>
    <template v-else>
      <p>共 {{ models.length }} 个已注册模型。</p>
      <ul class="model-list">
        <li v-for="m in models" :key="m.name">
          <span class="name">{{ m.name }}</span>
          <span :class="m.available ? 'ok' : 'muted'">
            {{ m.available ? "已加载" : "未加载" }}
          </span>
        </li>
      </ul>
    </template>
  </section>
</template>

<style scoped>
.model-list {
  list-style: none;
  padding: 0;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 0.5rem;
}
.model-list li {
  display: flex;
  justify-content: space-between;
  background: #fff;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  padding: 0.5rem 0.75rem;
}
.name {
  font-weight: 600;
}
.ok {
  color: #16a34a;
}
.muted {
  color: #94a3b8;
}
.error {
  color: #dc2626;
}
</style>
