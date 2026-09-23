<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api, type HealthInfo } from "../api/client";

const health = ref<HealthInfo | null>(null);
const error = ref("");

onMounted(async () => {
  try {
    health.value = await api.health();
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
  }
});
</script>

<template>
  <section>
    <h1>UniKT SaaS</h1>
    <p>
      基于知识追踪（Knowledge Tracing）的在线学习能力评估服务。
      架构：Vite 前端 → SpringBoot 业务后端 → Python 模型推理。
    </p>

    <h2>服务状态</h2>
    <p v-if="error" class="error">后端不可达：{{ error }}</p>
    <p v-else-if="health">
      后端：{{ health.status }} ｜ 推理服务：{{
        health.inferenceUp ? "在线" : "离线"
      }}
      ｜ 可用模型：{{ health.modelCount }}
    </p>
    <p v-else>检测中…</p>
  </section>
</template>

<style scoped>
.error {
  color: #dc2626;
}
</style>
