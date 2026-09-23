<script setup lang="ts">
import { computed, onMounted } from "vue";
import { RouterLink, RouterView } from "vue-router";
import { services, startServicePolling } from "./composables/useServices";

onMounted(() => startServicePolling());

const statusClass = computed(() =>
  services.backend === "up" && services.inferenceUp ? "on" : "off",
);
const statusText = computed(() =>
  services.backend === "down"
    ? "后端离线"
    : services.inferenceUp
      ? "推理在线"
      : "推理离线",
);
const statusTitle = computed(() =>
  services.backend === "down"
    ? "SpringBoot 后端不可达"
    : services.inferenceUp
      ? `Python 推理服务在线（${services.modelCount} 个模型）`
      : "推理服务不可达（需在 WSL 启动 unikt-inference）",
);
</script>

<template>
  <div class="layout">
    <header class="topbar">
      <RouterLink to="/" class="brand">
        <span class="logo" aria-hidden="true"></span>
        UniKT SaaS
      </RouterLink>
      <nav>
        <RouterLink to="/">首页</RouterLink>
        <RouterLink to="/models">模型</RouterLink>
        <RouterLink to="/playground">演练场</RouterLink>
        <RouterLink to="/docs">文档</RouterLink>
      </nav>
      <span class="status" :title="statusTitle">
        <span class="dot" :class="statusClass"></span>
        {{ statusText }}
      </span>
    </header>
    <main class="content">
      <RouterView />
    </main>
  </div>
</template>

<style scoped>
.topbar {
  position: sticky;
  top: 0;
  z-index: 10;
  display: flex;
  align-items: center;
  gap: 1.6rem;
  padding: 0.65rem 1.5rem;
  background: rgb(255 255 255 / 88%);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid var(--border);
}

.brand {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-weight: 700;
  font-size: 1.05rem;
  color: var(--text);
  text-decoration: none;
}

.logo {
  width: 12px;
  height: 12px;
  border-radius: 4px;
  background: linear-gradient(135deg, var(--primary), #7c3aed);
}

.topbar nav {
  display: flex;
  gap: 0.4rem;
  flex: 1;
}

.topbar nav a {
  color: var(--muted);
  text-decoration: none;
  padding: 0.3rem 0.8rem;
  border-radius: 8px;
  transition:
    background 0.15s ease,
    color 0.15s ease;
}

.topbar nav a:hover {
  color: var(--text);
  background: #f1f5f9;
}

.topbar nav a.router-active {
  color: var(--primary);
  background: var(--primary-weak);
  font-weight: 600;
}

.status {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  font-size: 0.85rem;
  color: var(--muted);
  white-space: nowrap;
}

.content {
  max-width: 1020px;
  margin: 0 auto;
  padding: 1.8rem 1.5rem 3rem;
}

@media (max-width: 640px) {
  .topbar {
    flex-wrap: wrap;
    gap: 0.5rem;
  }
}
</style>
