<script setup lang="ts">
import { computed, onMounted } from "vue";
import { RouterLink, RouterView, useRouter } from "vue-router";
import { services, startServicePolling } from "./composables/useServices";
import { auth, logout } from "./composables/useAuth";

const router = useRouter();
onMounted(() => {
  startServicePolling();
  // 任一请求 401（会话过期/被禁用/被降权）→ 清身份回登录页
  window.addEventListener("unikt:unauthorized", () => {
    auth.me.value = null;
    auth.loaded.value = false;
    void router.push({ name: "login", query: { next: router.currentRoute.value.fullPath } });
  });
});

const statusClass = computed(() =>
  services.backend === "up" && (services.inferenceUp || services.nodesOnline > 0)
    ? "on"
    : "off",
);
const statusText = computed(() => {
  if (services.backend === "down") return "后端离线";
  if (services.inferenceUp || services.nodesOnline > 0) {
    return services.nodesOnline > 0
      ? `推理在线 · ${services.nodesOnline} 节点`
      : "推理在线";
  }
  return "推理离线";
});
const statusTitle = computed(() => {
  if (services.backend === "down") return "SpringBoot 后端不可达";
  if (services.nodesOnline > 0) return `${services.nodesOnline}/${services.nodesTotal} 个计算节点在线`;
  if (services.inferenceUp) return `默认推理服务在线（${services.modelCount} 个模型）`;
  return "推理服务不可达（可在「节点」页部署远程推理）";
});

async function doLogout(): Promise<void> {
  await logout();
  await router.push("/");
}
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
        <template v-if="auth.isLoggedIn.value">
          <span class="divider" aria-hidden="true"></span>
          <span class="group-label">我的</span>
          <RouterLink to="/keys">API 密钥</RouterLink>
          <template v-if="auth.isAdmin.value">
            <span class="divider" aria-hidden="true"></span>
            <span class="group-label">管理</span>
            <RouterLink to="/nodes">节点</RouterLink>
            <RouterLink to="/admin/users">用户</RouterLink>
          </template>
        </template>
      </nav>
      <div class="right">
        <span class="status" :title="statusTitle">
          <span class="dot" :class="statusClass"></span>
          {{ statusText }}
        </span>
        <template v-if="auth.isLoggedIn.value">
          <span class="user">{{ auth.me.value?.username }}</span>
          <button class="link-btn" type="button" @click="doLogout">退出</button>
        </template>
        <RouterLink v-else class="login-link" to="/login">登录 / 注册</RouterLink>
      </div>
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
  align-items: center;
  flex-wrap: wrap;
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

.divider {
  width: 1px;
  height: 1.1rem;
  background: var(--border);
  margin: 0 0.4rem;
}

.group-label {
  color: #94a3b8;
  font-size: 0.78rem;
}

.right {
  display: flex;
  align-items: center;
  gap: 0.8rem;
  white-space: nowrap;
}

.status {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  font-size: 0.85rem;
  color: var(--muted);
}

.user {
  font-size: 0.88rem;
  font-weight: 600;
}

.link-btn {
  background: none;
  border: none;
  color: var(--muted);
  font-size: 0.85rem;
  padding: 0;
}

.link-btn:hover {
  color: var(--danger);
}

.login-link {
  font-size: 0.88rem;
  color: var(--primary);
  text-decoration: none;
  font-weight: 600;
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
