<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { api, ApiError, type AdminUserInfo } from "../api/client";

const users = ref<AdminUserInfo[]>([]);
const error = ref("");
const loading = ref(true);

async function reload(): Promise<void> {
  try {
    users.value = await api.admin.users();
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
  } finally {
    loading.value = false;
  }
}
onMounted(reload);

const rows = computed(() => users.value);

async function patch(u: AdminUserInfo, body: { role?: string; active?: boolean }): Promise<void> {
  error.value = "";
  try {
    await api.admin.updateUser(u.id, body);
    users.value = await api.admin.users();
  } catch (e) {
    error.value =
      e instanceof ApiError ? e.message : e instanceof Error ? e.message : String(e);
  }
}

function toggleRole(u: AdminUserInfo): void {
  void patch(u, { role: u.role === "ADMIN" ? "USER" : "ADMIN" });
}

function toggleActive(u: AdminUserInfo): void {
  void patch(u, { active: !u.active });
}

function fmt(iso: string): string {
  return new Date(iso).toLocaleDateString();
}
</script>

<template>
  <section>
    <h1>用户管理</h1>
    <p class="sub">管理员可见。提升/降级管理员、启用/禁用账号。</p>

    <div v-if="error" class="banner-error">{{ error }} <button class="btn ghost small" type="button" @click="reload">重试</button></div>
    <p v-else-if="loading">加载中…</p>

    <div v-else class="card">
      <table class="table">
        <thead>
          <tr>
            <th>用户名</th>
            <th>角色</th>
            <th>状态</th>
            <th>密钥数</th>
            <th>注册时间</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="u in rows" :key="u.id">
            <td>{{ u.username }}</td>
            <td>
              <span :class="u.role === 'ADMIN' ? 'badge ok' : 'badge muted'">{{ u.role }}</span>
            </td>
            <td>{{ u.active ? "启用" : "禁用" }}</td>
            <td class="num">{{ u.keyCount }}</td>
            <td class="num">{{ fmt(u.createdAt) }}</td>
            <td class="ops">
              <button class="btn ghost small" @click="toggleRole(u)">
                {{ u.role === "ADMIN" ? "降为普通" : "升为管理员" }}
              </button>
              <button class="btn ghost small danger" @click="toggleActive(u)">
                {{ u.active ? "禁用" : "启用" }}
              </button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </section>
</template>

<style scoped>
.sub {
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

.ops {
  display: flex;
  gap: 0.4rem;
  flex-wrap: wrap;
}

.btn.danger:hover {
  border-color: var(--danger);
  color: var(--danger);
  background: transparent;
}
</style>
