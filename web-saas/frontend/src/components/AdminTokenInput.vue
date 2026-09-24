<script setup lang="ts">
import { ref } from "vue";
import { getAdminToken, setAdminToken } from "../api/adminToken";

const open = ref(false);
const value = ref(getAdminToken());
const saved = ref(false);

function save(): void {
  setAdminToken(value.value.trim());
  saved.value = true;
  setTimeout(() => (saved.value = false), 1500);
}
</script>

<template>
  <div class="admin-token">
    <button class="toggle" type="button" @click="open = !open">
      ⚙ 管理令牌{{ value ? "（已设置）" : "" }}
    </button>
    <div v-if="open" class="panel">
      <input
        v-model="value"
        type="password"
        placeholder="X-Admin-Token（服务端 UNIKT_ADMINTOKEN）"
        @keyup.enter="save"
      />
      <button class="btn small" type="button" @click="save">保存</button>
      <span v-if="saved" class="ok">已保存</span>
    </div>
  </div>
</template>

<style scoped>
.admin-token {
  font-size: 0.85rem;
}

.toggle {
  background: none;
  border: none;
  color: var(--muted);
  padding: 0.2rem 0;
  font-size: inherit;
}

.toggle:hover {
  color: var(--text);
}

.panel {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  margin-top: 0.5rem;
  flex-wrap: wrap;
}

.panel input {
  flex: 1;
  min-width: 220px;
  padding: 0.45rem 0.7rem;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
}

.ok {
  color: var(--success);
}
</style>
