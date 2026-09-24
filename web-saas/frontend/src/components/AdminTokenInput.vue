<script setup lang="ts">
import { ref } from "vue";
import { api } from "../api/client";
import { getAdminToken, setAdminToken } from "../api/adminToken";

const open = ref(false);
const value = ref(getAdminToken());
/** idle / checking / ok / bad / off / unreachable */
const state = ref("idle");

async function save(): Promise<void> {
  setAdminToken(value.value.trim());
  state.value = "checking";
  window.dispatchEvent(new CustomEvent("unikt:admin-token"));
  if (!value.value.trim()) {
    state.value = "idle";
    return;
  }
  // 先看后端是否开启令牌校验，再验证令牌本身（区分三态 + 网络错误）
  let required = true;
  try {
    required = (await api.health()).adminTokenRequired ?? true;
  } catch {
    state.value = "unreachable";
    return;
  }
  if (!required) {
    state.value = "off";
    return;
  }
  try {
    await api.keys.list();
    state.value = "ok";
  } catch {
    state.value = "bad";
  }
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
      <button class="btn small" type="button" @click="save">保存并验证</button>
      <span v-if="state === 'checking'" class="muted">验证中…</span>
      <span v-else-if="state === 'ok'" class="ok">✓ 令牌有效</span>
      <span v-else-if="state === 'bad'" class="bad">✗ 令牌无效，请核对</span>
      <span v-else-if="state === 'off'" class="muted">
        后端未开启令牌校验（内网模式），当前无需令牌
      </span>
      <span v-else-if="state === 'unreachable'" class="bad">无法连接后端，稍后再试</span>
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

.muted {
  color: var(--muted);
}

.ok {
  color: var(--success);
}

.bad {
  color: var(--danger);
}
</style>
