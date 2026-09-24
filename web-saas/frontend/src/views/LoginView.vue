<script setup lang="ts">
import { ref } from "vue";
import { useRoute, useRouter } from "vue-router";
import { ApiError } from "../api/client";
import { login, register } from "../composables/useAuth";

const route = useRoute();
const router = useRouter();
const mode = ref<"login" | "register">("login");
const username = ref("");
const password = ref("");
const error = ref("");
const busy = ref(false);

async function submit(): Promise<void> {
  error.value = "";
  busy.value = true;
  try {
    if (mode.value === "login") {
      await login(username.value.trim(), password.value);
    } else {
      await register(username.value.trim(), password.value);
    }
    const next = typeof route.query.next === "string" ? route.query.next : "/";
    await router.push(next);
  } catch (e) {
    if (e instanceof ApiError && e.status === 401) {
      error.value = "用户名或密码不正确";
    } else if (e instanceof ApiError && e.status === 409) {
      error.value = "用户名已被占用";
    } else {
      error.value = e instanceof Error ? e.message : String(e);
    }
  } finally {
    busy.value = false;
  }
}
</script>

<template>
  <section class="wrap">
    <div class="card box">
      <div class="tabs">
        <button
          :class="mode === 'login' ? 'tab active' : 'tab'"
          type="button"
          @click="mode = 'login'; error = ''"
        >
          登录
        </button>
        <button
          :class="mode === 'register' ? 'tab active' : 'tab'"
          type="button"
          @click="mode = 'register'; error = ''"
        >
          注册
        </button>
      </div>

      <div class="field">
        <label for="lg-user">用户名</label>
        <input
          id="lg-user"
          v-model="username"
          maxlength="30"
          placeholder="字母 / 数字 / 下划线，3-30 位"
          @keyup.enter="submit"
        />
      </div>
      <div class="field">
        <label for="lg-pass">密码</label>
        <input
          id="lg-pass"
          v-model="password"
          type="password"
          maxlength="64"
          :placeholder="mode === 'register' ? '至少 8 位' : '密码'"
          @keyup.enter="submit"
        />
      </div>

      <p v-if="error" class="banner-error">{{ error }}</p>

      <button class="btn full" :disabled="busy || !username || !password" @click="submit">
        {{ busy ? "请稍候…" : mode === "login" ? "登录" : "注册并登录" }}
      </button>

      <p class="hint">
        注册后即可签发自己的 API 密钥；浏览模型与演练场无需登录。
      </p>
    </div>
  </section>
</template>

<style scoped>
.wrap {
  display: flex;
  justify-content: center;
  padding-top: 3rem;
}

.box {
  width: 100%;
  max-width: 380px;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.tabs {
  display: flex;
  border-bottom: 1px solid var(--border);
}

.tab {
  flex: 1;
  background: none;
  border: none;
  padding: 0.6rem;
  color: var(--muted);
  border-bottom: 2px solid transparent;
}

.tab.active {
  color: var(--primary);
  border-bottom-color: var(--primary);
  font-weight: 600;
}

.full {
  width: 100%;
}

.hint {
  color: var(--muted);
  font-size: 0.83rem;
  text-align: center;
  margin: 0;
}
</style>
