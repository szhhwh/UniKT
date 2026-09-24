/** 当前登录用户状态：模块级单例。 */
import { computed, ref } from "vue";
import { api, type AuthInfo } from "../api/client";

const me = ref<AuthInfo | null>(null);
const loaded = ref(false);

export async function initAuth(): Promise<void> {
  if (loaded.value) return;
  try {
    me.value = await api.auth.me();
  } catch {
    me.value = null;
  } finally {
    loaded.value = true;
  }
}

export async function login(username: string, password: string): Promise<AuthInfo> {
  const info = await api.auth.login(username, password);
  me.value = info;
  return info;
}

export async function register(username: string, password: string): Promise<AuthInfo> {
  const info = await api.auth.register(username, password);
  me.value = info;
  return info;
}

export async function logout(): Promise<void> {
  try {
    await api.auth.logout();
  } finally {
    me.value = null;
  }
}

export const auth = {
  me: me,
  loaded,
  isLoggedIn: computed(() => me.value !== null),
  isAdmin: computed(() => me.value?.role === "ADMIN"),
};
