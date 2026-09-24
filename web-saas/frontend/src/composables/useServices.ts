/** 全局服务状态：模块级单例，顶栏与首页共享，15s 轮询。 */
import { reactive } from "vue";
import { api, type HealthInfo } from "../api/client";

export type BackendState = "loading" | "up" | "down";

export const services = reactive({
  backend: "loading" as BackendState,
  inferenceUp: false,
  modelCount: 0,
  docsAvailable: false,
  nodesOnline: 0,
  nodesTotal: 0,
  exp: {
    status: "not-configured" as "ok" | "unreachable" | "not-configured",
    url: "",
  },
});

let timer: ReturnType<typeof setInterval> | null = null;

export async function refreshServices(): Promise<void> {
  try {
    const h: HealthInfo = await api.health();
    services.backend = "up";
    services.inferenceUp = h.inferenceUp;
    services.modelCount = h.modelCount;
    services.docsAvailable = h.docsAvailable;
    services.nodesOnline = h.nodesOnline ?? 0;
    services.nodesTotal = h.nodesTotal ?? 0;
  } catch {
    services.backend = "down";
    services.inferenceUp = false;
  }
  try {
    const e = await api.expHealth();
    services.exp = e;
  } catch {
    services.exp = { status: "unreachable", url: "" };
  }
}

export function startServicePolling(): void {
  if (timer !== null) return;
  void refreshServices();
  timer = setInterval(() => void refreshServices(), 15000);
}
