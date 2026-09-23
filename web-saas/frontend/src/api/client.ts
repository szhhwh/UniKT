/** SpringBoot 后端 API 客户端。开发期经 Vite 代理到 localhost:8080。 */

export interface ModelInfo {
  name: string;
  available: boolean;
  numSkills: number | null;
}

export interface PredictRequest {
  model: string;
  questions: number[];
  skills: number[];
  responses: number[];
}

export interface PredictResponse {
  model: string;
  predictions: number[];
}

export interface HealthInfo {
  status: string;
  service: string;
  inferenceUp: boolean;
  modelCount: number;
  docsAvailable: boolean;
}

export interface ExpHealth {
  status: "ok" | "unreachable" | "not-configured";
  reachable: boolean;
  url: string;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ message: res.statusText }));
    throw new Error(body.message ?? body.detail ?? `请求失败：${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<HealthInfo>("/health"),
  models: () => request<ModelInfo[]>("/models"),
  expHealth: () => request<ExpHealth>("/exp/health"),
  predict: (payload: PredictRequest) =>
    request<PredictResponse>("/predict", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
};
