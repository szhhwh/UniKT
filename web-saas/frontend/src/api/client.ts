/** SpringBoot 后端 API 客户端。开发期经 Vite 代理到 localhost:8080。 */
import { getAdminToken } from "./adminToken";

export interface ModelInfo {
  name: string;
  available: boolean;
  numSkills: number | null;
  node: string | null;
}

export interface HealthInfo {
  status: string;
  service: string;
  inferenceUp: boolean;
  modelCount: number;
  docsAvailable: boolean;
  nodesOnline?: number;
  nodesTotal?: number;
  adminTokenRequired?: boolean;
}

export interface NodeInfo {
  id: number;
  name: string;
  sshTarget: string;
  baseUrl: string;
  repoPath: string;
  status: "NEW" | "DEPLOYING" | "ONLINE" | "OFFLINE" | "FAILED";
  modelsCount: number;
  lastError: string | null;
  deployLog: string;
  createdAt: string;
  updatedAt: string;
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

export interface ExpHealth {
  status: "ok" | "unreachable" | "not-configured";
  reachable: boolean;
  url: string;
}

export interface KeyInfo {
  id: number;
  name: string;
  keyPrefix: string;
  active: boolean;
  lastUsedAt: string | null;
  requestCount: number;
  createdAt: string;
}

export interface CreatedKey {
  id: number;
  name: string;
  key: string;
}

export interface SkillInfo {
  id: number;
  name: string | null;
  questions: number;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const token = getAdminToken();
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (token) {
    headers["X-Admin-Token"] = token;
  }
  const res = await fetch(`/api${path}`, { headers, ...init });
  if (res.status === 204) {
    return undefined as T;
  }
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const message = body?.message || body?.detail || `请求失败：${res.status}`;
    throw new Error(message);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<HealthInfo>("/health"),
  models: () => request<ModelInfo[]>("/models"),
  skills: (model: string) => request<SkillInfo[]>(`/skills/${encodeURIComponent(model)}`),
  expHealth: () => request<ExpHealth>("/exp/health"),
  predict: (payload: PredictRequest) =>
    request<PredictResponse>("/predict", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  nodes: {
    list: () => request<NodeInfo[]>("/nodes"),
    create: (payload: { name: string; sshTarget: string; baseUrl: string; repoPath: string }) =>
      request<NodeInfo>("/nodes", { method: "POST", body: JSON.stringify(payload) }),
    deploy: (id: number) => request<{ message: string }>(`/nodes/${id}/deploy`, { method: "POST" }),
    remove: (id: number) => request<void>(`/nodes/${id}`, { method: "DELETE" }),
  },
  keys: {
    list: () => request<KeyInfo[]>("/keys"),
    create: (name: string) =>
      request<CreatedKey>("/keys", { method: "POST", body: JSON.stringify({ name }) }),
    revoke: (id: number) => request<void>(`/keys/${id}`, { method: "DELETE" }),
  },
};
