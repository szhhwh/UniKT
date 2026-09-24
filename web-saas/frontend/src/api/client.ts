/** SpringBoot 后端 API 客户端。开发期经 Vite 代理到 localhost:8080。 */

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

export interface AuthInfo {
  username: string;
  role: "USER" | "ADMIN";
}

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
  dailyCount: number;
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
  const res = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (res.status === 204) {
    return undefined as T;
  }
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const message = body?.message || body?.detail || `请求失败：${res.status}`;
    if (res.status === 401 && !path.startsWith("/auth/")) {
      // 会话失效/被踢（禁用、降权、过期）：清身份并跳登录页
      window.dispatchEvent(new CustomEvent("unikt:unauthorized", { detail: message }));
    }
    throw new ApiError(res.status, message);
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
    list: (scope?: "all") =>
      request<KeyInfo[]>(scope === "all" ? "/keys?scope=all" : "/keys"),
    create: (name: string) =>
      request<CreatedKey>("/keys", { method: "POST", body: JSON.stringify({ name }) }),
    revoke: (id: number) => request<void>(`/keys/${id}`, { method: "DELETE" }),
  },
  auth: {
    register: (username: string, password: string) =>
      request<AuthInfo>("/auth/register", {
        method: "POST",
        body: JSON.stringify({ username, password }),
      }),
    login: (username: string, password: string) =>
      request<AuthInfo>("/auth/login", {
        method: "POST",
        body: JSON.stringify({ username, password }),
      }),
    logout: () => request<{ message: string }>("/auth/logout", { method: "POST" }),
    me: () => request<AuthInfo>("/auth/me"),
  },
  admin: {
    users: () => request<AdminUserInfo[]>("/admin/users"),
    updateUser: (id: number, patch: { role?: string; active?: boolean }) =>
      request<void>(`/admin/users/${id}`, {
        method: "PATCH",
        body: JSON.stringify(patch),
      }),
  },
};

export interface AdminUserInfo {
  id: number;
  username: string;
  role: "USER" | "ADMIN";
  active: boolean;
  keyCount: number;
  createdAt: string;
}
