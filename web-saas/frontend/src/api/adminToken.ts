/** 管理令牌：门户对节点/密钥做改动操作时随请求附带（存 localStorage）。 */
const KEY = "uniktAdminToken";

export function getAdminToken(): string {
  return localStorage.getItem(KEY) ?? "";
}

export function setAdminToken(token: string): void {
  if (token) {
    localStorage.setItem(KEY, token);
  } else {
    localStorage.removeItem(KEY);
  }
}
