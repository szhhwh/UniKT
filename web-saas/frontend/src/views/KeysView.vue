<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api, type KeyInfo } from "../api/client";
import AdminTokenInput from "../components/AdminTokenInput.vue";

const keys = ref<KeyInfo[]>([]);
/** 列表加载失败（替换表格）。 */
const loadError = ref("");
/** 操作失败（表格上方横幅，不清空表格）。 */
const opError = ref("");
const loading = ref(true);
const name = ref("");
const creating = ref(false);
/** 新建成功后的明文 key（仅展示一次）。 */
const freshKey = ref<string | null>(null);

onMounted(async () => {
  try {
    keys.value = await api.keys.list();
  } catch (e) {
    loadError.value = e instanceof Error ? e.message : String(e);
  } finally {
    loading.value = false;
  }
});

async function create(): Promise<void> {
  if (freshKey.value && !window.confirm("上一枚密钥还未保存，继续生成将无法再看到它。继续？")) {
    return;
  }
  opError.value = "";
  creating.value = true;
  try {
    const created = await api.keys.create(name.value.trim());
    freshKey.value = created.key;
    name.value = "";
    keys.value = await api.keys.list();
  } catch (e) {
    opError.value = e instanceof Error ? e.message : String(e);
  } finally {
    creating.value = false;
  }
}

async function revoke(k: KeyInfo): Promise<void> {
  if (!window.confirm(`吊销密钥「${k.name}」？使用它的调用方将立即失效。`)) {
    return;
  }
  try {
    await api.keys.revoke(k.id);
    keys.value = await api.keys.list();
  } catch (e) {
    opError.value = e instanceof Error ? e.message : String(e);
  }
}

/** 复制到剪贴板；非安全上下文（http 内网）退回 execCommand，再失败提示手动复制。 */
async function copyKey(): Promise<void> {
  if (!freshKey.value) return;
  try {
    await navigator.clipboard.writeText(freshKey.value);
    copied.value = true;
  } catch {
    try {
      const ta = document.createElement("textarea");
      ta.value = freshKey.value;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
      copied.value = true;
    } catch {
      window.alert("复制失败，请手动选中文本复制。");
    }
  }
  setTimeout(() => (copied.value = false), 1500);
}

const copied = ref(false);

function fmtTime(iso: string | null): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleString();
}
</script>

<template>
  <section>
    <div class="head">
      <h1>API 密钥</h1>
      <AdminTokenInput />
    </div>
    <p class="sub">
      为你的应用（刷题系统、教学平台…）签发密钥，通过开放接口调用知识追踪推理。
    </p>

    <div class="card create-card">
      <div class="create-row">
        <div class="field grow">
          <label for="k-name">调用方名称</label>
          <input
            id="k-name"
            v-model="name"
            maxlength="60"
            placeholder="例如：小程序后端 / 毕设演示"
            @keyup.enter="create"
          />
        </div>
        <button class="btn" :disabled="creating || !name.trim()" @click="create">
          {{ creating ? "生成中…" : "生成密钥" }}
        </button>
      </div>
    </div>

    <div v-if="freshKey" class="card fresh">
      <p><strong>密钥已生成（仅此一次可见，请立即保存）：</strong></p>
      <div class="key-row">
        <code class="key">{{ freshKey }}</code>
        <button class="btn ghost small" @click="copyKey">
          {{ copied ? "已复制 ✓" : "复制" }}
        </button>
        <button class="btn ghost small" @click="freshKey = null">我已保存</button>
      </div>
    </div>

    <p v-if="opError" class="banner-error">{{ opError }}</p>
    <p v-else-if="loadError" class="banner-error">加载失败：{{ loadError }}</p>
    <p v-else-if="loading">加载中…</p>

    <template v-else>
      <p v-if="keys.length === 0" class="empty">还没有密钥。</p>
      <div v-else class="card">
        <table class="table">
          <thead>
            <tr>
              <th>名称</th>
              <th>密钥前缀</th>
              <th>状态</th>
              <th>调用次数</th>
              <th>最近使用</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="k in keys" :key="k.id">
              <td>{{ k.name }}</td>
              <td><code>{{ k.keyPrefix }}</code></td>
              <td>
                <span :class="k.active ? 'badge ok' : 'badge muted'">
                  {{ k.active ? "启用" : "已吊销" }}
                </span>
              </td>
              <td class="num">{{ k.requestCount }}</td>
              <td class="num">{{ fmtTime(k.lastUsedAt) }}</td>
              <td>
                <button
                  v-if="k.active"
                  class="btn ghost small danger"
                  @click="revoke(k)"
                >
                  吊销
                </button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </template>

    <h2>调用方式</h2>
    <div class="card">
      <pre class="example">curl -X POST http://&lt;服务地址&gt;:8080/api/v1/predict \
  -H "X-API-Key: unikt_你的密钥" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "DKT",
    "questions": [1, 2, 3],
    "skills":   [5, 5, 12],
    "responses":[1, 0, 1]
  }'

# 查看可用模型
curl -H "X-API-Key: unikt_你的密钥" http://&lt;服务地址&gt;:8080/api/v1/models</pre>
      <p class="hint">
        密钥仅存 SHA-256 哈希，泄露后在此页吊销重发即可；无效密钥返回 401。
      </p>
    </div>
  </section>
</template>

<style scoped>
.sub {
  color: var(--muted);
  max-width: 40rem;
}

.head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
  flex-wrap: wrap;
}

.create-card {
  margin: 1rem 0;
}

.create-row {
  display: flex;
  gap: 1rem;
  align-items: flex-end;
  flex-wrap: wrap;
}

.grow {
  flex: 1;
  min-width: 240px;
}

.fresh {
  border-color: #bbf7d0;
  background: #f0fdf4;
}

.key-row {
  display: flex;
  align-items: center;
  gap: 0.7rem;
  flex-wrap: wrap;
}

.key {
  background: #0f172a;
  color: #e2e8f0;
  padding: 0.45rem 0.8rem;
  border-radius: 6px;
  font-size: 0.88rem;
  word-break: break-all;
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

.example {
  background: #0f172a;
  color: #cbd5e1;
  border-radius: var(--radius-sm);
  padding: 1rem 1.1rem;
  font-size: 0.8rem;
  line-height: 1.6;
  overflow-x: auto;
}

.hint {
  color: var(--muted);
  font-size: 0.83rem;
  margin-top: 0.7rem;
}

.empty {
  color: var(--muted);
}

.btn.danger:hover {
  border-color: var(--danger);
  color: var(--danger);
  background: transparent;
}
</style>
