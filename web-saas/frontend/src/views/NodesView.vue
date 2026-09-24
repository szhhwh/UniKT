<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from "vue";
import { api, type NodeInfo } from "../api/client";

const nodes = ref<NodeInfo[]>([]);
const error = ref("");
const loading = ref(true);
const expandedId = ref<number | null>(null);
const formOpen = ref(false);
const creating = ref(false);
const formError = ref("");

const form = ref({ name: "", sshTarget: "", baseUrl: "", repoPath: "/root/unikt" });

let timer: ReturnType<typeof setInterval> | null = null;
let inflight: Promise<void> | null = null;
let dirty = false;

async function refresh(): Promise<void> {
  // 在途保护：请求未返回时标记 dirty，返回后补发一轮（增删后不丢显式刷新）
  if (inflight) {
    dirty = true;
    return inflight;
  }
  inflight = doRefresh();
  try {
    await inflight;
  } finally {
    inflight = null;
    if (dirty) {
      dirty = false;
      void refresh();
    }
  }
}

async function doRefresh(): Promise<void> {
  try {
    nodes.value = await api.nodes.list();
    error.value = "";
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
  } finally {
    loading.value = false;
  }
}

onMounted(async () => {
  await refresh();
  timer = setInterval(() => void refresh(), 5000);
});
onUnmounted(() => {
  if (timer !== null) clearInterval(timer);
});

const statusMeta: Record<string, { cls: string; text: string }> = {
  NEW: { cls: "muted", text: "未部署" },
  DEPLOYING: { cls: "warn", text: "部署中…" },
  ONLINE: { cls: "ok", text: "在线" },
  OFFLINE: { cls: "muted", text: "离线" },
  FAILED: { cls: "fail", text: "失败" },
};

/** 与后端 @Pattern 对齐的提交前校验（含 trim）。 */
const formOk = computed(() => {
  const f = form.value;
  return (
    f.name.trim().length > 0 &&
    f.name.trim().length <= 40 &&
    /^[A-Za-z0-9][A-Za-z0-9@._-]*$/.test(f.sshTarget.trim()) &&
    /^https?:\/\/[A-Za-z0-9.:/-]+$/.test(f.baseUrl.trim().replace(/\/+$/, "")) &&
    /^\/[A-Za-z0-9._/-]*$/.test(f.repoPath.trim())
  );
});

/** 节点卡片级临时提示（不被轮询清除）。 */
const cardMsg = ref<Record<number, string>>({});

async function addNode(): Promise<void> {
  formError.value = "";
  creating.value = true;
  try {
    const f = form.value;
    // 创建成功即关闭表单并刷新：部署失败的信息挂到节点卡片上，
    // 避免用户重试时重复创建同名节点
    const created = await api.nodes.create({
      name: f.name.trim(),
      sshTarget: f.sshTarget.trim(),
      baseUrl: f.baseUrl.trim().replace(/\/+$/, ""),
      repoPath: f.repoPath.trim(),
    });
    formOpen.value = false;
    form.value = { name: "", sshTarget: "", baseUrl: "", repoPath: "/root/unikt" };
    await refresh();
    try {
      await api.nodes.deploy(created.id);
      expandedId.value = created.id;
    } catch (e) {
      cardMsg.value = {
        ...cardMsg.value,
        [created.id]: e instanceof Error ? e.message : String(e),
      };
    }
  } catch (e) {
    formError.value = e instanceof Error ? e.message : String(e);
  } finally {
    creating.value = false;
  }
}

async function deploy(n: NodeInfo): Promise<void> {
  // 清掉上一轮的卡片级错误提示
  const rest = { ...cardMsg.value };
  delete rest[n.id];
  cardMsg.value = rest;
  try {
    await api.nodes.deploy(n.id);
    // 乐观置为部署中：后端线程池提交到首次落库之间有短暂空窗
    n.status = "DEPLOYING";
    expandedId.value = n.id;
  } catch (e) {
    cardMsg.value = { ...cardMsg.value, [n.id]: e instanceof Error ? e.message : String(e) };
  }
}

async function remove(n: NodeInfo): Promise<void> {
  if (!window.confirm(`删除节点「${n.name}」？（远端已部署的服务不会回收）`)) {
    return;
  }
  try {
    await api.nodes.remove(n.id);
    await refresh();
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
  }
}

function toggleLog(n: NodeInfo): void {
  expandedId.value = expandedId.value === n.id ? null : n.id;
}
</script>

<template>
  <section>
    <div class="head">
      <h1>计算节点</h1>
      <div class="head-right">
        <button class="btn small" @click="formOpen = !formOpen">
          {{ formOpen ? "收起" : "+ 添加节点" }}
        </button>
      </div>
    </div>
    <p class="sub">
      注册一台 SSH 可达的 Linux 服务器，点「部署」自动完成：环境安装 → 代码同步 →
      systemd 服务 → 上线为推理节点。页面在本机，计算在节点上。
    </p>

    <div v-if="formOpen" class="card form-card">
      <div class="form-grid">
        <div class="field">
          <label for="n-name">名称</label>
          <input id="n-name" v-model="form.name" maxlength="40" placeholder="wsl-4060" />
        </div>
        <div class="field">
          <label for="n-ssh">SSH 目标（~/.ssh/config 别名或 user@host）</label>
          <input
            id="n-ssh"
            v-model="form.sshTarget"
            maxlength="255"
            pattern="[A-Za-z0-9][A-Za-z0-9@._-]*"
            placeholder="wsl-root"
          />
        </div>
        <div class="field">
          <label for="n-url">推理服务地址（http://IP:8100）</label>
          <input
            id="n-url"
            v-model="form.baseUrl"
            maxlength="255"
            pattern="https?://[A-Za-z0-9.:/-]+"
            placeholder="http://100.73.139.45:8100"
          />
        </div>
        <div class="field">
          <label for="n-repo">远端仓库路径（绝对路径）</label>
          <input
            id="n-repo"
            v-model="form.repoPath"
            maxlength="255"
            pattern="/[A-Za-z0-9._/-]*"
            placeholder="/root/unikt"
          />
        </div>
      </div>
      <p v-if="formError" class="banner-error">{{ formError }}</p>
      <button class="btn" :disabled="creating || !formOk" @click="addNode">
        {{ creating ? "保存中…" : "保存并部署" }}
      </button>
      <p class="hint">
        前提：本机可免密 SSH 到目标机（root 登录，ssh 别名已配好），且节点 IP 可直连。
      </p>
    </div>

    <p v-if="error" class="banner-error">{{ error }}</p>
    <p v-else-if="loading">加载中…</p>
    <p v-else-if="nodes.length === 0" class="empty">
      还没有节点。添加一台服务器，或先在「演练场」用默认推理服务体验。
    </p>

    <div v-for="n in nodes" :key="n.id" class="card node-card">
      <div class="row">
        <div class="ident">
          <span class="name">{{ n.name }}</span>
          <span
            class="badge"
            :class="(statusMeta[n.status] ?? { cls: 'muted' }).cls"
          >
            {{ (statusMeta[n.status] ?? { text: n.status }).text }}
          </span>
          <span v-if="n.status === 'ONLINE'" class="meta">{{ n.modelsCount }} 个模型</span>
        </div>
        <div class="actions">
          <button
            class="btn small"
            :disabled="n.status === 'DEPLOYING'"
            @click="deploy(n)"
          >
            {{ n.status === "DEPLOYING" ? "部署中…" : n.status === "ONLINE" ? "重新部署" : "部署" }}
          </button>
          <button v-if="n.deployLog" class="btn ghost small" @click="toggleLog(n)">
            {{ expandedId === n.id ? "收起日志" : "日志" }}
          </button>
          <button class="btn ghost small danger" @click="remove(n)">删除</button>
        </div>
      </div>
      <div class="meta-line">
        <span>SSH：{{ n.sshTarget }}</span>
        <span>地址：{{ n.baseUrl }}</span>
        <span>路径：{{ n.repoPath }}</span>
      </div>
      <p v-if="n.lastError" class="banner-error">{{ n.lastError }}</p>
      <p v-if="cardMsg[n.id]" class="banner-error">{{ cardMsg[n.id] }}</p>
      <pre v-if="expandedId === n.id && n.deployLog" class="log">{{ n.deployLog }}</pre>
    </div>
  </section>
</template>

<style scoped>
.head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
  flex-wrap: wrap;
}

.head-right {
  display: flex;
  align-items: center;
  gap: 1.2rem;
  flex-wrap: wrap;
}

.sub {
  color: var(--muted);
  max-width: 44rem;
}

.form-card {
  margin: 1rem 0 1.4rem;
}

.form-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
  gap: 1rem;
  margin-bottom: 1rem;
}

.hint {
  margin-top: 0.7rem;
  color: var(--muted);
  font-size: 0.82rem;
}

.node-card {
  margin-bottom: 0.9rem;
}

.row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.8rem;
  flex-wrap: wrap;
}

.ident {
  display: flex;
  align-items: center;
  gap: 0.7rem;
  flex-wrap: wrap;
}

.name {
  font-weight: 650;
}

.meta {
  color: var(--muted);
  font-size: 0.85rem;
}

.actions {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.meta-line {
  margin-top: 0.5rem;
  display: flex;
  gap: 1.4rem;
  flex-wrap: wrap;
  color: var(--muted);
  font-size: 0.83rem;
  font-variant-numeric: tabular-nums;
}

.badge.fail {
  background: var(--danger-weak);
  color: var(--danger);
}

.log {
  margin: 0.8rem 0 0;
  background: #0f172a;
  color: #cbd5e1;
  border-radius: var(--radius-sm);
  padding: 0.9rem 1rem;
  font-size: 0.78rem;
  line-height: 1.5;
  overflow-x: auto;
  white-space: pre-wrap;
  max-height: 320px;
}

.btn.danger:hover {
  border-color: var(--danger);
  color: var(--danger);
  background: transparent;
}

.empty {
  color: var(--muted);
}
</style>
