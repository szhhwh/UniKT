<script setup lang="ts">
import { computed } from "vue";
import { RouterLink } from "vue-router";
import { services } from "../composables/useServices";

/** 推理可用 = 默认服务在线 或 任一计算节点在线（与顶栏口径一致）。 */
const inferenceOk = computed(
  () => services.inferenceUp || services.nodesOnline > 0,
);
const inferenceText = computed(() =>
  services.nodesOnline > 0
    ? `在线 · ${services.nodesOnline} 个计算节点`
    : services.inferenceUp
      ? "在线"
      : "暂不可用",
);
</script>

<template>
  <section>
    <div class="hero card">
      <h1>知识追踪推理服务</h1>
      <p class="sub">
        选一个训练好的模型，输入学生的作答记录，实时得到「下一题答对概率」
        与各知识点的掌握度。
      </p>
      <div class="actions">
        <RouterLink to="/playground"><button class="btn">进入演练场</button></RouterLink>
        <RouterLink to="/models"><button class="btn ghost">浏览可用模型</button></RouterLink>
      </div>
    </div>

    <h2>服务状态</h2>
    <div class="grid">
      <div class="card stat">
        <span class="dot" :class="services.backend === 'up' ? 'on' : 'off'"></span>
        <div class="stat-body">
          <span class="stat-title">门户服务</span>
          <span class="stat-sub">{{
            services.backend === "up" ? "运行中" : "不可用，请稍后再试"
          }}</span>
        </div>
      </div>
      <div class="card stat">
        <span class="dot" :class="inferenceOk ? 'on' : 'off'"></span>
        <div class="stat-body">
          <span class="stat-title">模型推理</span>
          <span class="stat-sub">{{ inferenceOk ? inferenceText : "暂不可用" }}</span>
        </div>
      </div>
      <RouterLink to="/docs" class="card stat link">
        <span class="dot" :class="services.docsAvailable ? 'on' : 'off'"></span>
        <div class="stat-body">
          <span class="stat-title">使用文档</span>
          <span class="stat-sub">{{ services.docsAvailable ? "可查阅" : "未构建" }}</span>
        </div>
      </RouterLink>
      <RouterLink to="/nodes" class="card stat link">
        <span class="dot" :class="services.nodesOnline > 0 ? 'on' : 'off'"></span>
        <div class="stat-body">
          <span class="stat-title">计算节点（管理）</span>
          <span class="stat-sub">在线 {{ services.nodesOnline }} / {{ services.nodesTotal }}</span>
        </div>
      </RouterLink>
    </div>
  </section>
</template>

<style scoped>
.hero {
  padding: 2.2rem 2rem;
  background:
    radial-gradient(1200px 300px at 20% -60%, var(--primary-weak), transparent),
    var(--surface);
}

.hero h1 {
  font-size: 1.7rem;
}

.sub {
  color: var(--muted);
  max-width: 40rem;
}

.actions {
  display: flex;
  gap: 0.8rem;
  margin-top: 1.2rem;
}

.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 0.9rem;
}

.stat {
  display: flex;
  align-items: center;
  gap: 0.8rem;
  color: inherit;
  text-decoration: none;
}

.stat.link:hover {
  border-color: var(--primary);
}

.stat-body {
  display: flex;
  flex-direction: column;
}

.stat-title {
  font-weight: 600;
}

.stat-sub {
  font-size: 0.82rem;
  color: var(--muted);
}
</style>
