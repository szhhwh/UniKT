<script setup lang="ts">
import { RouterLink } from "vue-router";
import { services } from "../composables/useServices";
</script>

<template>
  <section>
    <div class="hero card">
      <h1>知识追踪，从训练到预测一站完成</h1>
      <p class="sub">
        基于 UniKT 的在线学习能力评估服务：文档查阅、模型训练（实验中心）、
        实时掌握度推理，统一入口。
      </p>
      <div class="actions">
        <RouterLink to="/playground"><button class="btn">进入演练场</button></RouterLink>
        <RouterLink to="/models"><button class="btn ghost">浏览模型</button></RouterLink>
      </div>
    </div>

    <h2>服务状态</h2>
    <div class="grid">
      <div class="card stat">
        <span class="dot" :class="services.backend === 'up' ? 'on' : 'off'"></span>
        <div class="stat-body">
          <span class="stat-title">业务后端</span>
          <span class="stat-sub">{{
            services.backend === "up" ? "SpringBoot · 运行中" : "不可达"
          }}</span>
        </div>
      </div>
      <div class="card stat">
        <span class="dot" :class="services.inferenceUp ? 'on' : 'off'"></span>
        <div class="stat-body">
          <span class="stat-title">推理服务</span>
          <span class="stat-sub">{{
            services.inferenceUp
              ? `在线 · ${services.modelCount} 个模型`
              : "离线（WSL: systemctl start unikt-inference）"
          }}</span>
        </div>
      </div>
      <RouterLink to="/docs" class="card stat link">
        <span class="dot" :class="services.docsAvailable ? 'on' : 'off'"></span>
        <div class="stat-body">
          <span class="stat-title">文档中心</span>
          <span class="stat-sub">{{
            services.docsAvailable ? "Sphinx · 已就绪" : "未构建"
          }}</span>
        </div>
      </RouterLink>
      <a
        v-if="services.exp.url"
        :href="services.exp.url"
        target="_blank"
        rel="noopener"
        class="card stat link"
      >
        <span class="dot" :class="services.exp.status === 'ok' ? 'on' : 'off'"></span>
        <div class="stat-body">
          <span class="stat-title">实验中心 ↗</span>
          <span class="stat-sub">{{
            services.exp.status === "ok" ? "运行中 · 训练任务/监控" : "未启动"
          }}</span>
        </div>
      </a>
      <div v-else class="card stat">
        <span class="dot off"></span>
        <div class="stat-body">
          <span class="stat-title">实验中心</span>
          <span class="stat-sub">未配置（unikt.exp-base-url）</span>
        </div>
      </div>
    </div>

    <h2>架构</h2>
    <div class="card arch">
      <div class="flow">
        <span class="node">浏览器 / Vite 前端<small>:5174</small></span>
        <span class="arrow">→</span>
        <span class="node">SpringBoot 业务后端<small>:8080 · 鉴权/编排</small></span>
        <span class="arrow">→</span>
        <span class="node">Python 推理服务<small>:8100 · WSL/GPU</small></span>
        <span class="arrow">→</span>
        <span class="node">UniKT 模型<small>60+ · checkpoint</small></span>
      </div>
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
  font-size: 1.75rem;
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

.arch .flow {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.7rem;
}

.node {
  background: #f8fafc;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 0.45rem 0.8rem;
  font-size: 0.88rem;
  display: inline-flex;
  flex-direction: column;
}

.node small {
  color: var(--muted);
  font-size: 0.75rem;
}

.arrow {
  color: var(--muted);
}
</style>
