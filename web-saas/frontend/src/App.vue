<script setup lang="ts">
import { RouterLink, RouterView } from "vue-router";
</script>

<template>
  <div class="layout">
    <header class="topbar">
      <span class="brand">UniKT SaaS</span>
      <nav>
        <RouterLink to="/">首页</RouterLink>
        <RouterLink to="/models">模型</RouterLink>
        <RouterLink to="/playground">演练场</RouterLink>
      </nav>
    </header>
    <main class="content">
      <RouterView />
    </main>
  </div>
</template>

<style scoped>
.topbar {
  display: flex;
  align-items: center;
  gap: 2rem;
  padding: 0.75rem 1.5rem;
  border-bottom: 1px solid #e2e8f0;
}
.brand {
  font-weight: 700;
}
.topbar nav {
  display: flex;
  gap: 1rem;
}
.topbar a {
  color: #475569;
  text-decoration: none;
}
.topbar a.router-active {
  color: #2563eb;
  font-weight: 600;
}
</style>
