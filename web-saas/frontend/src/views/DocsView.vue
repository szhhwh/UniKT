<script setup lang="ts">
import { services } from "../composables/useServices";
</script>

<template>
  <section class="docs-wrap">
    <div v-if="!services.docsAvailable" class="card empty">
      <h1>文档中心</h1>
      <p>本地文档尚未构建。两种方式查看：</p>
      <ol class="steps">
        <li>
          在仓库根执行
          <code>pixi run -e docs sphinx-build -b html docs/source web-saas/docs-build</code
          >，随后重启 SpringBoot 后端
        </li>
        <li>
          或访问线上版
          <a href="https://unikt.readthedocs.io/en/latest/" target="_blank" rel="noopener"
            >unikt.readthedocs.io ↗</a
          >
        </li>
      </ol>
    </div>
    <iframe
      v-else
      src="/docs-static/index.html"
      title="UniKT 文档"
      class="docs-frame"
    ></iframe>
  </section>
</template>

<style scoped>
.docs-wrap {
  margin: -0.4rem 0 0;
}

.docs-frame {
  width: 100%;
  height: calc(100vh - 140px);
  min-height: 480px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: #fff;
}

.empty .steps {
  color: var(--muted);
  line-height: 2;
}

.empty code {
  background: #f1f5f9;
  padding: 0.1rem 0.4rem;
  border-radius: 4px;
  font-size: 0.85em;
}
</style>
