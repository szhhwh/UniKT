import vue from "@vitejs/plugin-vue";
import { defineConfig } from "vite";

// 开发期代理：/api 业务接口与 /docs-static（Sphinx 产物）指向 SpringBoot
export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5174,
    proxy: {
      "/api": {
        target: "http://localhost:8080",
        changeOrigin: true,
      },
      "/docs-static": {
        target: "http://localhost:8080",
        changeOrigin: true,
      },
    },
  },
});
