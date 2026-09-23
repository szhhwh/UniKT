import vue from "@vitejs/plugin-vue";
import { defineConfig } from "vite";

// 开发期把 /api 代理到 SpringBoot 后端（8080），避免跨域
export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5174,
    proxy: {
      "/api": {
        target: "http://localhost:8080",
        changeOrigin: true,
      },
    },
  },
});
