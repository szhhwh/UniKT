import { createRouter, createWebHistory } from "vue-router";

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", name: "home", component: () => import("../views/HomeView.vue") },
    { path: "/models", name: "models", component: () => import("../views/ModelsView.vue") },
    {
      path: "/nodes",
      name: "nodes",
      component: () => import("../views/NodesView.vue"),
    },
    { path: "/keys", name: "keys", component: () => import("../views/KeysView.vue") },
    {
      path: "/playground",
      name: "playground",
      component: () => import("../views/PlaygroundView.vue"),
    },
    { path: "/docs", name: "docs", component: () => import("../views/DocsView.vue") },
    { path: "/:pathMatch(.*)*", redirect: "/" },
  ],
});

export default router;
