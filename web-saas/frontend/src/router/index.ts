import { createRouter, createWebHistory } from "vue-router";

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", name: "home", component: () => import("../views/HomeView.vue") },
    { path: "/models", name: "models", component: () => import("../views/ModelsView.vue") },
    {
      path: "/playground",
      name: "playground",
      component: () => import("../views/PlaygroundView.vue"),
    },
    { path: "/nodes", name: "nodes", component: () => import("../views/NodesView.vue") },
    { path: "/keys", name: "keys", component: () => import("../views/KeysView.vue") },
    { path: "/data", name: "data", component: () => import("../views/DatasetsView.vue") },
    {
      path: "/training",
      name: "training",
      component: () => import("../views/TrainingView.vue"),
    },
    { path: "/docs", name: "docs", component: () => import("../views/DocsView.vue") },
    { path: "/login", name: "login", component: () => import("../views/LoginView.vue") },
    {
      path: "/admin/users",
      name: "admin-users",
      component: () => import("../views/AdminUsersView.vue"),
    },
    { path: "/:pathMatch(.*)*", redirect: "/" },
  ],
});

router.beforeEach(async (to) => {
  const { initAuth, auth } = await import("../composables/useAuth");
  await initAuth();
  if (
    ["keys", "admin-users", "data", "training"].includes(String(to.name))
    && !auth.isLoggedIn.value
  ) {
    return { name: "login", query: { next: to.fullPath } };
  }
  if ((to.name === "nodes" || to.name === "admin-users") && !auth.isAdmin.value) {
    return { name: "home" };
  }
  return true;
});

export default router;
