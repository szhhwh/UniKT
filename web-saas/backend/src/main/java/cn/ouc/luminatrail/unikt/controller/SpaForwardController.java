package cn.ouc.luminatrail.unikt.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * SPA 路由兜底：前端路由（history 模式）直链/刷新时回 index.html，
 * 由前端路由器接管。仅生产托管模式有意义（dev 走 Vite）。
 */
@RestController
public class SpaForwardController {

    @GetMapping({
            "/", "/login", "/models", "/playground", "/nodes", "/keys",
            "/docs", "/admin/users"
    })
    public String spa() {
        return "forward:/index.html";
    }
}
