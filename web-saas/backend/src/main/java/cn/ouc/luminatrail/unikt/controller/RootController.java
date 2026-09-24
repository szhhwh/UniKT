package cn.ouc.luminatrail.unikt.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

/** 根路径 "/" 直达 SPA 首页（资源层兜底覆盖不了空路径，实测 404）。 */
@RestController
public class RootController {

    @GetMapping("/")
    public String root() {
        return "forward:/index.html";
    }
}
