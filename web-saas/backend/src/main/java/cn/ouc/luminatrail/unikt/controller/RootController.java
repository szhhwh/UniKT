package cn.ouc.luminatrail.unikt.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;

/** 根路径 "/" 直达 SPA 首页（资源层兜底覆盖不了空路径，实测 404）。须为
 *  视图解析下的 @Controller——@RestController 会把 "forward:..." 当响应体原文输出。 */
@Controller
public class RootController {

    @GetMapping("/")
    public String root() {
        return "forward:/index.html";
    }
}
