package cn.ouc.luminatrail.unikt.config;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.ResourceHandlerRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

/**
 * Web MVC 配置：开发期 CORS（Vite 5174）与 Sphinx 文档静态托管
 * （/docs-static/**）。挂载前缀与 SPA 的 /docs 路由错开：门户页面由
 * 前端渲染，iframe 从 /docs-static/ 取 Sphinx 内容。文档目录由
 * ``unikt.docs-dir`` 指定（Sphinx 构建产物）；目录不存在时托管静默跳过。
 */
@Configuration
public class WebConfig implements WebMvcConfigurer {

    private final InferenceProperties props;

    public WebConfig(InferenceProperties props) {
        this.props = props;
    }

    @Override
    public void addCorsMappings(CorsRegistry registry) {
        registry.addMapping("/api/**")
                .allowedOriginPatterns("http://localhost:5174", "http://127.0.0.1:5174")
                .allowedMethods("GET", "POST", "PUT", "DELETE")
                .allowedHeaders("*");
    }

    @Override
    public void addResourceHandlers(ResourceHandlerRegistry registry) {
        String docsDir = props.docsDir();
        if (docsDir != null && !docsDir.isBlank()) {
            Path dir = Path.of(docsDir).toAbsolutePath().normalize();
            if (Files.isDirectory(dir)) {
                registry.addResourceHandler("/docs-static/**")
                        .addResourceLocations(dir.toUri() + "/")
                        .setCachePeriod(600);
            }
        }
        // 生产模式：托管前端构建产物（单端口部署）。SPA 兜底放在资源层
        // （PathResourceResolver），不能用 /** 的 Controller——控制器优先
        // 级高于资源处理器，会把 /docs-static、静态资产一起劫持掉。
        String frontendDir = props.frontendDir();
        if (frontendDir != null && !frontendDir.isBlank()) {
            Path dir = Path.of(frontendDir).toAbsolutePath().normalize();
            if (Files.isDirectory(dir)) {
                registry.addResourceHandler("/**")
                        .addResourceLocations(dir.toUri().toString())
                        .setCachePeriod(600)
                        .resourceChain(true)
                        .addResolver(new org.springframework.web.servlet.resource.PathResourceResolver() {
                            @Override
                            protected org.springframework.core.io.Resource getResource(
                                    String resourcePath,
                                    org.springframework.core.io.Resource location)
                                    throws IOException {
                                if (resourcePath.isEmpty()) {
                                    // "/" 根路径：直接回 index.html（createRelative("")
                                    // 会拿到目录本身，目录不可作为响应体）
                                    return location.createRelative("index.html");
                                }
                                org.springframework.core.io.Resource requested =
                                        location.createRelative(resourcePath);
                                if (requested.exists() && requested.isReadable()
                                        && !requested.getFile().isDirectory()) {
                                    return requested;
                                }
                                // SPA history 路由兜底：非接口/文档/带扩展名的
                                // 路径回 index.html（前端路由器接管）
                                boolean spa = (resourcePath.isEmpty()
                                        || !resourcePath.contains("."))
                                        && !resourcePath.startsWith("api/")
                                        && !resourcePath.startsWith("docs-static/");
                                if (spa) {
                                    return location.createRelative("index.html");
                                }
                                return null; // 404
                            }
                        });
            }
        }
    }

    /** 文档目录是否就绪（首页状态卡用）：以 index.html 存在为准。 */
    public boolean docsAvailable() {
        String docsDir = props.docsDir();
        if (docsDir == null || docsDir.isBlank()) {
            return false;
        }
        return Files.isRegularFile(Path.of(docsDir).toAbsolutePath()
                .normalize().resolve("index.html"));
    }
}
