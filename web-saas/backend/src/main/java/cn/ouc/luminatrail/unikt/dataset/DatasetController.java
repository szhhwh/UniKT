package cn.ouc.luminatrail.unikt.dataset;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import java.io.IOException;
import java.io.InputStream;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import cn.ouc.luminatrail.unikt.user.PortalPrincipal;

/** 数据集管理：上传（校验+统计+暂存）→ P3 训练用；用户只见自己的。 */
@RestController
@RequestMapping("/api/datasets")
@org.springframework.validation.annotation.Validated
public class DatasetController {

    public record DatasetDto(Long id, String name, String slug, String status,
                             long interactions, long users, long questions, long skills,
                             boolean hasSkillNames, String lastError, Instant createdAt) {

        static DatasetDto from(UserDataset d) {
            return new DatasetDto(d.getId(), d.getName(), d.getSlug(), d.getStatus(),
                    d.getInteractions(), d.getUsers(), d.getQuestions(), d.getSkills(),
                    d.isHasSkillNames(), d.getLastError(), d.getCreatedAt());
        }
    }

    private final DatasetRepository datasets;
    private final DatasetIngestService ingest;
    private final cn.ouc.luminatrail.unikt.training.JobRepository jobs;

    public DatasetController(DatasetRepository datasets, DatasetIngestService ingest,
                             cn.ouc.luminatrail.unikt.training.JobRepository jobs) {
        this.datasets = datasets;
        this.ingest = ingest;
        this.jobs = jobs;
    }

    @GetMapping
    public List<DatasetDto> list(@RequestParam(required = false) String scope) {
        if (isAdmin() && "all".equals(scope)) {
            return datasets.findAll().stream()
                    .sorted((a, b) -> Long.compare(b.getId(), a.getId()))
                    .map(DatasetDto::from)
                    .toList();
        }
        return datasets.findByOwnerIdOrderByIdDesc(currentUserId()).stream()
                .map(DatasetDto::from)
                .toList();
    }

    /** 上传：interactions 必需，skills（名称表）可选。校验失败返回 400 人话错误。 */
    @PostMapping
    public ResponseEntity<?> create(@RequestParam @NotBlank @Size(max = 60) String name,
                                    @RequestParam MultipartFile interactions,
                                    @RequestParam(required = false) MultipartFile skills)
            throws IOException {
        if (interactions == null || interactions.isEmpty()) {
            return ResponseEntity.badRequest().body(Map.of("message", "请选择 interactions.csv 文件"));
        }
        Long ownerId = currentUserId();
        UserDataset d = new UserDataset();
        d.setOwnerId(ownerId);
        d.setName(name.trim());
        d.setSlug(ingest.slugFor(name.trim(), datasets));

        // 临时目录用 UUID：并发同名上传不再互相覆盖/误删（slug 落库前无唯一性保证）
        java.nio.file.Path tmp = java.nio.file.Path.of("data", "uploads",
                "tmp-" + java.util.UUID.randomUUID());
        try (InputStream in = interactions.getInputStream()) {
            DatasetIngestService.Stats stats =
                    ingest.validateAndStoreInteractions(in, tmp.resolve("interactions.csv"));
            if (skills != null && !skills.isEmpty()) {
                try (InputStream sin = skills.getInputStream()) {
                    ingest.validateAndStoreSkills(sin, tmp.resolve("skills.csv"));
                    d.setHasSkillNames(true);
                }
            }
            d.setInteractions(stats.interactions());
            d.setUsers(stats.users());
            d.setQuestions(stats.questions());
            d.setSkills(stats.skills());
            d.setStatus(UserDataset.READY);
            // 先落库拿 id，再把暂存挪到正式目录；挪盘失败则删除记录
            // （不留"就绪"但底下没有文件的空壳）
            UserDataset saved = datasets.save(d);
            java.nio.file.Path formal = ingest.stagedDir(saved.getId());
            java.nio.file.Files.createDirectories(formal.getParent());
            try {
                java.nio.file.Files.move(tmp, formal,
                        java.nio.file.StandardCopyOption.REPLACE_EXISTING);
            } catch (IOException moveFail) {
                datasets.delete(saved);
                deleteDirQuietly(tmp);
                return ResponseEntity.status(500)
                        .body(Map.of("message", "保存暂存失败，请重试：" + moveFail.getMessage()));
            }
            d = saved;
        } catch (IllegalArgumentException e) {
            deleteDirQuietly(tmp);
            return ResponseEntity.badRequest().body(Map.of("message", e.getMessage()));
        } catch (IOException e) {
            deleteDirQuietly(tmp);
            return ResponseEntity.status(500).body(Map.of("message", "文件处理失败: " + e.getMessage()));
        } catch (RuntimeException e) {
            // 兜底（如并发同名撞 slug 唯一约束）：清理临时目录，返回人话
            deleteDirQuietly(tmp);
            return ResponseEntity.status(500)
                    .body(Map.of("message", "保存失败，请稍后重试：" + e.getMessage()));
        }
        return ResponseEntity.status(HttpStatus.CREATED).body(DatasetDto.from(d));
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<?> delete(@PathVariable Long id) {
        UserDataset d = datasets.findById(id).orElse(null);
        if (d == null) {
            return ResponseEntity.notFound().build();
        }
        if (!Objects.equals(d.getOwnerId(), currentUserId()) && !isAdmin()) {
            return ResponseEntity.status(HttpStatus.FORBIDDEN).build();
        }
        // 训练进行中的数据集不许删：任务的 rsync/训练还在引用它，
        // 删掉会让任务炸出难懂的远端错误；也顺手堵住 slug 释放入口
        if (jobs.existsByDatasetIdAndStatus(id,
                cn.ouc.luminatrail.unikt.training.TrainingJob.RUNNING)) {
            return ResponseEntity.status(409)
                    .body(Map.of("message", "该数据集有训练任务进行中，请先取消"));
        }
        datasets.deleteById(id);
        ingest.deleteStaged(id);
        return ResponseEntity.noContent().build();
    }

    private static void deleteDirQuietly(java.nio.file.Path dir) {
        try (var walk = java.nio.file.Files.walk(dir)) {
            walk.sorted(java.util.Comparator.reverseOrder())
                    .forEach(p -> p.toFile().delete());
        } catch (IOException ignored) {
            // 不存在即视为已清理
        }
    }

    private static Long currentUserId() {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        if (auth != null && auth.getPrincipal() instanceof PortalPrincipal p) {
            return p.getId();
        }
        return null;
    }

    private static boolean isAdmin() {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        return auth != null && auth.getAuthorities().stream()
                .anyMatch(a -> "ROLE_ADMIN".equals(a.getAuthority()));
    }
}
