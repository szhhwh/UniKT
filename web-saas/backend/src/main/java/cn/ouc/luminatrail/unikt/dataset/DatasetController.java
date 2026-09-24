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

    public DatasetController(DatasetRepository datasets, DatasetIngestService ingest) {
        this.datasets = datasets;
        this.ingest = ingest;
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

        try (InputStream in = interactions.getInputStream()) {
            // 先用临时 id 暂存不行——slug 唯一性已查；文件先落到以 slug 命名的临时目录，
            // 实体落库拿到 id 后改名，避免半成品
            java.nio.file.Path tmp = java.nio.file.Path.of("data", "uploads", "tmp-" + d.getSlug());
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
            d = datasets.save(d);
            // 挪到正式目录（按 id）
            java.nio.file.Path formal = ingest.stagedDir(d.getId());
            java.nio.file.Files.createDirectories(formal.getParent());
            java.nio.file.Files.move(tmp, formal,
                    java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        } catch (IllegalArgumentException e) {
            // 校验失败：清理临时目录后返回人话错误
            try {
                java.nio.file.Path tmp = java.nio.file.Path.of("data", "uploads", "tmp-" + d.getSlug());
                try (var walk = java.nio.file.Files.walk(tmp)) {
                    walk.sorted(java.util.Comparator.reverseOrder())
                            .forEach(pth -> pth.toFile().delete());
                }
            } catch (IOException ignored) {
                // 清理失败不阻塞错误返回
            }
            return ResponseEntity.badRequest().body(Map.of("message", e.getMessage()));
        } catch (IOException e) {
            return ResponseEntity.status(500).body(Map.of("message", "文件处理失败: " + e.getMessage()));
        }
        return ResponseEntity.status(HttpStatus.CREATED).body(DatasetDto.from(d));
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> delete(@PathVariable Long id) {
        UserDataset d = datasets.findById(id).orElse(null);
        if (d == null) {
            return ResponseEntity.notFound().build();
        }
        if (!Objects.equals(d.getOwnerId(), currentUserId()) && !isAdmin()) {
            return ResponseEntity.status(HttpStatus.FORBIDDEN).build();
        }
        datasets.deleteById(id);
        ingest.deleteStaged(id);
        return ResponseEntity.noContent().build();
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
