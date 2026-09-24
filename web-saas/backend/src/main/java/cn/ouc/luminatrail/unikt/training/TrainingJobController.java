package cn.ouc.luminatrail.unikt.training;

import cn.ouc.luminatrail.unikt.dataset.DatasetRepository;
import cn.ouc.luminatrail.unikt.dataset.UserDataset;
import jakarta.validation.Valid;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import cn.ouc.luminatrail.unikt.user.PortalPrincipal;

/** 训练任务接口：创建（触发编排）+ 列表（状态/日志尾）。 */
@RestController
@RequestMapping("/api/training-jobs")
public class TrainingJobController {

    public record CreateJobRequest(
            @NotNull Long datasetId,
            @NotBlank @Pattern(regexp = "[A-Za-z0-9]+", message = "模型名只能字母数字")
                    String modelName,
            @Min(1) @Max(30) int epochs) {
    }

    public record JobDto(Long id, Long datasetId, String datasetName, String modelName,
                         int epochs, String status, String runDir, String logTail,
                         String lastError, Instant createdAt, Instant updatedAt) {

        static JobDto from(TrainingJob j, UserDataset ds) {
            String name = j.getDatasetName() != null ? j.getDatasetName()
                    : ds != null ? ds.getName() : "#" + j.getDatasetId();
            return new JobDto(j.getId(), j.getDatasetId(), name,
                    j.getModelName(), j.getEpochs(), j.getStatus(), j.getRunDir(),
                    j.getLogTail(), j.getLastError(), j.getCreatedAt(), j.getUpdatedAt());
        }
    }

    private final TrainingService training;
    private final JobRepository jobs;
    private final DatasetRepository datasets;

    public TrainingJobController(TrainingService training, JobRepository jobs,
                                 DatasetRepository datasets) {
        this.training = training;
        this.jobs = jobs;
        this.datasets = datasets;
    }

    @GetMapping
    public List<JobDto> list(@RequestParam(required = false) String scope) {
        List<TrainingJob> list = isAdmin() && "all".equals(scope)
                ? jobs.findAll().stream()
                        .sorted((a, b) -> Long.compare(b.getId(), a.getId())).toList()
                : jobs.findByOwnerIdOrderByIdDesc(currentUserId());
        return list.stream()
                .map(j -> JobDto.from(j, datasets.findById(j.getDatasetId()).orElse(null)))
                .toList();
    }

    @org.springframework.web.bind.annotation.DeleteMapping("/{id}")
    public ResponseEntity<?> cancel(@PathVariable Long id) {
        try {
            return ResponseEntity.ok(JobDto.from(
                    training.cancel(currentUserId(), isAdmin(), id),
                    null));
        } catch (IllegalArgumentException e) {
            return ResponseEntity.badRequest().body(Map.of("message", e.getMessage()));
        }
    }

    @PostMapping
    public ResponseEntity<?> create(@Valid @RequestBody CreateJobRequest req) {
        try {
            TrainingJob job = training.create(currentUserId(), req.datasetId(),
                    req.modelName(), req.epochs());
            UserDataset ds = datasets.findById(req.datasetId()).orElse(null);
            return ResponseEntity.status(HttpStatus.CREATED).body(JobDto.from(job, ds));
        } catch (IllegalArgumentException | IllegalStateException e) {
            return ResponseEntity.badRequest().body(Map.of("message", e.getMessage()));
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
