package cn.ouc.luminatrail.unikt.node;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import java.time.Instant;
import java.util.List;
import org.springframework.data.domain.Sort;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * 计算节点管理：注册远程机器 → 一键自动部署 → 状态/日志查询。
 *
 * 列表接口只读库（状态探测由 DeployService 的后台定时任务负责）。
 * 输入字段带严格白名单校验（sshTarget/repoPath 会进入远端 shell 命令），
 * 命令拼接侧另有单引号包裹（见 SshExecutor.shellQuote），双层防注入。
 * 全部接口要求 ADMIN 角色（SecurityConfig）。
 */
@RestController
@RequestMapping("/api/nodes")
public class NodeController {

    /** 新建节点请求。 */
    public record CreateNodeRequest(
            @NotBlank @Size(max = 40) String name,
            @NotBlank @Size(max = 255)
            @Pattern(regexp = "[A-Za-z0-9][A-Za-z0-9@._-]*",
                    message = "SSH 目标须以字母数字开头，只能含 @._- 等字符")
                    String sshTarget,
            @NotBlank @Size(max = 255)
            @Pattern(regexp = "https?://[A-Za-z0-9.:/\\-]+", message = "baseUrl 需为 http(s)://host:port")
                    String baseUrl,
            @NotBlank @Size(max = 255)
            @Pattern(regexp = "/[A-Za-z0-9._/-]*", message = "repoPath 需为绝对路径")
                    String repoPath) {
    }

    /** 节点视图。 */
    public record NodeDto(
            Long id, String name, String sshTarget, String baseUrl, String repoPath,
            String status, int modelsCount, String lastError, String deployLog,
            Instant createdAt, Instant updatedAt) {

        static NodeDto from(ComputeNode n) {
            return new NodeDto(n.getId(), n.getName(), n.getSshTarget(), n.getBaseUrl(),
                    n.getRepoPath(), n.getStatus(), n.getModelsCount(), n.getLastError(),
                    n.getDeployLog(), n.getCreatedAt(), n.getUpdatedAt());
        }
    }

    private final NodeRepository nodes;
    private final DeployService deployer;

    public NodeController(NodeRepository nodes, DeployService deployer) {
        this.nodes = nodes;
        this.deployer = deployer;
    }

    @GetMapping
    public List<NodeDto> list() {
        return nodes.findAll(Sort.by("id")).stream().map(NodeDto::from).toList();
    }

    @PostMapping
    public ResponseEntity<NodeDto> create(@Valid @RequestBody CreateNodeRequest req) {
        ComputeNode n = new ComputeNode();
        n.setName(req.name().trim());
        n.setSshTarget(req.sshTarget().trim());
        n.setBaseUrl(DeployService.normalize(req.baseUrl().trim()));
        n.setRepoPath(req.repoPath().trim());
        return ResponseEntity.status(HttpStatus.CREATED).body(NodeDto.from(nodes.save(n)));
    }

    @PostMapping("/{id}/deploy")
    public ResponseEntity<?> deploy(@PathVariable Long id) {
        if (!nodes.existsById(id)) {
            return ResponseEntity.notFound().build();
        }
        boolean started = deployer.deployAsync(id);
        if (!started) {
            return ResponseEntity.status(HttpStatus.CONFLICT)
                    .body(java.util.Map.of("message", "该节点正在部署中"));
        }
        return ResponseEntity.accepted().body(java.util.Map.of("message", "部署已开始"));
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> delete(@PathVariable Long id) {
        if (!nodes.existsById(id)) {
            return ResponseEntity.notFound().build();
        }
        nodes.deleteById(id);
        return ResponseEntity.noContent().build();
    }
}
