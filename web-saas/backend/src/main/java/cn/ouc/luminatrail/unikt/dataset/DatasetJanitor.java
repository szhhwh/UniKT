package cn.ouc.luminatrail.unikt.dataset;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.stereotype.Component;

/**
 * 启动清扫：上传流程两次 save / 挪盘之间崩溃留下的孤儿——
 * (a) data/uploads/tmp-* 目录（JVM 崩溃时没人清理）；
 * (b) slug 仍是 generic_tmp-* 或暂存目录缺失却 READY 的库行（标 FAILED，
 *     避免用户提交训练后在 rsync 处炸出难懂错误）。
 */
@Component
public class DatasetJanitor implements ApplicationRunner {

    private static final Logger log = LoggerFactory.getLogger(DatasetJanitor.class);

    private final DatasetRepository datasets;
    private final DatasetIngestService ingest;

    public DatasetJanitor(DatasetRepository datasets, DatasetIngestService ingest) {
        this.datasets = datasets;
        this.ingest = ingest;
    }

    @Override
    public void run(ApplicationArguments args) {
        int dirs = 0;
        try (var walk = Files.walk(Path.of("data", "uploads"), 1)) {
            for (var p : walk.filter(Files::isDirectory).toList()) {
                if (p.getFileName().toString().startsWith("tmp-")) {
                    deleteRecursively(p);
                    dirs++;
                }
            }
        } catch (IOException ignored) {
            // 目录不存在：无事可扫
        }
        int rows = 0;
        for (UserDataset d : datasets.findAll()) {
            boolean orphan = d.getSlug().startsWith("generic_tmp-")
                    || (UserDataset.READY.equals(d.getStatus())
                        && !Files.isDirectory(ingest.stagedDir(d.getId())));
            if (orphan) {
                d.setStatus(UserDataset.FAILED);
                d.setLastError("上传中断（服务重启），请删除后重新上传");
                datasets.save(d);
                rows++;
            }
        }
        if (dirs > 0 || rows > 0) {
            log.info("DatasetJanitor：清理 {} 个 tmp 目录，标记 {} 行孤儿数据集", dirs, rows);
        }
    }

    private static void deleteRecursively(Path dir) {
        try (var walk = Files.walk(dir)) {
            walk.sorted(java.util.Comparator.reverseOrder())
                    .forEach(p -> p.toFile().delete());
        } catch (IOException ignored) {
            // 尽力而为
        }
    }
}
