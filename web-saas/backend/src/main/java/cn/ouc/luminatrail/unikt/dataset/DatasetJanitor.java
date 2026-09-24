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
 * 启动清扫：上传流程两次 save 之间崩溃留下的孤儿——
 * (a) data/uploads/tmp-* 目录（JVM 崩溃时没人清理）；
 * (b) slug 仍是 generic_tmp-* 的库行（save#1 后崩溃，从未回填正式
 *     slug，留着只会让训练在 rsync 处炸出难懂错误）。
 *
 * 刻意**不**处理"READY 但暂存目录缺失"的行：stagedDir 是相对路径，
 * 从别的工作目录启动一次（IDEA 配错/仓库根目录跑 jar）就会全员误标
 * FAILED 且不可逆。目录缺失留给训练提交时的 rsync 自然报错。
 */
@Component
public class DatasetJanitor implements ApplicationRunner {

    private static final Logger log = LoggerFactory.getLogger(DatasetJanitor.class);

    private final DatasetRepository datasets;

    public DatasetJanitor(DatasetRepository datasets) {
        this.datasets = datasets;
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
            if (d.getSlug().startsWith("generic_tmp-")) {
                d.setStatus(UserDataset.FAILED);
                d.setLastError("上传中断（服务重启），请删除后重新上传");
                datasets.save(d);
                rows++;
            }
        }
        if (dirs > 0 || rows > 0) {
            log.info("DatasetJanitor：清理 {} 个 tmp 目录，标记 {} 行未完成上传", dirs, rows);
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
