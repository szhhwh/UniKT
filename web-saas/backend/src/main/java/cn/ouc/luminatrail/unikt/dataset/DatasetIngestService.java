package cn.ouc.luminatrail.unikt.dataset;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import org.springframework.stereotype.Service;

/**
 * 上传 CSV 的校验、统计与暂存。
 *
 * Java 侧做入门校验（列/取值/规模/统计），完整的清洗由节点上的
 * generic 数据集流程兜底（P3 触发）。暂存目录：data/uploads/&lt;id&gt;/。
 */
@Service
public class DatasetIngestService {

    /** 校验+统计结果。 */
    public record Stats(long interactions, long users, long questions, long skills) {
    }

    private static final long MAX_ROWS = 2_000_000;
    private static final Set<String> REQUIRED =
            Set.of("user_id", "item_id", "skill_id", "correct");

    /** 校验 interactions 流并写入暂存目录；返回统计。人话错误直接抛 IllegalArgumentException。 */
    public Stats validateAndStoreInteractions(InputStream in, Path target) {
        List<String[]> rows = parseCsv(in);
        if (rows.size() < 3) {
            throw new IllegalArgumentException("数据太少：至少需要表头 + 2 行记录");
        }
        String[] header = rows.get(0);
        Set<String> cols = new LinkedHashSet<>();
        for (String h : header) {
            cols.add(h.trim().toLowerCase(Locale.ROOT));
        }
        if (!cols.containsAll(REQUIRED)) {
            throw new IllegalArgumentException(
                    "缺少必需列: " + REQUIRED + "；实际的列为 " + cols);
        }
        int idxUser = indexOf(header, "user_id");
        int idxItem = indexOf(header, "item_id");
        int idxSkill = indexOf(header, "skill_id");
        int idxCorrect = indexOf(header, "correct");

        Set<String> users = new HashSet<>();
        Set<String> questions = new HashSet<>();
        Set<String> skills = new HashSet<>();
        long n = 0;
        for (int i = 1; i < rows.size(); i++) {
            String[] r = rows.get(i);
            if (r.length != header.length) {
                throw new IllegalArgumentException("第 " + (i + 1) + " 行列数与表头不一致");
            }
            String user = r[idxUser].trim();
            String item = r[idxItem].trim();
            String skill = r[idxSkill].trim();
            String correct = r[idxCorrect].trim().toLowerCase(Locale.ROOT);
            if (user.isEmpty() || item.isEmpty() || skill.isEmpty()) {
                throw new IllegalArgumentException(
                        "第 " + (i + 1) + " 行 user_id/item_id/skill_id 有空值");
            }
            if (!correct.equals("0") && !correct.equals("1")
                    && !correct.equals("true") && !correct.equals("false")) {
                throw new IllegalArgumentException(
                        "第 " + (i + 1) + " 行 correct 不是 0/1：" + r[idxCorrect]);
            }
            users.add(user);
            questions.add(item);
            skills.add(skill);
            n++;
            if (n > MAX_ROWS) {
                throw new IllegalArgumentException("超过最大行数 " + MAX_ROWS);
            }
        }
        if (users.size() < 2) {
            throw new IllegalArgumentException("至少需要 2 个学生（user_id），当前 " + users.size());
        }
        if (skills.size() < 2) {
            throw new IllegalArgumentException("至少需要 2 个知识点（skill_id），当前 " + skills.size());
        }
        if (n < 100) {
            throw new IllegalArgumentException(
                    "交互数太少（" + n + " 行）：知识追踪训练至少需要数百条作答记录");
        }
        store(target, rows);
        return new Stats(n, users.size(), questions.size(), skills.size());
    }

    /** 校验 skills 名称表（skill_id, skill_name 两列）并暂存；返回是否有名称。 */
    public void validateAndStoreSkills(InputStream in, Path target) {
        List<String[]> rows = parseCsv(in);
        if (rows.size() < 2) {
            throw new IllegalArgumentException("skills.csv 至少需要表头 + 1 行");
        }
        Set<String> cols = new HashSet<>();
        for (String h : rows.get(0)) {
            cols.add(h.trim().toLowerCase(Locale.ROOT));
        }
        if (!cols.containsAll(Set.of("skill_id", "skill_name"))) {
            throw new IllegalArgumentException(
                    "skills.csv 需要 skill_id 与 skill_name 两列；实际的列为 " + cols);
        }
        store(target, rows);
    }

    /**
     * 展示名 → 数据集 slug（generic_ 前缀 + 仅小写字母数字短横线，全局
     * 唯一）。slug 会成为节点目录名与 shell 参数，必须 ASCII：中文名
     * 落到 "data"/"data-2" 这类兜底名（展示名保留中文）。
     */
    public String slugFor(String name, DatasetRepository repo) {
        String base = name.toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9]+", "-")
                .replaceAll("(^-+|-+$)", "");
        if (base.isEmpty() || base.length() < 2) {
            base = "data";
        }
        String slug = "generic_" + base;
        int i = 2;
        while (repo.existsBySlug(slug)) {
            slug = "generic_" + base + "-" + i++;
        }
        return slug;
    }

    public Path stagedDir(long datasetId) {
        return Path.of("data", "uploads", String.valueOf(datasetId));
    }

    /** 删除暂存目录（递归、幂等）。 */
    public void deleteStaged(long datasetId) {
        try (var walk = Files.walk(stagedDir(datasetId))) {
            walk.sorted(java.util.Comparator.reverseOrder())
                    .forEach(p -> p.toFile().delete());
        } catch (IOException ignored) {
            // 目录不存在即视为已删除
        }
    }

    private static void store(Path target, List<String[]> rows) {
        try {
            Files.createDirectories(target.getParent());
            StringBuilder sb = new StringBuilder();
            for (String[] r : rows) {
                for (int i = 0; i < r.length; i++) {
                    if (i > 0) {
                        sb.append(',');
                    }
                    sb.append(csvEscape(r[i]));
                }
                sb.append('\n');
            }
            Files.writeString(target, sb.toString(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new IllegalStateException("暂存文件写入失败: " + e.getMessage(), e);
        }
    }

    /** 最小 CSV 解析：支持双引号包裹与 "" 转义。 */
    private static List<String[]> parseCsv(InputStream in) {
        List<String[]> rows = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(in, StandardCharsets.UTF_8), 1 << 16)) {
            StringBuilder field = new StringBuilder();
            List<String> row = new ArrayList<>();
            boolean inQuotes = false;
            int c;
            while ((c = reader.read()) != -1) {
                char ch = (char) c;
                if (inQuotes) {
                    if (ch == '"') {
                        if (reader.ready()) {
                            reader.mark(1);
                            int next = reader.read();
                            if (next == '"') {
                                field.append('"');
                                continue;
                            }
                            reader.reset();
                        }
                        inQuotes = false;
                    } else {
                        field.append(ch);
                    }
                } else if (ch == '"') {
                    inQuotes = true;
                } else if (ch == ',') {
                    row.add(field.toString());
                    field.setLength(0);
                } else if (ch == '\n' || ch == '\r') {
                    if (ch == '\r' && reader.ready()) {
                        reader.mark(1);
                        if (reader.read() != '\n') {
                            reader.reset();
                        }
                    }
                    row.add(field.toString());
                    field.setLength(0);
                    if (!(row.size() == 1 && row.get(0).isEmpty())) {
                        rows.add(row.toArray(new String[0]));
                    }
                    row = new ArrayList<>();
                } else {
                    field.append(ch);
                }
            }
            if (field.length() > 0 || !row.isEmpty()) {
                row.add(field.toString());
                rows.add(row.toArray(new String[0]));
            }
        } catch (IOException e) {
            throw new IllegalArgumentException("CSV 读取失败: " + e.getMessage());
        }
        return rows;
    }

    private static String csvEscape(String v) {
        if (v.contains(",") || v.contains("\"") || v.contains("\n")) {
            return '"' + v.replace("\"", "\"\"") + '"';
        }
        return v;
    }

    private static int indexOf(String[] header, String col) {
        for (int i = 0; i < header.length; i++) {
            if (header[i].trim().toLowerCase(Locale.ROOT).equals(col)) {
                return i;
            }
        }
        throw new IllegalArgumentException("缺少列: " + col);
    }
}
