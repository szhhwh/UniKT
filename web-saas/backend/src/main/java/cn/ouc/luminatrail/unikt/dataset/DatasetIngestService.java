package cn.ouc.luminatrail.unikt.dataset;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.HashSet;
import java.util.ArrayList;
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

    /**
     * 校验 interactions 流并写入暂存目录（单遍流式：边解析边校验边写出，
     * 内存只保留三个 id 集合）。人话错误直接抛 IllegalArgumentException。
     */
    public Stats validateAndStoreInteractions(InputStream in, Path target) {
        Set<String> users = new HashSet<>();
        Set<String> questions = new HashSet<>();
        Set<String> skills = new HashSet<>();
        long[] n = {0};
        boolean[] headerSeen = {false};
        try {
            Files.createDirectories(target.getParent());
        } catch (IOException e) {
            throw new IllegalStateException("暂存目录创建失败: " + e.getMessage(), e);
        }
        try (InputStream raw = in;
                java.io.BufferedWriter out = Files.newBufferedWriter(
                        target, StandardCharsets.UTF_8)) {
            int[] idx = new int[4];
            int[] headerLen = {0};
            int[] tsIdx = {-1};
            streamCsv(raw, (lineNo, r) -> {
                try {
                    if (!headerSeen[0]) {
                        Set<String> cols = new LinkedHashSet<>();
                        for (String h : r) {
                            cols.add(h.trim().toLowerCase(Locale.ROOT));
                            if (h.trim().toLowerCase(Locale.ROOT).equals("timestamp")) {
                                tsIdx[0] = java.util.Arrays.asList(r).indexOf(h);
                            }
                        }
                        if (!cols.containsAll(REQUIRED)) {
                            throw new IllegalArgumentException(
                                    "缺少必需列: " + REQUIRED + "；实际的列为 " + cols);
                        }
                        idx[0] = indexOf(r, "user_id");
                        idx[1] = indexOf(r, "item_id");
                        idx[2] = indexOf(r, "skill_id");
                        idx[3] = indexOf(r, "correct");
                        headerLen[0] = r.length;
                        headerSeen[0] = true;
                        writeRow(out, r);
                        return true;
                    }
                    if (r.length != headerLen[0]) {
                        throw new IllegalArgumentException("第 " + lineNo + " 行列数与表头不一致");
                    }
                    String user = r[idx[0]].trim();
                    String item = r[idx[1]].trim();
                    String skill = r[idx[2]].trim();
                    String correct = r[idx[3]].trim().toLowerCase(Locale.ROOT);
                    if (user.isEmpty() || item.isEmpty() || skill.isEmpty()) {
                        throw new IllegalArgumentException(
                                "第 " + lineNo + " 行 user_id/item_id/skill_id 有空值");
                    }
                    if (!correct.equals("0") && !correct.equals("1")
                            && !correct.equals("true") && !correct.equals("false")) {
                        throw new IllegalArgumentException(
                                "第 " + lineNo + " 行 correct 不是 0/1：" + r[idx[3]]);
                    }
                    if (tsIdx[0] >= 0
                            && !r[tsIdx[0]].trim().matches("[0-9]{1,19}")) {
                        throw new IllegalArgumentException(
                                "第 " + lineNo + " 行 timestamp 需为整数（Unix 秒），实际："
                                        + r[tsIdx[0]]);
                    }
                    users.add(user);
                    questions.add(item);
                    skills.add(skill);
                    writeRow(out, r);
                    n[0]++;
                    if (n[0] > MAX_ROWS) {
                        throw new IllegalArgumentException("超过最大行数 " + MAX_ROWS);
                    }
                    return true;
                } catch (IOException e) {
                    throw new RuntimeException("暂存写入失败: " + e.getMessage(), e);
                }
            });
        } catch (IOException e) {
            throw new IllegalArgumentException("CSV 读取失败: " + e.getMessage());
        } catch (RuntimeException e) {
            if (e.getCause() instanceof IOException) {
                throw new IllegalArgumentException("暂存写入失败: " + e.getCause().getMessage());
            }
            throw e;
        }
        if (!headerSeen[0]) {
            throw new IllegalArgumentException("文件为空");
        }
        if (n[0] < 100) {
            throw new IllegalArgumentException(
                    "交互数太少（" + n[0] + " 行）：知识追踪训练至少需要数百条作答记录");
        }
        if (users.size() < 2) {
            throw new IllegalArgumentException("至少需要 2 个学生（user_id），当前 " + users.size());
        }
        if (skills.size() < 2) {
            throw new IllegalArgumentException("至少需要 2 个知识点（skill_id），当前 " + skills.size());
        }
        return new Stats(n[0], users.size(), questions.size(), skills.size());
    }

    private static void writeRow(java.io.BufferedWriter out, String[] r) throws IOException {
        for (int i = 0; i < r.length; i++) {
            if (i > 0) {
                out.write(',');
            }
            out.write(csvEscape(r[i]));
        }
        out.write('\n');
    }

    /** 校验 skills 名称表（skill_id, skill_name 两列）并流式暂存。 */
    public void validateAndStoreSkills(InputStream in, Path target) {
        long[] n = {0};
        boolean[] headerSeen = {false};
        try {
            Files.createDirectories(target.getParent());
        } catch (IOException e) {
            throw new IllegalStateException("暂存目录创建失败: " + e.getMessage(), e);
        }
        try (InputStream raw = in;
                java.io.BufferedWriter out = Files.newBufferedWriter(
                        target, StandardCharsets.UTF_8)) {
            streamCsv(raw, (lineNo, r) -> {
                try {
                    if (!headerSeen[0]) {
                        Set<String> cols = new HashSet<>();
                        for (String h : r) {
                            cols.add(h.trim().toLowerCase(Locale.ROOT));
                        }
                        if (!cols.containsAll(Set.of("skill_id", "skill_name"))) {
                            throw new IllegalArgumentException(
                                    "skills.csv 需要 skill_id 与 skill_name 两列；实际的列为 "
                                            + cols);
                        }
                        headerSeen[0] = true;
                        writeRow(out, r);
                        return true;
                    }
                    writeRow(out, r);
                    n[0]++;
                    if (n[0] > MAX_ROWS) {
                        throw new IllegalArgumentException("skills.csv 超过最大行数 " + MAX_ROWS);
                    }
                    return true;
                } catch (IOException e) {
                    throw new RuntimeException("暂存写入失败: " + e.getMessage(), e);
                }
            });
        } catch (IOException e) {
            throw new IllegalArgumentException("CSV 读取失败: " + e.getMessage());
        } catch (RuntimeException e) {
            if (e.getCause() instanceof IOException) {
                throw new IllegalArgumentException("暂存写入失败: " + e.getCause().getMessage());
            }
            throw e;
        }
        if (!headerSeen[0] || n[0] == 0) {
            throw new IllegalArgumentException("skills.csv 至少需要表头 + 1 行");
        }
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


    /** 最小 CSV 解析：支持双引号包裹与 "" 转义。 */
    /**
     * 单遍流式处理：逐行解析（支持引号/转义/\r\n/UTF-8 BOM），每行交给
     * handler 消费，handler 返回 false 立即终止（校验失败止损，不把整个
     * 文件读进内存——200MB 上传在全量模式下会 OOM）。
     */
    private static void streamCsv(InputStream in, CsvRowHandler handler)
            throws IOException {
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(in, StandardCharsets.UTF_8), 1 << 16)) {
            StringBuilder field = new StringBuilder(64);
            List<String> row = new ArrayList<>(8);
            boolean inQuotes = false;
            boolean firstChar = true;
            int c;
            long lineNo = 0;
            while ((c = reader.read()) != -1) {
                char ch = (char) c;
                if (firstChar) {
                    firstChar = false;
                    if (ch == '\uFEFF') {
                        continue; // Excel UTF-8 CSV 的 BOM
                    }
                }
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
                        lineNo++;
                        if (!handler.accept((int) lineNo, row.toArray(new String[0]))) {
                            return;
                        }
                    }
                    row = new ArrayList<>(8);
                } else {
                    field.append(ch);
                }
            }
            if (field.length() > 0 || !row.isEmpty()) {
                row.add(field.toString());
                handler.accept((int) lineNo + 1, row.toArray(new String[0]));
            }
        }
    }

    @FunctionalInterface
    private interface CsvRowHandler {

        /** 消费一行（lineNo 从 1 计）；返回 false 终止读取。 */
        boolean accept(int lineNo, String[] cells);
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
