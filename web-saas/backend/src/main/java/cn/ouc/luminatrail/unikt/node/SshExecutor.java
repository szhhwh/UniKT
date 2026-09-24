package cn.ouc.luminatrail.unikt.node;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

/**
 * SSH/rsync 执行器：直接调用本机 ssh 客户端（复用 ~/.ssh/config 与密钥，
 * 不引入 Java SSH 库）。所有命令用参数数组传递，不经本地 shell。
 *
 * 进程输出由独立的排水线程并发读取，主线程先限时 waitFor 再收流——
 * 避免「子进程 stderr 写满管道导致死锁」和「黑洞主机挂死读线程」两类
 * 问题（超时后 destroyForcibly 强杀并收尸）。
 *
 * 远端命令中的变量必须经过 {@link #shellQuote} 单引号包裹，防注入。
 */
@Component
public class SshExecutor {

    private static final Logger log = LoggerFactory.getLogger(SshExecutor.class);

    /** 排水线程池：daemon，仅负责读子进程输出。 */
    // 弹性池：部署/轮询/编排/取消多源并发时，固定小池会让第 3 个进程的
    // 输出管道无人排水 → 子进程写阻塞 → waitFor 超时误杀
    private static final ExecutorService DRAIN_POOL = Executors.newCachedThreadPool(
            r -> {
                Thread t = new Thread(r, "ssh-drain");
                t.setDaemon(true);
                return t;
            });

    /** 一次 SSH 命令的结果。 */
    public record SshResult(int exitCode, String stdout, String stderr, Duration duration) {
        public boolean ok() {
            return exitCode == 0;
        }
    }

    /** 在 target 上执行 command，超时（秒）后强杀。 */
    public SshResult run(String target, String command, int timeoutSeconds) {
        List<String> argv = new ArrayList<>(List.of(
                "ssh",
                "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=10",
                "-o", "StrictHostKeyChecking=accept-new",
                target,
                command));
        return exec(argv, timeoutSeconds);
    }

    /** 本机 rsync（源目录 → target:destDir，内容同步）。 */
    public SshResult rsync(String srcDir, String target, String destDir, int timeoutSeconds) {
        List<String> argv = List.of(
                "rsync", "-a", "--delete", "--exclude", "__pycache__",
                srcDir + "/",
                target + ":" + destDir + "/");
        return exec(argv, timeoutSeconds);
    }

    private SshResult exec(List<String> argv, int timeoutSeconds) {
        long start = System.nanoTime();
        try {
            Process p = new ProcessBuilder(argv)
                    .redirectErrorStream(false)
                    .directory(new File("."))
                    .start();
            CompletableFuture<String> out = drainAsync(p.getInputStream());
            CompletableFuture<String> err = drainAsync(p.getErrorStream());
            try {
                boolean finished = p.waitFor(timeoutSeconds, TimeUnit.SECONDS);
                if (!finished) {
                    p.destroyForcibly();
                    p.waitFor(5, TimeUnit.SECONDS);
                    return result(p, -1, out, err, start,
                            "timeout after " + timeoutSeconds + "s");
                }
                return result(p, p.exitValue(), out, err, start, null);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                p.destroyForcibly();
                return result(p, -1, out, err, start, "interrupted");
            }
        } catch (IOException e) {
            return new SshResult(-1, "", "spawn failed: " + e.getMessage(),
                    Duration.ofNanos(System.nanoTime() - start));
        }
    }

    private SshResult result(
            Process p, int code, CompletableFuture<String> out,
            CompletableFuture<String> err, long start, String forcedError) {
        String stdout = await(out);
        String stderr = await(err);
        if (forcedError != null) {
            // 超时/中断路径：输出可能不完整，以错误说明为准
            stderr = stderr.isBlank() ? forcedError : stderr + " (" + forcedError + ")";
        }
        return new SshResult(code, stdout.trim(), stderr.trim(),
                Duration.ofNanos(System.nanoTime() - start));
    }

    private static CompletableFuture<String> drainAsync(java.io.InputStream in) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                return new String(in.readAllBytes(), StandardCharsets.UTF_8);
            } catch (IOException e) {
                return "";
            } finally {
                try {
                    in.close();
                } catch (IOException ignored) {
                    // 流随进程销毁关闭
                }
            }
        }, DRAIN_POOL);
    }

    private static String await(CompletableFuture<String> f) {
        try {
            return f.get(5, TimeUnit.SECONDS);
        } catch (Exception e) {
            return "";
        }
    }

    /** 远端 shell 单引号包裹（内部单引号转义为 '\''），用于把路径安全拼进命令。 */
    public static String shellQuote(String s) {
        return "'" + s.replace("'", "'\\''") + "'";
    }

    /** 读取 stdout 尾部若干行，用于日志展示。 */
    public static String tail(String text, int lines) {
        if (text == null || text.isEmpty()) {
            return "";
        }
        String[] all = text.split("\n");
        int from = Math.max(0, all.length - lines);
        return String.join("\n", java.util.Arrays.copyOfRange(all, from, all.length));
    }
}
