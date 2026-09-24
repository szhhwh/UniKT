package cn.ouc.luminatrail.unikt.common;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.springframework.stereotype.Component;

/**
 * 进程内滑动窗口限流器（够课程规模；多实例部署需换 Redis，记录在案）。
 *
 * key 形如 "predict:1.2.3.4"；窗口内超过 limit 次即拒绝。
 */
@Component
public class RateLimiter {

    private final Map<String, Deque<Long>> hits = new ConcurrentHashMap<>();

    /** 尝试获取一次配额：窗口内未超限则记录并返回 true。 */
    public boolean tryAcquire(String key, int limit, long windowMs) {
        long now = System.currentTimeMillis();
        Deque<Long> q = hits.computeIfAbsent(key, k -> new ArrayDeque<>());
        synchronized (q) {
            while (!q.isEmpty() && now - q.peekFirst() >= windowMs) {
                q.pollFirst();
            }
            if (q.size() >= limit) {
                return false;
            }
            q.addLast(now);
        }
        if (hits.size() > 10000) {
            // 粗暴防内存膨胀：极端 key 量时整体重置（计数保守即可，不需精确）
            hits.clear();
        }
        return true;
    }
}
