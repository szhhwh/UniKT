package cn.ouc.luminatrail.unikt.training;

import java.util.List;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.data.repository.query.Param;

/** 训练任务仓储。变更一律走带状态守卫的条件 UPDATE：返回 0 行即任务已被取消/终结，调用方不得再写。 */
public interface JobRepository extends JpaRepository<TrainingJob, Long> {

    List<TrainingJob> findByOwnerIdOrderByIdDesc(Long ownerId);

    long countByNodeIdAndStatus(Long nodeId, String status);

    long countByOwnerIdAndStatusIn(Long ownerId, java.util.List<String> statuses);

    boolean existsByDatasetIdAndStatus(Long datasetId, String status);

    /**
     * 更新日志尾（仅 RUNNING 时生效）。updatedAt 只在日志内容真正变化
     * 时刷新——它是"30 分钟无进展"超时判定的依据，卡死任务（日志停止
     * 增长）不能自己给自己续命。截断到 3600 字符与列宽一致。
     */
    @Modifying
    @Transactional
    @Query("update TrainingJob j set j.logTail = :lt, j.updatedAt = :now"
            + " where j.id = :id and j.status = :running"
            + " and (j.logTail is null or j.logTail <> :lt)")
    int appendLogIfRunning(@Param("id") Long id, @Param("lt") String lt,
                           @Param("now") java.time.Instant now,
                           @Param("running") String running);

    /** 失败并带日志（同一条 UPDATE：终态与日志原子落库，缺一不可）。 */
    @Modifying
    @Transactional
    @Query("update TrainingJob j set j.status = 'FAILED', j.lastError = :e,"
            + " j.logTail = :lt, j.updatedAt = :now"
            + " where j.id = :id and j.status = :running")
    int markFailedWithLog(@Param("id") Long id, @Param("e") String e,
                          @Param("lt") String lt, @Param("now") java.time.Instant now,
                          @Param("running") String running);

    /** 完成并带日志（同上，最终指标行随终态一起落库）。 */
    @Modifying
    @Transactional
    @Query("update TrainingJob j set j.status = 'DONE', j.runDir = :rd,"
            + " j.logTail = :lt, j.updatedAt = :now"
            + " where j.id = :id and j.status = :running")
    int completeWithLog(@Param("id") Long id, @Param("rd") String rd,
                        @Param("lt") String lt, @Param("now") java.time.Instant now,
                        @Param("running") String running);

    /** 标记已启动（仅 RUNNING 时生效；被取消返回 0）。 */
    @Modifying
    @Transactional
    @Query("update TrainingJob j set j.logTail = :lt, j.updatedAt = :now"
            + " where j.id = :id and j.status = :running")
    int markLaunched(@Param("id") Long id, @Param("lt") String lt,
                     @Param("now") java.time.Instant now,
                     @Param("running") String running);

    /** 完成（仅 RUNNING 时生效）。 */
    @Modifying
    @Transactional
    @Query("update TrainingJob j set j.status = 'DONE', j.runDir = :rd,"
            + " j.updatedAt = :now where j.id = :id and j.status = :running")
    int complete(@Param("id") Long id, @Param("rd") String rd,
                 @Param("now") java.time.Instant now,
                 @Param("running") String running);

    /** 失败（仅 RUNNING 时生效，不覆盖"已取消"）。 */
    @Modifying
    @Transactional
    @Query("update TrainingJob j set j.status = 'FAILED', j.lastError = :e,"
            + " j.updatedAt = :now where j.id = :id and j.status = :running")
    int markFailed(@Param("id") Long id, @Param("e") String e,
                   @Param("now") java.time.Instant now,
                   @Param("running") String running);
}
