package cn.ouc.luminatrail.unikt.apikey;

import java.time.Instant;
import java.util.List;
import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

/** API Key 仓储。 */
public interface ApiKeyRepository extends JpaRepository<ApiKeyUser, Long> {

    Optional<ApiKeyUser> findByKeyHash(String keyHash);

    List<ApiKeyUser> findByOwnerIdOrderByIdDesc(Long ownerId);

    /**
     * 原子消耗一次当日配额：条件相对自增（跨天自动重置为 1），
     * WHERE 里的配额守卫防止并发超发——返回 0 行即当日已满。
     */
    @Modifying
    @Query("update ApiKeyUser u set"
            + " u.dailyCount = case when u.dailyDate = :day then u.dailyCount + 1 else 1 end,"
            + " u.dailyDate = :day,"
            + " u.requestCount = u.requestCount + 1, u.lastUsedAt = :now"
            + " where u.id = :id and (u.dailyDate <> :day or u.dailyCount < :limit)")
    int consume(@Param("id") Long id, @Param("day") String day,
                @Param("limit") long limit, @Param("now") Instant now);
}
