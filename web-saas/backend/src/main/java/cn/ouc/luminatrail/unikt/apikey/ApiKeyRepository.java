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

    /** 原子写入当日计数与用量（配额不超发）。 */
    @Modifying
    @Query("update ApiKeyUser u set u.dailyCount = :count, u.dailyDate = :day,"
            + " u.requestCount = u.requestCount + 1, u.lastUsedAt = :now"
            + " where u.id = :id")
    int consume(@Param("id") Long id, @Param("day") String day,
                @Param("count") long count, @Param("now") Instant now);
}
