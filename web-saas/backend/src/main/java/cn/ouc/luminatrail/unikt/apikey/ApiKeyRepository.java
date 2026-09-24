package cn.ouc.luminatrail.unikt.apikey;

import java.time.Instant;
import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

/** API Key 仓储。 */
public interface ApiKeyRepository extends JpaRepository<ApiKeyUser, Long> {

    Optional<ApiKeyUser> findByKeyHash(String keyHash);

    /** 原子自增用量（避免 detached merge 的并发丢更新）。 */
    @Modifying
    @Query("update ApiKeyUser u set u.requestCount = u.requestCount + 1,"
            + " u.lastUsedAt = :now where u.id = :id")
    int touch(@Param("id") Long id, @Param("now") Instant now);
}
