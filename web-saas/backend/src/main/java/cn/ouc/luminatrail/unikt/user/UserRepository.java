package cn.ouc.luminatrail.unikt.user;

import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;

/** 用户仓储。 */
public interface UserRepository extends JpaRepository<User, Long> {

    Optional<User> findByUsername(String username);

    boolean existsByUsername(String username);
}
