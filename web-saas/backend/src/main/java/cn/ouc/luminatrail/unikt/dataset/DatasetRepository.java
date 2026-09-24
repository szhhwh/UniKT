package cn.ouc.luminatrail.unikt.dataset;

import java.util.List;
import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;

/** 数据集仓储。 */
public interface DatasetRepository extends JpaRepository<UserDataset, Long> {

    List<UserDataset> findByOwnerIdOrderByIdDesc(Long ownerId);

    Optional<UserDataset> findBySlug(String slug);

    boolean existsBySlug(String slug);
}
