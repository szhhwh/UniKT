package cn.ouc.luminatrail.unikt.training;

import java.util.List;
import org.springframework.data.jpa.repository.JpaRepository;

/** 训练任务仓储。 */
public interface JobRepository extends JpaRepository<TrainingJob, Long> {

    List<TrainingJob> findByOwnerIdOrderByIdDesc(Long ownerId);

    long countByNodeIdAndStatus(Long nodeId, String status);
}
