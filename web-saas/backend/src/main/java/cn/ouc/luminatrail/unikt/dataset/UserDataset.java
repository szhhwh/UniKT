package cn.ouc.luminatrail.unikt.dataset;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import java.time.Instant;

/** 用户上传的自有数据集（generic_<slug>，训练流水线的数据源）。 */
@Entity
@Table(name = "user_dataset")
public class UserDataset {

    public static final String READY = "READY";
    public static final String FAILED = "FAILED";

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private Long ownerId;

    /** 展示名。 */
    @Column(nullable = false, length = 60)
    private String name;

    /** generic_<slug>：节点上的数据集目录名，全局唯一。 */
    @Column(nullable = false, unique = true, length = 80)
    private String slug;

    private String status = READY;

    private long interactions;

    private long users;

    private long questions;

    private long skills;

    private boolean hasSkillNames;

    @Column(length = 4000)
    private String lastError;

    private Instant createdAt = Instant.now();

    public Long getId() {
        return id;
    }

    public Long getOwnerId() {
        return ownerId;
    }

    public void setOwnerId(Long ownerId) {
        this.ownerId = ownerId;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getSlug() {
        return slug;
    }

    public void setSlug(String slug) {
        this.slug = slug;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public long getInteractions() {
        return interactions;
    }

    public void setInteractions(long interactions) {
        this.interactions = interactions;
    }

    public long getUsers() {
        return users;
    }

    public void setUsers(long users) {
        this.users = users;
    }

    public long getQuestions() {
        return questions;
    }

    public void setQuestions(long questions) {
        this.questions = questions;
    }

    public long getSkills() {
        return skills;
    }

    public void setSkills(long skills) {
        this.skills = skills;
    }

    public boolean isHasSkillNames() {
        return hasSkillNames;
    }

    public void setHasSkillNames(boolean hasSkillNames) {
        this.hasSkillNames = hasSkillNames;
    }

    public String getLastError() {
        return lastError;
    }

    public void setLastError(String lastError) {
        if (lastError != null && lastError.length() > 3900) {
            lastError = lastError.substring(0, 3900);
        }
        this.lastError = lastError;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }
}
