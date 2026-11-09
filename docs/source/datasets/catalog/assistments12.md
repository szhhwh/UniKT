# ASSISTments 2012

2012—2013 学年的学校数据，包含情感状态预测信息。

数据集链接：[2012—2013 官方数据页](https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect)。

## 数据参数

| 参数 | 内容 |
| --- | --- |
| 数据文件 | `2012-2013-data-with-predictions-4-final.csv` |
| 数据行 / 不重复作答日志 | 6,123,270 / 6,123,270，按 `problem_log_id` 去重 |
| 学生 / 题目 / 技能 | 46,674 / 179,999 / 265，分别按 `user_id`、`problem_id`、非空 `skill_id` 去重 |
| 缺失技能的行数 | 3,411,457 |

去重时排除空值及 `NA`、`nan`、`null` 标记（不区分大小写）。

## 字段

| 字段名称 | 字段语义说明 |
| :--- | :--- |
| `problem_log_id` | 答题交互日志的主键自增 ID |
| `skill` | 知识组件（技能）英文名称 |
| `problem_id` | 题目编号 |
| `user_id` | 匿名学生编号 |
| `assignment_id` | 作业编号 |
| `assistment_id` | 母题编号 |
| `start_time` | 学生打开题目的起始系统时间戳 |
| `end_time` | 学生提交或离开题目的结束时间戳 |
| `problem_type` | 题目题型（选择、代数式、数值等） |
| `original` | 是否为主干母题（1 为主干题，0 为支架辅助题） |
| `correct` | 作答得分；可能包含部分得分，1 表示首次正确 |
| `bottom_hint` | 是否触发到底部答案解析提示 |
| `hint_count` | 请求提示次数 |
| `actions` | 该题目内发生的完整动作行为文本串 |
| `attempt_count` | 答案提交尝试次数 |
| `ms_first_response` | 首答响应耗时（毫秒） |
| `tutor_mode` | 辅导系统运行模式 |
| `sequence_id` | 练习序列单元编号 |
| `student_class_id` | 班级编号 |
| `position` | 作业在班级作业页面中的位置 |
| `type` | 题目呈现方式（线性呈现或随机抽取） |
| `base_sequence_id` | 基础序列编号 |
| `skill_id` | 知识组件（技能）数字编号 |
| `teacher_id` | 教师编号 |
| `school_id` | 学校编号 |
| `overlap_time` | 重叠时间，单位为毫秒 |
| `template_id` | 参数化题目模板编号 |
| `answer_id` | 选项代号 |
| `answer_text` | 提交的具体文本 |
| `first_action` | 首次交互类型，包括尝试作答和请求提示 |
| `problemlogid` | 对应交互事件的副本索引标识符 |
| `Average_confidence(FRUSTRATED)` | **挫败感**：情感检测模型输出的学生处于受挫状态的置信概率 |
| `Average_confidence(CONFUSED)` | **困惑感**：情感检测模型输出的学生处于困惑状态的置信概率 |
| `Average_confidence(CONCENTRATING)` | **专注度**：学生高度聚焦投入解题的置信概率 |
| `Average_confidence(BORED)` | **厌烦感**：学生走神、厌学或倦怠状态的置信概率 |

