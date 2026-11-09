# ASSISTments 2009

2009—2010 学年的 Skill Builder 数学练习数据。

数据集链接：[2009—2010 官方数据页](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data)。

## 数据参数

| 参数 | 内容 |
| --- | --- |
| 数据文件 | `skill_builder_data_corrected_collapsed.csv` |
| 数据行 / 不重复作答日志 | 346,860 / 346,860，按 `order_id` 去重 |
| 学生 / 题目 / 技能 | 4,217 / 26,688 / 149，分别按 `user_id`、`problem_id`、非空 `skill_id` 去重 |
| 缺失技能的行数 | 63,755 |

去重时排除空值及 `NA`、`nan`、`null` 标记（不区分大小写）。

## 字段

| 字段名称 | 字段语义说明 |
| :--- | :--- |
| 空列名 | CSV 首列，表示行索引；部分读取工具显示为 `Unnamed: 0` |
| `order_id` | 答题日志的时间顺序编号 |
| `assignment_id` | 作业任务或练习任务集合的唯一编号 |
| `user_id` | 匿名学生的唯一标识符 |
| `assistment_id` | 题目母题（Assistment 单元）编号 |
| `problem_id` | 具体的题目编号（一道母题可包含主题或脚手架子题） |
| `original` | 是否为主题目（`1`=主干母题，`0`=由主题衍生出的辅助支架子题） |
| `correct` | **核心标签**：首次作答是否正确（`1`=首次尝试正确且未看提示，`0`=错误或查看了提示） |
| `attempt_count` | 该学生在该题目上提交答案的尝试次数 |
| `ms_first_response` | **首答耗时**：从题目展示给学生到学生做出首次交互（提交或点提示）的毫秒数 |
| `tutor_mode` | 辅导系统运行模式（`tutor` 交互辅导，`test` 纯测试） |
| `answer_type` | 题目作答形式（`choose_1` 单选，`algebra` 表达式输入，`open_response` 开放文本） |
| `sequence_id` | 知识体系中的题目链序列单元编号 |
| `student_class_id` | 班级唯一标识符 |
| `position` | 作业在班级作业页面中的位置 |
| `type` | 练习模式（如精通式训练 `MasterySection`） |
| `base_sequence_id` | 基础序列编号 |
| `skill_id` | 绑定的知识组件（技能/KC）唯一数值编号（部分题若未标注可能缺失） |
| `skill_name` | 知识组件的标准人类可读英文名称 |
| `teacher_id` | 授课教师编号 |
| `school_id` | 学校编号 |
| `hint_count` | 学生在该题目上请求提示的总次数 |
| `hint_total` | 系统为该题目配置的提示总级数 |
| `overlap_time` | 重叠时间，单位为毫秒 |
| `template_id` | 参数化题目生成模板编号（同一模板生成的题目共享相同解题模式） |
| `answer_id` | 单选类题目的内部选项编号 |
| `answer_text` | 学生实际提交的文本答案内容 |
| `first_action` | 首次交互类型，包括尝试作答和请求提示 |
| `bottom_hint` | 是否已经查看到底部终极提示（Bottom-out Hint，直接给出答案解析） |
| `opportunity` | 截止到当前步，该学生练习该技能的总次数机会计数 |
| `opportunity_original` | 仅计算母题练习机会的历史累计计数 |

