# ASSISTments 2015

2015 年的 Skill Builder 数学练习数据。

数据集链接：[2015 官方数据页](https://sites.google.com/site/assistmentsdata/datasets/2015-assistments-skill-builder-data)。

## 数据参数

| 参数 | 内容 |
| --- | --- |
| 数据文件 | `2015_100_skill_builders_main_problems.csv` |
| 技能练习集合 | 作答量最高的 100 个 Skill Builder，见上述官方数据页 |
| 数据行 / 不重复作答日志 | 708,631 / 708,631，按 `log_id` 去重 |
| 学生 / 技能练习集合 | 19,917 / 100，分别按 `user_id`、`sequence_id` 去重；文件无题目标识 |

去重时排除空值及 `NA`、`nan`、`null` 标记（不区分大小写）。

## 字段

| 字段名称 | 字段语义说明 |
| :--- | :--- |
| `user_id` | 匿名学生标识符 |
| `log_id` | 答题日志编号 |
| `sequence_id` | Skill Builder 技能练习编号 |
| `correct` | **核心标签**：答题结果（`1`=正确，`0`=错误） |

