# Junyi 2015

Junyi 2015 是均一教育平台（Junyi Academy）2012 年 10 月至 2015 年 1 月的 K-12 数学练习日志。

## 数据集链接

- 官方入口：[PSLC DataShop #1198](https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=1198)
- DataShop 文件页：[Files #1198](https://pslcdatashop.web.cmu.edu/Files?datasetId=1198)

## 数据规模

下表统计与下方字段均来自 `junyi_ProblemLog_original.csv` 与 `junyi_Exercise_table.csv`。

| 指标 | 数值 | 口径 |
| --- | --- | --- |
| 练习日志记录 | 25,925,992 | `junyi_ProblemLog_original.csv` 数据行数 |
| 学生数 | 247,606 | 日志中 `user_id` 去重计数 |
| 练习条目数 | 722 | 日志中 `exercise` 去重计数 |
| 练习元数据条目数 | 837 | `junyi_Exercise_table.csv` 数据行数 |
| 日志时间范围 | 2012-10-12 至 2015-01-11 | 日志中 `time_done` 微秒时间戳换算 |

## 字段

### 练习日志（`junyi_ProblemLog_original.csv`，2012-10-12 至 2015-01-11）

| 字段名称 | 数据类型 | 说明 |
| --- | --- | --- |
| `user_id` | Integer | 学生编号 |
| `exercise` | String | 练习名称，与练习元数据表的 `name` 对应 |
| `problem_type` | String | 题目模板标识；部分练习记录当次遇到的模板，其余取值为 `0` |
| `problem_number` | Integer | 该学生第几次练习该练习 |
| `topic_mode` | Boolean | 是否通过课程主题入口进入练习；该功能已关闭 |
| `suggested` | Boolean | 是否由系统依据知识图谱先修关系推荐 |
| `review_mode` | Boolean | 是否在获得熟练度后进行复习 |
| `time_done` | Integer | 练习完成时间，Unix 微秒时间戳 |
| `time_taken` | Integer | 该次练习总耗时，单位秒 |
| `time_taken_attempts` | String | 各次作答尝试的耗时，单位秒，以 `&` 分隔 |
| `correct` | Boolean | 首次作答是否正确；若请求过提示则为 `false` |
| `count_attempts` | Integer | 作答尝试次数 |
| `hint_used` | Boolean | 是否请求提示 |
| `count_hints` | Integer | 请求提示次数 |
| `hint_time_taken_list` | String | 每次提示的耗时，单位秒，以 `&` 分隔；未请求提示时为空 |
| `earned_proficiency` | Boolean | 是否达到熟练度 |
| `points_earned` | Integer | 本次练习获得的积分 |

### 练习元数据（`junyi_Exercise_table.csv`）

| 字段名称 | 数据类型 | 说明 |
| --- | --- | --- |
| `name` | String | 练习名称，同时作为练习唯一标识 |
| `live` | Boolean | 该练习是否可访问 |
| `prerequisites` | String | 先修练习名称，可为空 |
| `h_position` | Float | 知识图谱中的横坐标 |
| `v_position` | Float | 知识图谱中的纵坐标 |
| `creation_date` | Datetime | 练习创建时间 |
| `seconds_per_fast_problem` | Integer | 判定“快速完成”的阈值，单位秒 |
| `pretty_display_name` | String | 知识图谱中显示的中文名称 |
| `short_display_name` | String | 中文简称 |
| `topic` | String | 所属主题 |
| `area` | String | 所属领域 |
