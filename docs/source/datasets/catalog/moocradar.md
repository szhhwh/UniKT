# MOOC-Radar

MOOC-Radar 是清华大学知识工程实验室基于学堂在线（XuetangX）MOOC 练习数据构建的细粒度知识概念与认知标签数据集。

## 数据集链接

- 官方仓库：[THU-KEG/MOOC-Radar](https://github.com/THU-KEG/MOOC-Radar)
- 数据下载：[MoocRadar_Raw](https://bhpan.buaa.edu.cn/link/AAE5A458AFE444474EBEC928F98986E5B8)

## 数据规模

### 官方 README 整体数据

| 指标 | 数值 | 口径 |
| --- | --- | --- |
| 练习题 | 2,513 | 官方 README 初始版本整体口径 |
| 学生数 | 14,226 | 官方 README 整体口径 |
| 行为记录 | 超过 1,200 万 | 官方 README 整体口径，并非下方细粒度序列文件的作答行数 |
| 细粒度知识概念 | 5,600 | 官方 README 整体口径 |

官方 README 的 News 标注练习题数量已扩展到 9,384 道。发布包按 Coarse、Middle、Fine 三种粒度提供，本页字段对应细粒度文件。

### 细粒度文件统计

| 指标 | 数值 | 口径 |
| --- | --- | --- |
| `problem.json` 题目对象 | 9,383 | 文件行数与 `problem_id` 去重计数均为 9,383 |
| `student-problem-fine.json` 学生序列 | 14,224 | 序列对象数 |
| 序列事件 | 898,933 | 所有 `seq` 数组元素总数 |
| 序列涉及题目 | 2,513 | `seq` 中 `problem_id` 去重计数 |

## 字段

### 题目元数据（`problem.json`）

该文件为 JSON Lines 格式，每行一个题目对象。

| 字段名称 | 数据类型 | 说明 |
| --- | --- | --- |
| `problem_id` | String | 题目唯一标识，格式为 `Pm_{integer}` |
| `exercise_id` | String | 所属练习集合编号，格式为 `Ex_{integer}` |
| `course_id` | String | 所属课程编号，格式为 `C_{integer}` |
| `detail` | String | 题目详情，内容为字符串化的字典，含题干、选项、答案、题型、位置与语言等键 |
| `knowledge_type` | Integer | 知识分类代码 |
| `cognitive_dimension` | Integer | 布鲁姆认知维度代码 |
| `concepts` | List[String] | 专家标注的知识概念列表 |

### 学生行为序列（`student-problem-fine.json`）

该文件为列表，每个元素表示一名学生的行为序列，序列位于 `seq` 数组；数组元素字段如下。

| 字段名称 | 数据类型 | 说明 |
| --- | --- | --- |
| `user_id` | String | 学生唯一标识，格式为 `U_{integer}` |
| `log_id` | String | 做题日志标识 |
| `problem_id` | String | 题目唯一标识 |
| `is_correct` | Integer | 是否答对，1 为正确，0 为错误 |
| `attempts` | Integer | 作答尝试次数 |
| `score` | Float | 题目得分率，可为空 |
| `submit_time` | Datetime | 作答提交时间 |
