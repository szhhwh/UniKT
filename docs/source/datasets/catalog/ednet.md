# EdNet-KT1

EdNet 是韩国人工智能辅导平台 Santa 收集的英语 TOEIC 学习交互数据集，此处收录其专注单题作答记录的 KT1 层。

## 数据集链接

- 官方仓库：[riiid/ednet](https://github.com/riiid/ednet)
- 数据下载：[EdNet-KT1](http://bit.ly/ednet_kt1)、[EdNet-Contents](http://bit.ly/ednet-content)

## 数据规模

| 指标 | 规模 | 说明 |
| :--- | :--- | :--- |
| 学生数 | 784,309 | KT1 目录下独立学生数据文件总数，每名学生一个 CSV |
| 题目数 | 13,169 | `EdNet-Contents/questions.csv` 收录的有效题目总数 |
| 题目知识标签数 | 189 | 题目关联的专家标注知识点标签类型数 |
| 答题交互总数 | 95,294,926 | 全量 KT1 学生日志记录的单题作答交互总数 |

:::{note}
EdNet 数据集定义了 KT1 至 KT4 四个交互层级。UniKT 专注于知识追踪任务，收录并处理其核心做题层 KT1。
:::

## 字段说明

### 学生作答日志 `KT1/u{user_id}.csv`

| 字段名称 | 数据类型 | 说明 |
| :--- | :--- | :--- |
| `timestamp` | Integer | 题目下发时间，Unix 毫秒时间戳，经过官方固定时间偏移处理 |
| `solving_id` | Integer | 学习会话编号，对应一次题组作答，从 1 开始递增 |
| `question_id` | String | 题目编号，格式如 `q1` |
| `user_answer` | Char | 学生提交的答案选项，取值 a 至 d |
| `elapsed_time` | Float | 该题作答耗时，单位为毫秒 |

### 题目元数据 `EdNet-Contents/questions.csv`

| 字段名称 | 数据类型 | 说明 |
| :--- | :--- | :--- |
| `question_id` | String | 题目编号，格式如 `q1` |
| `bundle_id` | String | 题组编号，格式如 `b1`，一个题组包含 1 至 5 道题 |
| `explanation_id` | String | 解析文本资源编号，格式如 `e1`，数值通常与题组编号一致 |
| `correct_answer` | Char | 标准答案选项，取值 a 至 d |
| `part` | Integer | TOEIC 测试部分，取值为 1 至 7 |
| `tags` | String | 专家标注的知识标签编号，多个标签以分号分隔 |
| `deployed_at` | Integer | 题目上线时间，Unix 毫秒时间戳 |
