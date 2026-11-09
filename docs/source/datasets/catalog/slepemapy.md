# Slepemapy

Slepemapy 是 Masaryk 大学自适应学习研究组在 slepemapy.cz 上收集的地理事实练习数据，首个公开静态快照截至 2015 年 5 月 21 日。

## 数据集链接

- 官方数据目录：[Adaptive Learning Data - Slepemapy](https://www.fi.muni.cz/adaptivelearning/data/slepemapy/)
- 快照下载：[2015-05-21.zip](https://www.fi.muni.cz/adaptivelearning/data/slepemapy/2015-05-21.zip)

## 数据规模

| 指标 | 数值 | 口径 |
| --- | --- | --- |
| 学生数 | 91,331 | 官方发布包说明：2015-05-21 静态快照 |
| 地理条目数 | 1,459 | 同上 |
| 作答记录数 | 10,087,306 | 同上 |

官方数据目录另提供 `2015-ab-random-parts.zip` 与 `2016-ab-target-difficulty.zip`，属于不同实验版本，统计口径与上述数字不同，不可混用。

## 字段

发布包包含 `answer.csv`、`place.csv`、`place_type.csv` 三个分号分隔文件。

### 作答记录（`answer.csv`）

| 字段名称 | 数据类型 | 说明 |
| --- | --- | --- |
| `id` | Integer | 作答记录编号 |
| `user` | Integer | 学生编号 |
| `place_asked` | Integer | 被提问的地理条目编号 |
| `place_answered` | Integer | 学生回答的地理条目编号；回答“不知道”时为空 |
| `type` | Integer | 题型，1 为在地图上指出给定地点，2 为为高亮地点选择名称 |
| `inserted` | Datetime | 作答记录写入系统的时间 |
| `response_time` | Integer | 作答耗时，单位毫秒 |
| `place_map` | Integer | 题目所用地图对应的地理条目编号 |
| `language` | Integer | 界面语言，0 为捷克语，1 为英语，2 为西班牙语 |
| `options` | String | 候选项的地理条目编号列表，包含被提问条目 |
| `ip_country` | String | 客户端 IP 所在国家 |
| `ip_id` | Integer | 匿名 IP 标识 |

### 地理条目（`place.csv`）

| 字段名称 | 数据类型 | 说明 |
| --- | --- | --- |
| `id` | Integer | 地理条目编号 |
| `code` | String | 地理条目代码，尽可能使用 ISO 3166-1 alpha-2 |
| `name` | String | 地理条目名称 |
| `type` | Integer | 类型编号，对应 `place_type.csv` 的 `id` |
| `maps` | String | 出现该条目的地图编号列表 |

### 地理条目类型（`place_type.csv`）

| 字段名称 | 数据类型 | 说明 |
| --- | --- | --- |
| `id` | Integer | 类型编号 |
| `name` | String | 类型名称，如 country、city、continent |
