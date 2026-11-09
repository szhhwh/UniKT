# Practice Anatomy

Practice Anatomy 是 Masaryk 大学自适应学习研究组在 practiceanatomy.com 上收集的人体解剖学术语练习数据，静态快照覆盖 2015 年 11 月 16 日至 2017 年 1 月 3 日。

## 数据集链接

- 官方入口：[Practice Anatomy Data](http://data.practiceanatomy.com/)
- 在线系统：[practiceanatomy.com](https://practiceanatomy.com/)

## 数据规模

| 指标 | 数值 | 口径 |
| --- | --- | --- |
| 学生数 | 18,563 | 官方数据说明 |
| 解剖条目数 | 2,894 | 官方数据说明 |
| 作答记录数 | 1,182,065 | 官方数据说明 |

数据集仅包含免费内容的交互记录，付费内容交互已排除。系统提供捷克语与英语界面，约 84% 的用户来自捷克，11% 来自斯洛伐克。

## 字段

发布包包含一个逗号分隔的 `answers.csv`，共 19 个字段。

| 字段名称 | 数据类型 | 说明 |
| --- | --- | --- |
| `id` | Integer | 作答记录编号 |
| `user` | Integer | 学生编号 |
| `item_asked` | Integer | 被提问的解剖条目编号 |
| `term_name_asked` | String | 被提问的解剖条目名称 |
| `item_answered` | Integer | 学生回答的解剖条目编号；回答“不知道”时为空 |
| `term_name_answered` | String | 学生回答的解剖条目名称 |
| `context_name` | String | 图像或上下文名称 |
| `type` | String | 作答方向，`t2d` 为看词定位，`d2t` 为看图选词 |
| `options` | Integer | 选项数量，包含被提问条目 |
| `time` | Datetime | 作答记录写入系统的时间 |
| `response_time` | Integer | 作答耗时，单位毫秒 |
| `ip_country` | String | 客户端 IP 所在国家 |
| `ip_id` | Integer | 匿名 IP 标识 |
| `lang` | String | 术语语言，取值 `cs`、`la (cs)`、`en`、`la (en)` |
| `locations_asked` | String | 被提问条目的身体位置，JSON |
| `systems_asked` | String | 被提问条目所属器官系统，JSON |
| `locations_answered` | String | 回答条目的身体位置，JSON |
| `systems_anwered` | String | 回答条目所属器官系统，JSON |
| `practice_filter` | String | 用户使用的练习过滤器，JSON |
