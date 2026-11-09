# XES3G5M

XES3G5M 是采集自好未来学而思三年级数学课程的知识追踪基准数据集，提供题目与知识概念两级序列以及题目文本、解析和图片等辅助信息。

## 数据集链接

- 官方仓库：[ai4ed/XES3G5M](https://github.com/ai4ed/XES3G5M)
- 下载入口：[XES3G5M 数据包](https://drive.google.com/file/d/1eFiIYyh5O2V90RA0brammGH6EpHvPDQe/view)
- 赛事说明：[AAAI2023 Global Knowledge Tracing Challenge](https://ai4ed.cc/competitions/aaai2023competition)

## 数据规模与形态

| 指标 | 规模 | 说明 |
| :--- | :--- | :--- |
| 学生数 | 18,066 | 训练验证集 14,453 名，独立测试集 3,613 名 |
| 题目数 | 7,652 | `metadata/questions.json` 收录的有效题目 |
| 知识概念路径数 | 1,175 | `metadata/kc_routes_map.json` 映射的标准知识点路径 |
| 题目级序列数 | 34,578 | 训练验证窗口 30,965 条，测试集 3,613 条 |
| 概念级序列数 | 37,010 | 训练验证窗口 33,397 条，测试集 3,613 条 |

:::{note}
XES3G5M 源自 AAAI 2023 全球知识追踪挑战赛，官方发布包专为序列化建模设计，已预先按最大长度 200 进行窗口切分。UniKT 默认基于题目级序列 `question_level` 进行加载与评估。
:::

## 发布包字段

以下字段按发布包各 CSV 的实际表头分组。

### 所有序列文件共有的基础列

| 字段名称 | 数据类型 | 说明 |
| --- | --- | --- |
| `fold` | Integer | 训练验证集折号，取值 0 至 4；测试集为 -1 |
| `uid` | Integer | 学生编号 |
| `questions` | String | 题目编号序列，逗号分隔 |
| `concepts` | String | 知识概念编号序列，逗号分隔；题目级文件中多个概念以下划线分隔 |
| `responses` | String | 作答正误序列，1 为正确，0 为错误 |
| `timestamps` | String | 作答开始时间戳序列，毫秒级 |

### 按文件附加的列

| 字段名称 | 出现文件 | 说明 |
| --- | --- | --- |
| `selectmasks` | 题目级训练与窗口测试文件、`kc_level/train_valid_sequences.csv`、`kc_level/test_question_window_sequences.csv` | 批处理有效位掩码；-1 表示填充位置或仅作为历史、不参与损失与评估的交互 |
| `is_repeat` | `kc_level/train_valid_sequences.csv`、`kc_level/test.csv`、`kc_level/test_question_window_sequences.csv` | 1 表示当前概念与上一个概念属于同一道题，0 表示新题目 |
| `cidxs` | `kc_level/test.csv` | 知识概念级测试集中的交互索引序列 |
| `qidxs` | `kc_level/test_question_window_sequences.csv` | 题目级测试集中的交互索引序列 |
| `rest` | `kc_level/test_question_window_sequences.csv` | 该交互在原序列中剩余的知识概念数量，用于预测融合 |
| `orirow` | `kc_level/test_question_window_sequences.csv` | 该交互在测试集中的行号 |

题目级测试文件 `question_level/test_quelevel.csv` 仅包含基础列，不含 `selectmasks`、`is_repeat`、`cidxs`、`qidxs`、`rest`、`orirow`。

### 题目元数据（`metadata/questions.json`）

| 字段名称 | 数据类型 | 说明 |
| --- | --- | --- |
| `content` | String | 题干文本，保留 LaTeX |
| `kc_routes` | String | 知识概念路径文本 |
| `answer` | String | 标准答案 |
| `options` | List[String] | 选择题选项，填空题可为空 |
| `analysis` | String | 解析文本，保留 LaTeX |
| `type` | Integer | 题型，0 为填空题，1 为选择题 |

`metadata/kc_routes_map.json` 记录知识概念编号与大纲路径文本的映射；`metadata/embeddings/` 与 `metadata/images/` 分别提供题目、解析、知识概念的向量表示和题目图片。
