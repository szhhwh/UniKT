# Bridge to Algebra 2006–2007

KDD Cup 2010 教育数据挖掘竞赛的开发数据集，记录代数辅导系统中的学生解题步骤。

数据集链接：[KDD Cup 2010 数据下载页](https://pslcdatashop.web.cmu.edu/KDDCup/downloads.jsp)，下载需要登录。

## 数据参数

| 参数 | 内容 |
| --- | --- |
| 数据文件 | `bridge_to_algebra_2006_2007_train.txt` |
| 步骤记录 | 3,679,199 行 |
| 学生 / 题目名称 / 知识组件 | 1,146 / 19,355 / 493，分别按 `Anon Student Id`、`Problem Name` 和 `KC(SubSkills)` 去重 |
| 缺失知识组件的行数 | 1,861,723 |

去重时排除空值及 `NA`、`nan`、`null` 标记（不区分大小写）；知识组件按 `~~` 拆分，统计范围仅为训练文件。

## 字段

竞赛数据为制表符分隔格式，每行对应一个解题步骤。以下是字段说明；该数据集使用 `SubSkills` 认知模型。详见 [DataShop 数据格式说明](https://pslcdatashop.web.cmu.edu/KDDCup/rules_data_format.jsp)。

| 字段名称 | 字段语义说明 |
| :--- | :--- |
| `Row` | 数据行的物理递增序号 |
| `Anon Student Id` | 全局匿名的学生唯一标识符 |
| `Problem Hierarchy` | 课程单元与章节层级目录（用于组织题目上下文） |
| `Problem Name` | 完整的母题名称 |
| `Problem View` | 该学生第几次看到该题目（用于区分重做与复习） |
| `Step Name` | 母题内部具体的解题步骤标识符 |
| `Step Start Time` | 步骤开始时间，可为空 |
| `First Transaction Time` | 学生的首次操作（提交答案或请求提示）时间戳 |
| `Correct Transaction Time` | 学生最终正确完成该步骤的时间戳（若未做对可为空） |
| `Step End Time` | 该步骤最后一次操作的时间 |
| `Step Duration (sec)` | 分配至该步骤的各次操作时长之和，单位为秒；步骤开始时间缺失时可为空 |
| `Correct Step Duration (sec)` | 首次尝试正确时的步骤耗时（秒） |
| `Error Step Duration (sec)` | 首次尝试错误或请求提示时的步骤耗时（秒） |
| `Correct First Attempt` | **核心标签**：首次尝试是否正确（1=首次做对，0=发生错误或查看提示） |
| `Incorrects` | 在该步骤上的错误作答次数计数 |
| `Hints` | 在该步骤上请求提示的次数计数 |
| `Corrects` | 在该步骤上的正确作答次数计数 |
| `KC(SubSkills)` | 认知模型标注的知识组件（技能名称），多技能时用 `~~` 分隔 |
| `Opportunity(SubSkills)` | 该学生练习该知识组件的累计机会次数 |
