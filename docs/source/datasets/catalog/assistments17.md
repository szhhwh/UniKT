# ASSISTments 2017

2017 年 ASSISTments 纵向数据挖掘竞赛数据。

数据集链接：[2017 竞赛数据页](https://sites.google.com/view/assistmentsdatamining/dataset)。须填写申请表，数据链接通过邮件发送；使用须同意数据使用条款。

## 数据参数

| 参数 | 内容 |
| --- | --- |
| 数据文件 | `anonymized_full_release_competition_dataset.csv` |
| 数据行 | 942,816 |
| 学生 / 题目 / 技能 | 1,709 / 3,162 / 102，分别按 `studentId`、`problemId`、`skill` 去重 |

去重时排除空值及 `NA`、`nan`、`null` 标记（不区分大小写）。

## 字段

下表列出 `anonymized_full_release_competition_dataset.csv` 的全部字段。FR 指首次响应，KC 指知识组件；`—` 表示字段含义未知。

| 字段 | 含义 |
| --- | --- |
| `studentId` | 学生标识 |
| `MiddleSchoolId` | — |
| `InferredGender` | 学生的推测性别，取值包括 `Male`、`Female`，部分记录为空 |
| `SY ASSISTments Usage` | 学生使用系统的学年 |
| `AveKnow` | 依据贝叶斯知识追踪模型估计的学生平均知识水平 |
| `AveCarelessness` | 学生平均粗心程度的模型估计值 |
| `AveCorrect` | 学生平均正确率 |
| `NumActions` | 学生在系统中的操作总数 |
| `AveResBored` | 学生平均厌烦状态估计值 |
| `AveResEngcon` | 学生平均投入专注状态估计值 |
| `AveResConf` | 学生平均困惑状态估计值 |
| `AveResFrust` | 学生平均受挫状态估计值 |
| `AveResOfftask` | 学生平均偏离任务状态估计值 |
| `AveResGaming` | 学生平均钻系统规则空子行为估计值 |
| `action_num` | 本文件中每行均不重复的数值 |
| `skill` | 题目关联的技能标签 |
| `problemId` | 题目标识 |
| `problemType` | 题目类型 |
| `assignmentId` | 作业标识 |
| `assistmentId` | Assistment 母题标识 |
| `startTime` | 题目开始时间，Unix 时间戳，单位为秒 |
| `endTime` | 题目结束时间，Unix 时间戳，单位为秒 |
| `timeTaken` | 当前步骤所花时间 |
| `correct` | 作答是否正确 |
| `original` | 是否为题目而非支架题 |
| `hint` | 操作是否为提示响应 |
| `hintCount` | 截至当前的提示请求次数 |
| `hintTotal` | 该题的提示请求总次数 |
| `scaffold` | 是否为支架题 |
| `bottomHint` | 是否使用最终提示 |
| `attemptCount` | 截至当前在系统中尝试的题目总数 |
| `frIsHelpRequest` | 首次响应是否为帮助请求 |
| `frPast5HelpRequest` | 最近 5 次首次响应中包含帮助请求的次数 |
| `frPast8HelpRequest` | 最近 8 次首次响应中包含帮助请求的次数 |
| `stlHintUsed` | 是否使用倒数第二级提示 |
| `past8BottomOut` | 最近 8 道题中使用最终提示的题目数 |
| `totalFrPercentPastWrong` | 此 KC 历史题目中的错误比例 |
| `totalFrPastWrongCount` | 截至当前首次响应错误的总次数 |
| `frPast5WrongCount` | 最近 5 次首次响应中的错误次数 |
| `frPast8WrongCount` | 最近 8 次首次响应中的错误次数 |
| `totalFrTimeOnSkill` | 所有题目中，此 KC 的首次响应时间总和 |
| `timeSinceSkill` | 距上次遇到当前 KC 的时间 |
| `frWorkingInSchool` | 首次响应是否发生在上课时段，即早上 7 点至下午 3 点 |
| `totalFrAttempted` | 截至当前在系统中尝试的首次响应总数 |
| `totalFrSkillOpportunities` | 截至当前此 KC 的首次响应练习机会总数 |
| `responseIsFillIn` | 是否为无选项列表的填空式响应 |
| `responseIsChosen` | 是否从选项列表选择响应 |
| `endsWithScaffolding` | 题目是否以支架结束 |
| `endsWithAutoScaffolding` | 题目是否以自动支架结束 |
| `frTimeTakenOnScaffolding` | 支架题的首次响应耗时 |
| `frTotalSkillOpportunitiesScaffolding` | 截至当前此技能的首次响应练习机会总数 |
| `totalFrSkillOpportunitiesByScaffolding` | 截至当前此 KC 的首次响应支架练习机会总数 |
| `frIsHelpRequestScaffolding` | 支架题的首次响应是否为帮助请求 |
| `timeGreater5Secprev2wrong` | 连续两次答错后的长时间停顿标志 |
| `sumRight` | — |
| `helpAccessUnder2Sec` | 使用帮助所花时间是否少于 2 秒 |
| `timeGreater10SecAndNextActionRight` | — |
| `consecutiveErrorsInRow` | 所有题目中连续两次答错的总次数 |
| `sumTime3SDWhen3RowRight` | — |
| `sumTimePerSkill` | — |
| `totalTimeByPercentCorrectForskill` | 此 KC 的总耗时除以该 KC 的正确比例 |
| `Prev5count` | — |
| `timeOver80` | — |
| `manywrong` | — |
| `confidence(BORED)` | 厌烦状态预测置信度 |
| `confidence(CONCENTRATING)` | 专注状态预测置信度 |
| `confidence(CONFUSED)` | 困惑状态预测置信度 |
| `confidence(FRUSTRATED)` | 受挫状态预测置信度 |
| `confidence(OFF TASK)` | 偏离任务状态预测置信度 |
| `confidence(GAMING)` | 钻系统规则空子行为预测置信度 |
| `RES_BORED` | 重标度后的厌烦状态预测置信度 |
| `RES_CONCENTRATING` | 重标度后的专注状态预测置信度 |
| `RES_CONFUSED` | 重标度后的困惑状态预测置信度 |
| `RES_FRUSTRATED` | 重标度后的受挫状态预测置信度 |
| `RES_OFFTASK` | 重标度后的偏离任务状态预测置信度 |
| `RES_GAMING` | 重标度后的钻系统规则空子行为预测置信度 |
| `Ln-1` | 贝叶斯知识追踪模型在上一个时间步的知识估计值 |
| `Ln` | 贝叶斯知识追踪模型在当前时间步的知识估计值 |
| `MCAS` | 当年马萨诸塞州综合评估考试成绩；`-999` 表示缺失 |
| `Enrolled` | — |
| `Selective` | — |
| `isSTEM` | — |
