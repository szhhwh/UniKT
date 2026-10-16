"""DKT2 数据准备：Question 粒度序列 + 多概念折叠。"""

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.core import get_logger
from utils.model_data import QuestionModelData
from utils.model_data.packed_sequences import to_skill_combination_ids

logger = get_logger(__name__)


class DKT2ModelData(QuestionModelData):
    """DKT2 数据类。

    序列粒度为 Question。多概念题目按其概念集合整体折叠为单一新概念：对
    对交互记录中的技能集合去重，每一个独特的概念组合分配一个 id。这样
    每个题目恰好对应一个概念，便于 DKT 式的“输出全概念向量再按下一概念
    收集”预测。概念 id 采用 1-index（0 留给 padding）。
    """

    def __init__(self, data_src):
        super().__init__(data_src)

    def prepare_data(self, rc):
        data = self.load_split_data()
        sequences = self.pack_sequences(data)
        partition = sequences.partition(rc.data.fold)
        user_sequence = sequences.question
        user_response = sequences.response
        user_mask = sequences.mask
        num_questions = self.data_src.get_metadata("num_questions")

        skills_sequence, num_concepts = to_skill_combination_ids(sequences.skills)
        num_skills = num_concepts + 1

        dataset = DKT2Dataset(user_sequence, user_response, user_mask, skills_sequence)
        train_dataset, val_dataset, test_dataset = partition.apply(dataset)

        logger.info(
            f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
        )
        return train_dataset, val_dataset, test_dataset, num_skills, num_questions


class DKT2Dataset(Dataset):
    """每条样本返回 question/response/mask/skill 四元组（已折叠的概念序列）。"""

    def __init__(self, questions, responses, masks, skills):
        self.questions = torch.from_numpy(np.asarray(questions)).long()
        self.responses = torch.from_numpy(np.asarray(responses)).long()
        self.masks = torch.from_numpy(np.asarray(masks)).bool()
        self.skills = torch.from_numpy(np.asarray(skills)).long()

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return {
            "questions": self.questions[idx],
            "responses": self.responses[idx],
            "masks": self.masks[idx],
            "skills": self.skills[idx],
        }
