"""Question-level sequences and concept initialization for CDFKT."""

import polars as pl
import torch
from torch.utils.data import Dataset

from utils.core import get_logger
from utils.model_data import QuestionModelData

logger = get_logger(__name__)


class CDFKTDataset(Dataset):
    def __init__(self, questions, responses, masks):
        self.questions = torch.from_numpy(questions).long()
        self.responses = torch.from_numpy(responses).long()
        self.masks = torch.from_numpy(masks).bool()

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return self.questions[idx], self.responses[idx], self.masks[idx]


class CDFKTModelData(QuestionModelData):
    def prepare_data(self, rc):
        sequences = self.load_sequences()
        dataset = CDFKTDataset(sequences.question, sequences.response, sequences.mask)
        train, val, test = sequences.partition(rc.data.fold).apply(dataset)

        # Concept annotations initialize question embeddings without response statistics.
        concepts = (
            self.data_src.get_relation("question_skill")
            .group_by("question")
            .agg(pl.col("skill").sort())
        )
        self.question_concept_map = torch.full(
            (
                self.data_src.get_metadata("num_questions"),
                concepts["skill"].list.len().max(),
            ),
            -1,
            dtype=torch.long,
        )
        for question, skills in concepts.iter_rows():
            self.question_concept_map[question, : len(skills)] = torch.tensor(
                skills, dtype=torch.long
            )

        logger.info(
            f"CDFKT data prepared: train={len(train)}, val={len(val)}, test={len(test)}"
        )
        return train, val, test
