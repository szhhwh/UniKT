"""Question-level session packing for HiTSKT."""

from collections import Counter
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.model_data import QuestionModelData

logger = get_logger(__name__)


class HiTSKTDataset(Dataset):
    """Build source-format history and target sessions only when sampled."""

    def __init__(self, sessions, users, examples, action_size, session_size, eos):
        self.sessions = sessions
        self.users = users
        self.examples = examples
        self.action_size = action_size
        self.session_size = session_size
        self.padding = np.zeros((5, action_size), dtype=np.int64)
        self.padding[2] = 2
        self.padding[4] = 2
        self.session_eos = np.empty((5, action_size), dtype=np.int64)
        for channel, token in ((0, eos[0]), (1, eos[1]), (3, eos[2])):
            self.session_eos[channel, :-1] = token + 1
            self.session_eos[channel, -1] = token
        self.session_eos[2, :-1] = 4
        self.session_eos[2, -1] = 3
        self.session_eos[4] = self.session_eos[2]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        user_index, target_index = self.examples[index]
        user_sessions = self.users[user_index]
        history = user_sessions[
            max(0, target_index - self.session_size + 1) : target_index
        ]
        packed = np.empty((self.session_size + 1, 5, self.action_size), dtype=np.int64)
        history_count = len(history)
        for i, session_index in enumerate(history):
            packed[i] = self.sessions[session_index]
        for i in range(history_count, self.session_size - 1):
            packed[i] = self.padding
        packed[self.session_size - 1] = self.session_eos
        packed[self.session_size] = self.sessions[user_sessions[target_index]]
        labels = packed[self.session_size, 2].copy()
        valid = (labels != 2) & (labels != 3)
        return (
            torch.from_numpy(packed),
            torch.from_numpy(labels),
            torch.from_numpy(valid),
        )


class HiTSKTModelData(QuestionModelData):
    """Use framework question folds and assign one ID per skill combination."""

    @staticmethod
    def _timestamp_scale(rows, unit):
        if unit == "seconds":
            return 1
        if unit == "milliseconds":
            return 1000
        gaps = []
        previous_user = None
        previous_time = None
        for user, _, _, timestamp, *_ in rows:
            if user == previous_user and timestamp > previous_time:
                gaps.append(timestamp - previous_time)
            previous_user, previous_time = user, timestamp
        return 1000 if gaps and np.quantile(gaps, 0.9) >= 10000 else 1

    @staticmethod
    def _pack_session(events, action_size, eos):
        events = events[-(action_size - 1) :]
        packed = np.zeros((5, action_size), dtype=np.int64)
        packed[2] = 2
        packed[4] = 2
        for position, (question, skill, label, qno) in enumerate(events):
            packed[0, position] = question
            packed[1, position] = skill
            packed[2, position] = label
            packed[3, position] = qno
            packed[4, position] = 5 if position == 0 else events[position - 1][2]
        last = action_size - 1
        for channel, token in ((0, eos[0]), (1, eos[1]), (3, eos[2])):
            packed[channel, last] = token
        packed[2, last] = 3
        packed[4, len(events)] = events[-1][2]
        return packed

    @override
    def prepare_data(self, rc: Any):
        if rc.model.action_size < 2 or rc.model.session_size < 2:
            raise ValueError("HiTSKT requires action_size and session_size >= 2")
        folds = self.data_src.get_metadata("kfold_n_splits")
        if not 0 <= rc.data.fold < folds:
            raise ValueError(f"fold must be in [0, {folds})")

        data = self.load_split_frame(required=("timestamp",))
        annotations = data["skills"].to_list()
        vocabulary = {
            skills: i + 1
            for i, skills in enumerate(sorted({tuple(s) for s in annotations}))
        }
        n_combinations = len(vocabulary)
        rows = (
            data.sort(["user", "timestamp", "sequence_id", "seq_pos"])
            .select("user", "question", "label", "timestamp", "fold", "skills")
            .iter_rows()
        )
        rows = list(rows)
        scale = self._timestamp_scale(rows, rc.model.timestamp_unit)
        gap = rc.model.session_gap_hours * 3600 * scale

        question_counts = Counter()
        sessions = []
        users = []
        fold_examples = {"train": [], "val": [], "test": []}
        current_user = None
        current_fold = None
        current_events = []
        current_sessions = []
        previous_time = None
        max_qno = 0

        def finish_session():
            if current_events:
                current_sessions.append(len(sessions))
                sessions.append(
                    self._pack_session(current_events, rc.model.action_size, eos)
                )
                current_events.clear()

        def finish_user():
            finish_session()
            if not current_sessions:
                return
            user_index = len(users)
            users.append(current_sessions.copy())
            split = (
                "test"
                if current_fold == -1
                else "val"
                if current_fold == rc.data.fold
                else "train"
            )
            for target_index in range(1, len(current_sessions)):
                target = sessions[current_sessions[target_index]]
                if ((target[2] != 2) & (target[2] != 3)).sum() >= 2:
                    fold_examples[split].append((user_index, target_index))
            current_sessions.clear()

        n_questions = self.data_src.get_metadata("num_questions")
        eos = (n_questions + 1, n_combinations + 1, 0)
        for user, question, label, timestamp, fold, skills in rows:
            if label not in (0, 1):
                raise ValueError("HiTSKT requires binary labels")
            if user != current_user:
                if current_user is not None:
                    finish_user()
                current_user, current_fold = user, fold
                previous_time = None
            elif fold != current_fold:
                raise ValueError(f"User {user} has inconsistent fold labels")
            if previous_time is not None and timestamp - previous_time >= gap:
                finish_session()
            key = (user, question)
            question_counts[key] += 1
            max_qno = max(max_qno, question_counts[key])
            current_events.append(
                (
                    question + 1,
                    vocabulary[tuple(skills)],
                    label,
                    question_counts[key],
                )
            )
            previous_time = timestamp
        if current_user is not None:
            finish_user()

        eos = (n_questions + 1, n_combinations + 1, max_qno + 1)
        for packed in sessions:
            packed[3, packed[2] == 3] = eos[2]
        datasets = tuple(
            HiTSKTDataset(
                sessions,
                users,
                fold_examples[split],
                rc.model.action_size,
                rc.model.session_size,
                eos,
            )
            for split in ("train", "val", "test")
        )
        info = {
            "n_problem": n_questions + 3,
            "n_skill": n_combinations + 3,
            "n_qno": max_qno + 3,
            "timestamp_scale": scale,
        }
        logger.info(
            f"HiTSKT: {len(users)} users, {len(sessions)} sessions, {n_combinations} skill combinations, "
            f"time unit={'ms' if scale == 1000 else 's'}, samples={[len(dataset) for dataset in datasets]}"
        )
        return (*datasets, info)
