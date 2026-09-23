"""模型推理引擎：checkpoint 加载与真实前向计算。

加载路径与 ``evaluate.py`` 完全一致：run 目录里的 ``run_config.yaml``
无损重建 RunConfig → 注册表实例化 trainer → ``load_weights`` 载入
``best_model.pth``。本模块是 Python 侧唯一需要碰 ``model/`` 与
``utils/`` 的地方，Java 侧通过 HTTP 调用，不做任何张量计算。

run 目录发现顺序（每模型取最新一个含 best_model.pth 的目录）：
1. 环境变量 ``UNIKT_RUN_DIR_<MODEL>``（精确指定）
2. ``UNIKT_RUNS_DIR``（默认 ``<repo>/runs/normal``）下 ``<MODEL>_`` 前缀目录
"""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

# 仓库根目录入 sys.path，使 model/ 与 utils/ 可导入（推理服务独立进程运行）
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import model  # noqa: E402,F401 — 触发 trainer/model-config 注册发现
from utils.config import parse_run_archive  # noqa: E402
from utils.core import TRAINERS  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402
from utils.experiment_manager import ExperimentManager  # noqa: E402

CHECKPOINT_NAME = "best_model.pth"


def discover_model_names() -> list[str]:
    """从注册表发现模型名；发现失败时退回静态清单（仅列目录名）。"""
    try:
        from utils.core import TRAINERS as _t  # noqa: PLC0415

        return sorted(_t.keys())
    except Exception:  # pragma: no cover - 注册表结构变动时的兜底
        return sorted(
            p.name
            for p in (_REPO_ROOT / "model").iterdir()
            if p.is_dir() and not p.name.startswith("_") and p.name != "layers"
        )


@dataclass
class _LoadedModel:
    """一个已加载模型的运行时句柄。"""

    trainer: Any
    num_skills: int
    run_dir: Path


@dataclass
class _InferEntry:
    """``parse_run_archive`` 的入口节点（仅需 run_dir）。"""

    run_dir: str


class InferenceEngine:
    """推理引擎：按需懒加载 checkpoint，持有已加载模型。"""

    def __init__(self) -> None:
        self._models: dict[str, _LoadedModel] = {}
        self._lock = threading.Lock()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- run 目录发现 ----------

    @staticmethod
    def _runs_root() -> Path:
        return Path(os.environ.get("UNIKT_RUNS_DIR", _REPO_ROOT / "runs" / "normal"))

    def _find_run_dir(self, name: str) -> Path | None:
        exact = os.environ.get(f"UNIKT_RUN_DIR_{name.upper()}")
        if exact:
            p = Path(exact)
            return p if (p / CHECKPOINT_NAME).is_file() else None
        root = self._runs_root()
        if not root.is_dir():
            return None
        candidates = [
            d
            for d in root.iterdir()
            if d.is_dir()
            and d.name.startswith(f"{name}_")
            and (d / CHECKPOINT_NAME).is_file()
        ]
        return max(candidates, key=lambda d: d.stat().st_mtime) if candidates else None

    @property
    def available_models(self) -> dict[str, bool]:
        """模型名 -> 是否存在可加载的已训练 run 目录。"""
        return {
            name: self._find_run_dir(name) is not None
            for name in discover_model_names()
        }

    # ---------- 加载 ----------

    def _load_model(self, name: str) -> _LoadedModel:
        run_dir = self._find_run_dir(name)
        if run_dir is None:
            raise FileNotFoundError(
                f"模型 {name} 没有已训练的 run 目录（缺少 {CHECKPOINT_NAME}）。"
                f"先运行 python train.py -m {name} -d <dataset> 完成训练。"
            )

        # 与 evaluate.py 相同的重建路径：run_config.yaml -> RunConfig
        rc, _, resolved = parse_run_archive(
            ["--infer.run_dir", str(run_dir), "--general.device", self._device],
            prog="unikt-inference",
            description="UniKT SaaS inference service",
            entry_node="infer",
            entry_cls=_InferEntry,
        )
        rc.general.cloud_tracking = False
        rc.general.checkpoint_path = None  # 权重由 load_weights 手动载入
        rc.general.skip_test = True

        data_src = get_data_source(rc)
        exp_manager = ExperimentManager.from_run_dir(resolved)
        sub_manager = exp_manager.create_sub_experiment("inference")

        trainer = TRAINERS.get(name)(rc=rc, data_src=data_src, exp_manager=sub_manager)
        trainer.load_weights(str(resolved / CHECKPOINT_NAME))
        trainer.model.eval()

        metadata = data_src.get_metadata()
        return _LoadedModel(
            trainer=trainer,
            num_skills=int(metadata["num_skills"]),
            run_dir=resolved,
        )

    def _get(self, name: str) -> _LoadedModel:
        with self._lock:
            if name not in self._models:
                self._models[name] = self._load_model(name)
            return self._models[name]

    # ---------- 前向 ----------

    def predict(
        self,
        model: str,
        questions: list[int],
        skills: list[int],
        responses: list[int],
    ) -> list[float]:
        """对作答序列做逐步掌握度预测。

        前向采用 DKT 族约定：输入完整序列，位置 t 的输出预测 t+1 概率，
        取 ``out[t, skills[t+1]]`` 得到「下一题（该知识点）答对概率」，
        返回长度 = len(skills) - 1。其它前向签名的模型需要在
        ``_PREDICT_ADAPTERS`` 注册专属适配器。
        """
        loaded = self._get(model)

        if any(s < 0 or s >= loaded.num_skills for s in skills):
            raise ValueError(
                f"知识点 id 必须在 [0, {loaded.num_skills}) 内（训练数据共 "
                f"{loaded.num_skills} 个知识点）"
            )

        device = loaded.trainer.device_ or torch.device(self._device)
        seq = torch.tensor([skills], dtype=torch.long, device=device)
        resp = torch.tensor([responses], dtype=torch.long, device=device)
        mask = torch.ones_like(seq)

        with torch.inference_mode():
            out = loaded.trainer.model(seq, resp, mask)  # [1, L, num_skills]

        length = len(skills) - 1
        if out.ndim == 3:
            return [round(float(out[0, t, skills[t + 1]]), 4) for t in range(length)]
        # 某些模型只输出 [B, L]（下一题概率），此时输入约定不同，交给适配器
        raise NotImplementedError(
            f"模型 {model} 的输出形状 {tuple(out.shape)} 未适配，"
            "请在 engine._PREDICT_ADAPTERS 注册适配器"
        )


engine = InferenceEngine()
