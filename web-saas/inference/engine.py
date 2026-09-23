"""模型推理引擎：checkpoint 加载与真实前向计算.

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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

# 仓库根目录入 sys.path，使 model/ 与 utils/ 可导入（推理服务独立进程运行）
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# 推理服务进程统一以仓库根为工作目录：run_config.yaml 归档的
# ``data_base_path`` 等是相对路径（如 ./data/assistments09），只有
# 从仓库根解析才与训练时一致。start.sh 从本目录启动 uvicorn，故在此切换。
os.chdir(_REPO_ROOT)

import model  # noqa: E402,F401 — 触发 trainer/model-config 注册发现
from utils.config import parse_run_archive  # noqa: E402
from utils.core import TRAINERS  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402
from utils.experiment_manager import ExperimentManager  # noqa: E402

CHECKPOINT_NAME = "best_model.pth"


def discover_model_names() -> list[str]:
    """从注册表发现模型名；发现失败时退回静态清单（仅列目录名）."""
    try:
        return sorted(TRAINERS.keys())
    except Exception:  # pragma: no cover - 注册表结构变动时的兜底
        return sorted(
            p.name
            for p in (_REPO_ROOT / "model").iterdir()
            if p.is_dir() and not p.name.startswith("_") and p.name != "layers"
        )


@dataclass
class _LoadedModel:
    """一个已加载模型的运行时句柄."""

    trainer: Any
    num_skills: int
    run_dir: Path


@dataclass
class _InferEntry:
    """``parse_run_archive`` 的入口节点（仅需 run_dir）."""

    run_dir: str


class InferenceEngine:
    """推理引擎：按需懒加载 checkpoint，持有已加载模型."""

    def __init__(self) -> None:
        """初始化模型缓存、加载锁与推理设备."""
        self._models: dict[str, _LoadedModel] = {}
        self._lock = threading.Lock()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- run 目录发现 ----------

    @staticmethod
    def _runs_root() -> Path:
        """返回 run 目录根（``UNIKT_RUNS_DIR`` 可覆盖）."""
        return Path(os.environ.get("UNIKT_RUNS_DIR", _REPO_ROOT / "runs" / "normal"))

    def _find_run_dir(self, name: str) -> Path | None:
        """定位该模型最新的含 best_model.pth 的 run 目录；无则 None."""
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
    def available_models(self) -> dict[str, dict[str, Any]]:
        """模型名 -> ``{"available", "numSkills"}`` 的清单.

        available 表示存在可加载的已训练 run 目录；numSkills 从 run 目录
        引用的数据集 metadata.json 读取，供前端校验技能 id 范围，读取
        失败时为 None（不影响可用性判断）。
        """
        result: dict[str, dict[str, Any]] = {}
        for name in discover_model_names():
            run_dir = self._find_run_dir(name)
            if run_dir is None:
                result[name] = {"available": False, "numSkills": None}
            else:
                result[name] = {
                    "available": True,
                    "numSkills": self._dataset_num_skills(run_dir),
                }
        return result

    @staticmethod
    def _dataset_num_skills(run_dir: Path) -> int | None:
        """读 run_config.yaml 指向数据集的 metadata.json 的 num_skills.

        ``data_base_path`` 是数据根目录（如 ./data），metadata 位于
        ``<data_base_path>/<dataset>/metadata.json``。
        """
        try:
            cfg = yaml.safe_load((run_dir / "run_config.yaml").read_text())
            base = Path(cfg["data"]["data_base_path"])
            if not base.is_absolute():
                base = _REPO_ROOT / base
            meta_path = base / cfg["data"]["dataset"] / "metadata.json"
            meta = yaml.safe_load(meta_path.read_text())
            return int(meta["num_skills"])
        except Exception:  # 元数据缺失不阻塞模型列表
            return None

    # ---------- 加载 ----------

    def _load_model(self, name: str) -> _LoadedModel:
        """按 evaluate.py 同款路径重建模型并载入 best_model.pth."""
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
        """取模型句柄，未加载则先加载（线程安全）."""
        with self._lock:
            if name not in self._models:
                self._models[name] = self._load_model(name)
            return self._models[name]

    # ---------- 前向 ----------

    def predict(
        self,
        model_name: str,
        questions: list[int],
        skills: list[int],
        responses: list[int],
    ) -> list[float]:
        """对作答序列做逐步掌握度预测.

        第 j 项返回值是「基于前 j+1 步作答历史，第 j+2 题答对的概率」，
        长度 = len(skills) - 1。默认适配器支持两种模型输出约定；其它
        前向签名/输出形状的模型在 ``_PREDICT_ADAPTERS`` 注册专属适配器。
        questions 当前仅作 API 契约占位（与 skills 同义），前向用 skills。
        """
        loaded = self._get(model_name)

        if any(s < 0 or s >= loaded.num_skills for s in skills):
            raise ValueError(
                f"知识点 id 必须在 [0, {loaded.num_skills}) 内（训练数据共 "
                f"{loaded.num_skills} 个知识点）"
            )

        adapter = _PREDICT_ADAPTERS.get(model_name, _default_predict)
        return adapter(loaded, skills, responses)


def _default_predict(
    loaded: _LoadedModel, skills: list[int], responses: list[int]
) -> list[float]:
    """DKT 族默认适配器：支持 [B, L]（内部已 gather）与 [B, L, C] 两种输出."""
    device = loaded.trainer.device_ or torch.device("cpu")
    seq = torch.tensor([skills], dtype=torch.long, device=device)
    resp = torch.tensor([responses], dtype=torch.long, device=device)
    mask = torch.ones_like(seq)

    with torch.inference_mode():
        out = loaded.trainer.model(seq, resp, mask)

    if out.ndim == 2:
        # 模型内部已完成下一题 gather：out[0, t] 即「前 t 步历史 →
        # 第 t 题答对概率」，位置 0 是填充 0
        return [round(float(out[0, t]), 4) for t in range(1, len(skills))]
    if out.ndim == 3:
        # 未 gather 的模型：out[0, t, skills[t+1]] 为下一题概率
        return [
            round(float(out[0, t, skills[t + 1]]), 4) for t in range(len(skills) - 1)
        ]
    raise NotImplementedError(
        f"模型输出形状 {tuple(out.shape)} 未适配，请在 engine._PREDICT_ADAPTERS"
        f" 为该模型注册专属适配器"
    )


def _sakt_style_predict(
    loaded: _LoadedModel, skills: list[int], responses: list[int]
) -> list[float]:
    """SAKT 族适配器：``forward(sequence, response)`` 输出 [B, L-1].

    位置 i 即「前 i+1 步历史 → 第 i+2 题答对概率」，与 DKT 族语义一致，
    只是少了前导填充 0。
    """
    device = loaded.trainer.device_ or torch.device("cpu")
    seq = torch.tensor([skills], dtype=torch.long, device=device)
    resp = torch.tensor([responses], dtype=torch.long, device=device)

    with torch.inference_mode():
        out = loaded.trainer.model(seq, resp)

    return [round(float(out[0, i]), 4) for i in range(len(skills) - 1)]


#: 前向适配器注册表：模型名 -> (loaded, skills, responses) -> 概率列表.
#: 未注册的模型走 ``_default_predict``（DKT 族双分支）。
PredictAdapter = Callable[[_LoadedModel, list[int], list[int]], list[float]]
_PREDICT_ADAPTERS: dict[str, PredictAdapter] = {
    "SAKT": _sakt_style_predict,
}

engine = InferenceEngine()
