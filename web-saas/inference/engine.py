"""模型推理引擎：负责 checkpoint 加载与前向计算。

这是 Python 侧唯一需要碰 ``model/`` 与 ``utils/`` 的地方。SpringBoot 后端
通过 HTTP 调用本模块暴露的接口，Java 侧不做任何张量计算。

当前状态：脚手架。``predict`` 先返回计数基线（答对率），保证链路可跑通；
接入真实模型时替换 ``_load_model`` 与 ``predict`` 即可，接口不用动。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# 仓库根目录入 sys.path，使 model/ 与 utils/ 可导入（推理服务独立进程运行）
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def discover_model_names() -> list[str]:
    """从注册表发现模型名；发现失败时退回静态清单（仅列目录名）。"""
    try:
        from model import TRAINERS  # noqa: PLC0415 运行时导入，避免启动即扫描

        return sorted(TRAINERS.keys())
    except Exception:  # pragma: no cover - 注册表结构变动时的兜底
        return sorted(
            p.name
            for p in (_REPO_ROOT / "model").iterdir()
            if p.is_dir() and not p.name.startswith("_") and p.name != "layers"
        )


class InferenceEngine:
    """单例推理引擎：持有已加载的模型，按需懒加载 checkpoint。"""

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}

    @property
    def available_models(self) -> dict[str, bool]:
        """模型名 -> 是否已加载（当前全部未加载，接入 checkpoint 后生效）。"""
        return {name: name in self._models for name in discover_model_names()}

    def _load_model(self, name: str) -> Any:
        """加载指定模型的 checkpoint。

        TODO(推理负责人):
          1. 从 configs/ 读模型超参构造模型 (TRAINERS.get(name))
          2. utils/training/checkpoint.py 加载 weights
          3. 存入 self._models 缓存
        """
        raise NotImplementedError(f"模型 {name} 的加载逻辑尚未接入（脚手架阶段）")

    def predict(
        self,
        model: str,
        questions: list[int],
        skills: list[int],
        responses: list[int],
    ) -> list[float]:
        """对作答序列做逐步掌握度预测，返回长度 len-1 的概率列表。

        脚手架实现：滑动计数基线（前 i 步答对率）。链路验证用，非真实预测。
        checkpoint 接入后，_load_model 抛出的 NotImplementedError 会被替换为
        真实前向计算，此处 try/except 即可移除。
        """
        if model not in self._models:
            try:
                self._models[model] = self._load_model(model)
            except NotImplementedError:
                pass  # 脚手架阶段：未接入 checkpoint，退回基线

        out: list[float] = []
        correct = 0
        for i in range(len(questions) - 1):
            correct += responses[i]
            out.append(round(correct / (i + 1), 4))
        return out


engine = InferenceEngine()
