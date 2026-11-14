"""Base analyzer class for case analysis.

Minimal inference-only lifecycle: the analyzer is built from an archived
RunConfig plus a trained checkpoint, runs forward passes without an
optimizer or training loop, and hands extracted case data to an
injectable :class:`~utils.case_analysis.interfaces.CaseDataSink`.

Unlike the training path it does not subclass :class:`BaseTrainer`;
shared tensor/device helpers come from
:class:`~utils.training.inference_ops.InferenceOpsMixin`.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
from rich.live import Live

from ..core import get_logger
from ..core.progress import create_progress
from ..model_data import create_optimized_dataloader
from ..training import InferenceOpsMixin
from ..training.checkpoint import CheckpointManager
from ..training.runtime_components import RuntimeComponents
from .interfaces import CaseDataSink
from .sinks.dataframe_sink import DataFrameSink

logger = get_logger(__name__)


class BaseCaseAnalyzer(InferenceOpsMixin, ABC):
    """Base class for case analysis with inference-only capabilities.

    Constructed in one template step, mirroring ``BaseTrainer``::

        analyzer = MyAnalyzer(rc, data_src, checkpoint_path, sink=...)
        result = analyzer.run_inference()

    Subclasses implement:

    - :meth:`build_components`: assemble the model and inference
      dataset from ``rc`` + ``data_src`` (model-specific tensors may be
      stored on ``self`` here).
    - :meth:`forward_pass`: model forward pass (``y_hat`` / ``y_label``
      / ``y_predict`` contract).
    - :meth:`extract_case_data`: interpret batch outputs into the sink's
      key contract.
    """

    def __init__(
        self,
        rc: Any,
        data_src: Any,
        checkpoint_path: str,
        *,
        sink: CaseDataSink | None = None,
        device: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        """Construct and build the analyzer in one step.

        Args:
            rc: RunConfig instance — typically loaded from the run's
                archived ``run_config.yaml``.
            data_src: Data source used by :meth:`build_components`.
            checkpoint_path: Path to a trained checkpoint.
            sink: Optional case data sink; defaults to
                :class:`DataFrameSink`.
            device: Optional device override; falls back to
                ``rc.general.device`` then GPU auto-detection.
            batch_size: Optional inference batch size; falls back to
                ``rc.model.batch_size``.
        """
        self.run_config = rc
        self._data_src = data_src
        self.checkpoint_path = checkpoint_path
        self._inference_device = device
        self._inference_batch_size = batch_size
        self.sink: CaseDataSink = sink if sink is not None else DataFrameSink()
        self._components = self.build_components(rc, data_src)
        self._build()

    @abstractmethod
    def build_components(self, rc: Any, data_src: Any) -> RuntimeComponents:
        """Assemble the model and inference data.

        Args:
            rc: RunConfig instance.
            data_src: Data source.

        Returns:
            RuntimeComponents with at least ``model`` and ``val_data``
            (a Dataset or ready DataLoader) set. Model-specific tensors
            (graphs, matrices) may be stored on ``self`` here; move
            them to the device in :meth:`on_device` instead.
        """
        raise NotImplementedError("Subclasses must implement build_components method")

    @abstractmethod
    def forward_pass(self, batch_data: tuple[Any, ...]) -> dict:
        """Perform a forward pass for a single batch.

        Args:
            batch_data: A batch of data from the DataLoader.

        Returns:
            Dict containing at least ``"y_hat"``, ``"y_label"``,
            ``"y_predict"``.
        """
        raise NotImplementedError("Subclasses must implement forward_pass method")

    @abstractmethod
    def extract_case_data(self, batch_data: Any, outputs: dict) -> dict:
        """Extract model-specific data from batch outputs.

        The returned dict feeds the sink. For the default
        :class:`DataFrameSink` it must contain parallel lists keyed by
        ``user_ids`` / ``question_ids`` / ``labels`` / ``predictions``,
        optionally ``skills`` / ``logits`` / ``mask`` /
        ``knowledge_states`` plus any extra keys to pass through.

        Args:
            batch_data: Raw batch data from the DataLoader.
            outputs: Output dict from forward_pass.

        Returns:
            Dict with extracted per-batch data.
        """
        raise NotImplementedError("Subclasses must implement extract_case_data method")

    def on_device(self, device: torch.device) -> None:
        """Hook: move extra model-specific tensors to ``device``.

        Called during build once the device is resolved, after
        :meth:`build_components`. The default implementation does
        nothing.
        """
        pass

    def _build(self) -> None:
        """Finalize device, loader, weights and eval mode."""
        c = self._components
        if c.model is None or c.val_data is None:
            raise ValueError(
                "build_components must set RuntimeComponents.model and .val_data."
            )

        dev = self._inference_device or self.run_config.general.device
        self.device_ = torch.device(dev) if dev else self._try_gpu()
        self.on_device(self.device_)

        self.model = c.model
        batch_size = self._inference_batch_size or self.run_config.model.batch_size
        if isinstance(c.val_data, torch.utils.data.Dataset):
            # single-run inference on small data: no multiprocessing workers,
            # which would fork and trip Python 3.12+ deadlock warnings
            self.val_data = create_optimized_dataloader(
                c.val_data,
                batch_size=batch_size,
                shuffle=False,
                device=self.device_,
                num_workers=0,
                collate_fn=c.collate_fn,
            )
        else:
            self.val_data = c.val_data

        CheckpointManager.load_weights(self.checkpoint_path, self.model, self.device_)
        self.model.to(self.device_)
        self.model.eval()
        logger.debug("Analyzer built successfully for inference")

    @torch.no_grad()
    def run_inference(self) -> Any:
        """Run inference over the configured data and collect results."""
        logger.info("Running inference...")

        progress = create_progress()
        with Live(progress):
            inference_task = progress.add_task(
                "[bold cyan]Inference", total=len(self.val_data)
            )
            for batch_data in self.val_data:
                outputs = self.forward_pass(batch_data)
                case_data = self.extract_case_data(batch_data, outputs)
                self.sink.add_batch(case_data)
                progress.advance(inference_task)

        logger.info("Inference complete")
        return self.sink.result()


__all__ = ["BaseCaseAnalyzer"]
