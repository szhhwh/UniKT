"""Tests for the composed EfficiencyConfig schema: caching + stage fallback."""

import dataclasses

from utils.core import register_efficiency_stage
from utils.efficiency import config as eff_config_module
from utils.efficiency.config import (
    DefaultStageConfig,
    EfficiencyEntryConfig,
    GeneralEfficiencyConfig,
    build_efficiency_config_schema,
    get_efficiency_config_cls,
)
from utils.efficiency.stages.base import EfficiencyStage


class TestBuildEfficiencyConfigSchema:
    def test_second_call_reuses_cached_class(self, monkeypatch):
        # Force a rebuild first so the assertion covers the caching branch even
        # if an earlier import already warmed the cache; monkeypatch restores it.
        monkeypatch.setattr(eff_config_module, "_EFFICIENCY_CONFIG_CLS", None)
        first = build_efficiency_config_schema()
        assert build_efficiency_config_schema() is first
        assert get_efficiency_config_cls() is first

    def test_fields_are_entry_general_plus_registered_stages(self):
        from utils.core import get_supported_stages

        cls = get_efficiency_config_cls()
        field_names = [f.name for f in dataclasses.fields(cls)]
        assert field_names[:4] == ["run_dir", "checkpoint", "weights", "general"]
        assert set(field_names[4:]) == set(get_supported_stages())

    def test_entry_defaults_bound(self):
        cfg = get_efficiency_config_cls()()
        assert isinstance(cfg, EfficiencyEntryConfig)
        assert cfg.run_dir is None
        assert cfg.checkpoint == "best_model.pth"
        assert cfg.weights is None

    def test_general_defaults_bound(self):
        cfg = get_efficiency_config_cls()()
        assert isinstance(cfg.general, GeneralEfficiencyConfig)
        assert cfg.general.warmup_iters == 50

    def test_stage_config_cls_bound(self):
        from utils.core import get_supported_stages
        from utils.efficiency.stages.profile import ProfileStageConfig

        cls = get_efficiency_config_cls()()
        assert isinstance(cls.profile, ProfileStageConfig)
        # every registered stage declares its own config_cls in this repo
        assert all(
            not isinstance(getattr(cls, name), DefaultStageConfig)
            for name in get_supported_stages()
        )


class TestDefaultStageConfigFallback:
    def test_stage_without_config_cls_gets_default(
        self, registry_snapshot, monkeypatch
    ):
        @register_efficiency_stage("utest_nocfg_stage")
        class NoCfgStage(EfficiencyStage):
            name = "utest_nocfg_stage"

            def run(self, ctx):
                return None

            @classmethod
            def format_table(cls, result):
                return None

        monkeypatch.setattr(eff_config_module, "_EFFICIENCY_CONFIG_CLS", None)
        monkeypatch.setattr(
            "utils.core.get_supported_stages", lambda: ["utest_nocfg_stage"]
        )
        cls = build_efficiency_config_schema()
        field_names = [f.name for f in dataclasses.fields(cls)]
        assert field_names == [
            "run_dir",
            "checkpoint",
            "weights",
            "general",
            "utest_nocfg_stage",
        ]
        assert isinstance(cls().utest_nocfg_stage, DefaultStageConfig)
        # teardown: monkeypatch restores both the cache and get_supported_stages


def test_default_stage_config_is_instantiable_empty():
    assert vars(DefaultStageConfig()) == {}
