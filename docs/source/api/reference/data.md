# 数据处理

模型级数据基类、DataLoader 工厂、windowlate 数据集与数据源基类。各数据集处理器（`utils/data_process/<dataset>.py`）均继承 `DataSource`，按需实现 `load_src_data` / `clean_raw_data` / `transform_data`。

`utils.config.DataLoaderConfig` 定义加载参数，`utils.model_data.create_optimized_dataloader()` 负责构建 DataLoader。技能级模型通过 `SkillModelData.create_windowlate_dataloader()` 创建 windowlate 加载器，并可通过 `_create_windowlate_dataset()` 扩展模型特有的窗口特征。

```{eval-rst}
.. automodule:: utils.model_data.base_model_data
   :members:
```

```{eval-rst}
.. automodule:: utils.model_data.question_model_data
   :members:
```

```{eval-rst}
.. automodule:: utils.model_data.skill_model_data
   :members:
```

```{eval-rst}
.. automodule:: utils.model_data.loading
   :members:
```

```{eval-rst}
.. automodule:: utils.model_data.windowlate
   :members:
```

```{eval-rst}
.. automodule:: utils.data_process.data_source
   :members:
```
