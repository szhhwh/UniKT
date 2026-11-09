---
hide-toc: true
---

# 开发者指南

通过统一的注册、配置与训练接口，扩展模型、数据源和训练流程。

- [自定义模型](customize-model.md)：实现模型、数据准备与训练器，声明配置并注册。
- [多阶段训练](multi-stage-training.md)：组织训练阶段，管理组件构建、状态传递与联合推断。
- [自定义数据集](customize-dataset.md)：接入数据源，完成实体映射、序列切分与交叉验证数据准备。
- [回调与损失](customize-callback.md)：扩展训练生命周期，接入周期性更新与自定义损失。

```{toctree}
:maxdepth: 1
:hidden:

customize-model
multi-stage-training
customize-dataset
customize-callback
```
