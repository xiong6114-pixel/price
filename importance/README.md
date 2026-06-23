# Importance Folder

这个文件夹整理的是“论文当前主线实验实际使用到的代码”，并保留了原始目录结构，方便后续：

- 单独打包归档
- 发给别人复现
- 和主仓库对照定位

## 当前主结果

当前论文主版本是：

- `MA-TransA3C-main = v3.1b-30ep`

当前主线代码主要集中在：

- `case_studies/power/ev_public_charging_case/train_rllib.py`
- `case_studies/power/ev_public_charging_case/compare_baselines.py`
- `case_studies/power/ev_public_charging_case/envs/`
- `case_studies/power/ev_public_charging_case/features/`
- `case_studies/power/ev_public_charging_case/agents/`
- `case_studies/power/ev_public_charging_case/policies/`

## 同时整理进来的依赖

为了保证这些脚本可以独立理解和迁移，这里也一并保留了当前实验链路真正依赖的 `heron` 最小兼容层代码：

- `heron/core/`
- `heron/agents/`
- `heron/envs/`
- `heron/protocols/`
- `heron/scheduling/`
- `heron/utils/`

## 当前没有放进来的内容

为了让 `importance` 更干净，这里没有额外放入以下内容：

- `outputs/` 实验结果文件
- 与当前论文主线无关的缓存文件、`__pycache__`
- 当前主线未直接使用的额外脚本

## 说明

这不是整个仓库的完整镜像，而是“当前论文代码主线”的整理副本。
