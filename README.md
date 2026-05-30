# street-perception-experiment
Experimental process for visual rhythm perception evaluation of continuous street view sequences (CLIP pairwise comparison + TrueSkill ranking)

基于连续街景序列、CLIP成对比较和TrueSkill排序的街道感知自动评估框架。

## 方法概览

- **评价单元**：50m间隔的空间序列对
- **感知量化**：CLIP成对比较 + TrueSkill相对排序（替代绝对评分，提升区分度）
- **机制解析**：随机森林/XGBoost + SHAP归因
- **设计验证**：参数化干预 → 三维渲染回测 → 统计检验

## 仓库结构

- `README.md`            # 本文件（项目总览）
- `experiment_protocol.md` # 完整实验流程文档
- `code/`                # 核心脚本（感知生成、建模、诊断、模拟干预）
- `data/`                # 示例数据（采样点、特征矩阵等）
- `figures/`             # 结果图表

## 实验流程简述

1. **S层（刺激）**：语义分割提取6项街景指标，计算相邻点间的变化量（Δ值、变异系数）。
2. **O层（机体感知）**：利用CLIP对街景图像进行成对比较，通过TrueSkill算法生成稳定的相对感知得分。
3. **归因建模**：用Δ特征预测序列感知得分，识别关键影响指标与突变阈值。
4. **循证优化**：定位低分路段，设计干预节点，通过渲染重建并验证改善效果。

## 主要结果

- 模型可解释性：机动化变异系数、绿视率变化为最关键特征。
- 干预模拟统计显著性：p < 0.001，平均得分提升显著。

## 引用

本仓库为论文实验流程补充。文献信息待发表后更新。

## 许可

MIT License


## 数据集：论文数据集为山西省运城市街景图像（百度api抓取），论文公开后上传数据集地址
