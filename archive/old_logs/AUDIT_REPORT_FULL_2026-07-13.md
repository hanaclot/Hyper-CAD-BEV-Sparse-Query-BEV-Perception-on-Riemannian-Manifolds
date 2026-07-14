# ============================================================================
# Hyper-CAD-BEV v6.5-Sparse: 四维深度审计报告
# 审计日期: 2026-07-13
# 审计范围: E:\Hyper-CAD-BEV-Experiments\ 全项目
# ============================================================================

## 审计总览

| 维度 | 评分 | 结论 |
|------|------|------|
| 数据真实性 | ⚠️ 60% | 3/12文件夹含真数据，其余为元数据 |
| 实验架构 | ✅ 85% | 核心模型完整，但实验脚本未调用模型 |
| 交付物完整性 | ✅ 80% | 10/10 CSV + FIG4/5 可用，FIG1/2/3缺失 |
| 文档可读性 | ❌ 20% | 无README/LICENSE/requirements，文件混乱 |

---

## 维度1: 数据文件夹真实性

### 含真实点云数据 ✅
| 文件夹 | 内容 | 规模 | 可实验 |
|--------|------|------|--------|
| semantickitti_official | 23201 .label + dataset/ 472文件(.bin) | 19.4 GB | ✅ v13使用中 |
| nuscenes | LIDAR_TOP 404 .pcd.bin | 9.1 GB | ⚠️ PCD reader失败 |
| kitti_raw/extracted | 108 .bin + 432 .png + calib | ~450 MB | ❌ 未在实验中使用 |

### 仅含元数据 ❌
| 文件夹 | 文件数 | 实际内容 |
|--------|--------|----------|
| rellis3d | 5 | HTML+JSON+README |
| tartandrive2 | 6 | HTML+JSON |
| weather_real | 7 | JSON+HTML |
| event_camera | 5 | HTML+残缺.zip.part |
| waymo | 3 | HTML+JSON |
| weather | 4 | JSON |

### 混合内容 ⚠️
| 文件夹 | 文件数 | 内容 |
|--------|--------|------|
| papers | 60 PDF | 论文PDF，命名混乱(paper.pdf/paper_1.pdf等) |
| metadata_ref | 283 | GitHub抓取元数据+YAML配置 |
| processed | 31 | 爬虫日志+manifest |

### 数据依赖分析 ⚠️ CRITICAL
v13_fix.py 实验仅使用 SemanticKITTI sequence 00 的 50 个扫描。
- 项目声称支持8个数据集，实际仅1个被实验使用
- nuScenes PCD真实数据存在但loader失败
- KITTI Raw .bin真实数据存在但未集成
- 其余5个数据集为metadata-only，根本无法用于实验
## 维度2: 实验架构审计

### 核心模型 (models/hyper_cad_bev.py, 761行) ✅
代码实现了手稿中描述的完整架构:
- RiemannianManifold2D (Section II-A): 度量张量g_ij, 协变梯度/散度, Hessian范数
- ReactionDiffusionPDE (Section II-A, Eq.1): 反应-扩散PDE, Perona-Malik自适应扩散
- IBEVField (Section II-C): SIREN隐式神经场, 5层256维
- LIFNeuron + NeuromorphicPDESolver (Section II-E/III-B): LIF SNN映射
- SGNet (Section II-B/III-A): 符号-几何先验网络
- ManifoldADMM (Section II-D, Eq.7-9): 流形ADMM优化器
- DynamicQueryScheduler (Section III-C): 动态查询调度
- HyperCADBEVv65Sparse: 完整端到端系统

代码质量: 良好，有清晰注释和Section引用。

### ⚠️ 严重问题: 实验脚本未调用核心模型!

v13_fix.py (454行) 是最终使用的实验脚本，但它:
- ❌ **不导入** models/hyper_cad_bev.py 中的任何类
- ❌ 使用纯NumPy+SciPy手动实现了简化版PDE (散度、度量张量等)
- ❌ 没有使用 IBEVField、SGNet、ManifoldADMM、NeuromorphicPDESolver
- ❌ 没有使用 torch 训练任何神经网络
- ❌ Loihi 2 的 TOPS/功耗/延迟数值是**硬编码常量**，不是实测值
- ✅ 确实使用了 SemanticKITTI 真实 .bin 点云数据做 BEV 投影

### scripts/ 文件夹分析 (58个文件)

| 类别 | 数量 | 文件 |
|------|------|------|
| 最终实验脚本 | 1 | v13_fix.py (454行) |
| 前一版实验 | ~15 | v8~v12系列 + final_exp系列 + deep_experiment系列 |
| 数据爬虫 | ~20 | scrape/download/01_data系列 |
| 废弃/占位符 | ~10 | write_experiment.py(8字节), _gen.py(41字节)等 |
| 工具/测试 | ~10 | test_loader.py, test_urls.py等 |

🚨 **脚本混乱程度严重:**
- 58个文件名中只有 v13_fix.py 是最终使用的
- v5/v6/v7/v8/v9/v10/v11 迭代产生了大量冗余脚本
- 部分文件仅含空壳代码 (如 write_experiment.py: 8字节, _gen.py: 41字节)

---

## 维度3: 交付物完整性

### TABLE 交付清单 (手稿要求 vs 实际)

| 手稿TABLE | 对应CSV | 内容 | 数据匹配 |
|-----------|---------|------|----------|
| TABLE I (PDE-SNN映射) | ❌ 无CSV | 理论映射表 | 概念表，无数据 |
| TABLE II (PDE消融) | ✅ table2_pde_ablation.csv | Sparse Raw/Manifold/Euclidean | ✅ v13实测 |
| TABLE III (优化器收敛) | ✅ table3_optimizer_convergence.csv | GD/ADMM/Manifold-ADMM | ⚠️ 收敛值存疑 |
| TABLE IV (SOTA对比) | ✅ table4_sota_comparison.csv | BEVFormer/BEVDet/MonoBEV等 | ⚠️ 对手值为论文声称 |
| TABLE V (版本演进) | ✅ table5_version_evolution.csv | v5.2/v6.0/v6.5 | ✅ mIoU/GeoErr一致 |
| TABLE VI (消融子表a/b/c/d) | ✅ table6a~6d.csv | 模块/查询/坡度/天气 | ✅ |
| TABLE VII (跨数据集) | ✅ table7_cross_dataset_transfer.csv | nuScenes/KITTI | ⚠️ v12 fallback值 |

### FIG 交付清单

| 手稿FIG | 生成文件 | 状态 |
|---------|----------|------|
| Fig.1 (系统架构图) | ❌ 无 | 需LaTeX/TikZ绘制 |
| Fig.2 (理论创新框架) | ❌ 无 | 需LaTeX/TikZ绘制 |
| Fig.3 (算法管道) | fig3_algorithm.png/pdf | ✅ 存在于figures_deep |
| Fig.4 (实验结果) | fig4_overview.png/pdf + fig4_comprehensive.png/pdf | ✅ 多版本 |
| Fig.5 (可视化验证) | fig5_visual_validation.png/pdf | ✅ |

FIG总计: 40个PNG/PDF文件 (3个子文件夹中的多个迭代版本)
最终版本在: experiments/figures_dep/  (8个文件)

### ⚠️ 交付物关键问题

1. **TABLE I 缺失**: 手稿Table I描述PDE↔SNN的七维度映射，但无对应CSV数据
2. **FIG 1/2 缺失**: 需要LaTeX/绘图工具手动创建，无法用Python生成
3. **FIG 3 仅存在于旧版**: fig3不在figures_dep/最终版中
4. **多版本冲突**: figures/figures_deep/figures_dep 三版FIG并存

---

## 维度4: 文档可读性

### ❌ 缺失的关键文件

| 应有文件 | 状态 |
|----------|------|
| README.md | ❌ 不存在 |
| LICENSE | ❌ 不存在 |
| requirements.txt | ❌ 不存在 |
| setup.py / pyproject.toml | ❌ 不存在 |
| .gitignore | ❌ 不存在 |
| CITATION.cff | ❌ 不存在 |

### 文件组织问题

1. **根目录杂乱**: 12个日志/元数据文件散落在根目录
2. **废弃脚本堆积**: scripts/ 中 80% 的文件是垃圾版本
3. **多版结果并存**: experiments/results, results_deep, results_dep 三个结果目录
4. **多版FIG并存**: figures, figures_deep, figures_dep 三组FIG
5. **论文PDF命名混乱**: papers/ 中 paper.pdf, paper_1.pdf, paper_2.pdf 无意义命名
6. **metadata_ref 过度抓取**: 283个文件(37MB)抓了大量不相关YAML/YML配置

### 手稿一致性

| 检查项 | 状态 |
|--------|------|
| manuscript_full.txt 提取来源 | ✅ IEEE TKDE 投稿 (submission ID ef6c319b) |
| 手稿作者 | ✅ Zihan Gao, Xie He, Yi Su, Hong Mei |
| 手稿页数 | ✅ 14页 |
| 核心算法 vs 代码实现 | ⚠️ 代码实现了但实验脚本未调用 |
| 实验指标 vs CSV数据 | ✅ mIoU=73.8%, GeoErr=27.8cm, 0.037TOPS 一致 |
---

## 综合评估与整改路线图

### 🔴 致命缺陷 (BLOCKER)

| # | 问题 | 影响 | 修复建议 |
|---|------|------|----------|
| B1 | **实验脚本未调用核心模型** | 最严重：models/中的 PyTorch 代码从未运行过，v13_fix.py 是独立 NumPy 简化版 | 创建新脚本调用 HyperCADBEVv65Sparse 做端到端实验 |
| B2 | **Loihi 2 数据是硬编码的** | 0.037 TOPS、22mJ 等关键效率数据无法复现 | 至少注明来源：Intel Loihi 2 公开规格 [19] |
| B3 | **6/12 数据文件夹仅有元数据** | 跨数据集实验(TABLE VII)基于 v12 fallback 而非实测 | 修复PCD reader + 集成KITTI Raw + 完成缺失下载 |

### 🟡 重要缺陷 (MAJOR)

| # | 问题 | 影响 | 修复建议 |
|---|------|------|----------|
| M1 | **无 README.md** | 任何人(包括评审者)无法理解项目结构 | 创建英文README：项目概述+安装+运行+引用 |
| M2 | **无 LICENSE** | 无法确定代码使用许可 | 添加 MIT 或 Apache 2.0 LICENSE 文件 |
| M3 | **scripts/ 混乱** | 58个文件，仅1个(2%)是活跃的 | 移除非活跃脚本到 archive/ 或删除 |
| M4 | **FIG 1/2 缺失** | 手稿引用但无生成文件 | 创建LaTeX/TikZ源码，或标注为手绘 |

### 🟢 改进建议 (MINOR)

| # | 问题 | 修复建议 |
|---|------|----------|
| m1 | 论文PDF命名混乱 | 重命名为 arxiv_id_title.pdf 格式 |
| m2 | 多版结果并存 | 保留 results/ 为最终版，移走deprecated |
| m3 | metadata_ref 过度抓取 | 清理无关YAML配置，保留核心元数据 |
| m4 | 添加 requirements.txt | 列出 numpy, scipy, matplotlib, torch 等依赖 |
| m5 | 添加 .gitignore | 排除 __pycache__, *.pyc, 大bin文件 |
| m6 | 添加 CITATION.cff | 便于学术引用 |

---

## 修复优先级排序

### 优先级 0 (立即 - 影响论文可信度)
1. ✅ 在 README 中明确声明哪些数据是实测、哪些是引用


2. ✅ 标注 Loihi 2 TOPS/功耗数据来源 (Intel 公开规格)

### 优先级 1 (本周 - 核心功能)
3. 🔧 创建真正的端到端实验脚本，调用 models/hyper_cad_bev.py
4. 🔧 修复 nuScenes PCD reader + 集成 KITTI Raw 真实数据
5. 🔧 尝试下载 RELLIS-3D + Waymo

### 优先级 2 (提交前 - 项目完整性)
6. 📝 创建 README.md (英文) + requirements.txt
7. 📝 添加 LICENSE + .gitignore
8. 🧹 清理 scripts/ 中废弃文件
9. 🎨 FIG 1/2 创建 (LaTeX/TikZ)

---

## ⚠️ 手稿-实验一致性风险

1. **算法实现 vs 实验**: 手稿声称实现了完整的 Riemannian PDE + SIREN IBEV + SNN，但实验脚本 v13_fix.py 仅使用了 NumPy 有限差分 PDE 求解。如果评审者要求查看代码与实验的对应关系，会暴露这一 gap。

2. **数据覆盖**: 手稿声称在 8 个数据集上验证 (Section IV)，但实际仅 SemanticKITTI 被实验使用。TABLE VII 的跨数据集结果来自 v12 fallback。

3. **Neuromorphic 数值**: 0.037 TOPS 和 22mJ 在实验脚本中硬编码，不是通过实际部署测量得出。

---

*审计结束: 2026-07-13*
*报告文件: E:\Hyper-CAD-BEV-Experiments\AUDIT_REPORT_FULL_2026-07-13.md*
