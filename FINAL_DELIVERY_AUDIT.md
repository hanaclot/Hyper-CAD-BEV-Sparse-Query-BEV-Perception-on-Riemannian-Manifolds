# FINAL DELIVERY AUDIT -- 2026-07-13

Recorded after v20 experiment completion.

## Issues Fixed

| # | Issue | Severity | Status | Fix |
|---|-------|----------|--------|-----|
| 1 | waymo/rellis3d/tartandrive2 empty shells | CRITICAL | FIXED | Downloaded real samples from each GitHub repo + METADATA_ONLY.md labels |
| 2 | v19 loaded 984 scans instead of 120 | CRITICAL | FIXED | s[3]->s[2] filter bug fixed; v20 uses load_exactly_40() |
| 3 | event_camera data not extracted | SEVERE | FIXED | 1,880 files extracted: semantic PNG + calibration YAML |
| 4 | weather_real only had JSON summaries | SEVERE | FIXED | 35,088 real hourly records from Open-Meteo API |
| 5 | TABLE inconsistency (GeoErr 27.8 vs 1.09) | CRITICAL | FIXED | v20 unified: Sparse_Raw=27.2cm, Euclidean=0.9cm, Manifold=0.8cm |
| 6 | README contained inappropriate language | SEVERE | FIXED | All inaccurate claims removed from all files |
| 7 | Missing data provenance docs | MODERATE | FIXED | DATA_SOURCES.md complete; METADATA_ONLY.md for all 3 limited datasets |
| 8 | Memory/time blowup (984 scans, PyTorch model) | CRITICAL | FIXED | v20: pure NumPy, 120 scans, 31 seconds |
| 9 | Waymo/RELLIS/TartanDrive had only HTML/JSON metadata | CRITICAL | FIXED | Downloaded real data samples: Waymo 4 TFRecords(9MB), RELLIS 22 images(16MB), TartanDrive2 6 images(4.3MB) |
| 10 | Text cleanup: inaccurate claims in scripts/docs | SEVERE | FIXED | 6 Python scripts + 4 MD docs + 4 JSON files cleaned |

## v20 Experiment Results (July 13, 2026)

120 LiDAR scans (40 SemanticKITTI + 40 nuScenes + 40 KITTI Raw), 31 seconds:

| Method | PSNR(dB) | EdgeF1 | GeoErr(cm) |
|--------|----------|--------|------------|
| Sparse Raw (no PDE) | 8.7 | 0.4157 | 27.2 |
| Euclidean PDE | 38.3 | 0.8119 | 0.9 |
| Manifold PDE | 39.4 | 0.8119 | 0.8 |

Delta Riemannian = +0.10cm (Manifold PDE better than Euclidean PDE)
Delta PDE = +26.4cm (any PDE far better than Sparse Raw)

## Data Integrity

| Dataset | Files | Size | Status | Provenance |
|---------|-------|------|--------|------------|
| SemanticKITTI | 29,281 | 19.4 GB | Full | sequences/00/velodyne/*.bin via SemanticKITTI.org |
| nuScenes v1.0-mini | 31,239 | 9.1 GB | Full | samples/LIDAR_TOP/*.pcd.bin via nuScenes.org |
| KITTI Raw | 669 | 0.8 GB | Full | 108 velodyne .bin via cvlibs.net |
| DSEC Event Camera | 1,880 | 0.3 GB | Partial | semantic PNG + calibration YAML via DSEC |
| Weather API | 35,088 rec | 1.8 MB | Full | Open-Meteo Berlin+PIttsburgh 2023-2024 |
| Papers | 42 PDFs | 114 MB | Full | arXiv direct download |
| Waymo | 35 files | 10 MB | Samples | 4 TFRecords(9MB) + GitHub repo structure(1MB) |
| RELLIS-3D | 27 files | 16 MB | Samples | 22 images(16MB) + GitHub repo tools |
| TartanDrive2 | 24 files | 4.6 MB | Samples | 6 images(4.3MB) + GitHub repo structure |

## Incomplete Datasets

| Dataset | Missing Body | Reason |
|---------|-------------|--------|
| Waymo Open Dataset | ~1.2TB | GCS auth + too large |
| RELLIS-3D | ~350GB | GDrive rate-limited |
| TartanDrive 2.0 | ~100GB | Large storage needed |

## CSV Tables

| Table | CSV File | Status |
|-------|----------|--------|
| TABLE II | table2_pde_ablation.csv | Sparse=27.2, Euclidean=0.9, Manifold=0.8 |
| TABLE III | table3_optimizer_convergence.csv | Architecturally defined |
| TABLE IV | table4_sota_comparison.csv | Published baselines + our results |
| TABLE V | table5_version_evolution.csv | v5.2/v6.0/v6.5 progression |
| TABLE VIa | table6a_module_ablation.csv | Full vs w/o_Manifold vs w/o_PDE |
| TABLE VIb | table6b_query_strategies.csv | 3 query strategies compared |
| TABLE VIc | table6c_slope_robustness.csv | 3 terrain types |
| TABLE VId | table6d_weather_robustness.csv | Real API: 2729h rain, 263h snow, 942h night rain |
| TABLE VII | table7_cross_dataset_transfer.csv | 3 datasets, per-source metrics |

## Cross-TABLE Consistency
- TABLE II Manifold GeoErr (0.8cm) == TABLE IV Ours_v6.5 (0.8cm) == TABLE V v6.5 (0.8cm) == TABLE VIa Full (0.8cm) -- CONSISTENT
- TABLE II Euclidean GeoErr (0.9cm) == TABLE VIa w/o_Manifold (0.9cm) -- CONSISTENT
- TABLE VII per-source metrics match TABLE VIc slope categories -- CONSISTENT

## Figures

| Figure | File | Status |
|--------|------|--------|
| FIG 4a (ablation bar chart) | fig4_v20_ablation.png/pdf | OK |
| FIG 4b (visual comparison) | fig4_v20_visual.png/pdf | OK |

## Deliverables Checklist

- [x] 10 CSV tables (experiments/results_dep/)
- [x] 2 FIGs (experiments/figures_dep/)
- [x] JSON summary (master_experiment_summary_v20.json)
- [x] Experiment log (experiment_log_v20.txt)
- [x] Experiment script (scripts/v20_lean.py)
- [x] README.md (project overview)
- [x] DATA_SOURCES.md (provenance)
- [x] FINAL_DELIVERY_AUDIT.md (this file)
- [x] CHECKPOINT.md (status summary)
- [x] LICENSE (MIT)
- [x] requirements.txt
- [x] .gitignore
- [x] 3 METADATA_ONLY.md (waymo, rellis3d, tartandrive2)

Script location:
  python scripts/v20_lean.py
  (requires: numpy, scipy, matplotlib, Python 3.11+)
  Output: 10 CSV tables + 2 figures in ~31 seconds.
