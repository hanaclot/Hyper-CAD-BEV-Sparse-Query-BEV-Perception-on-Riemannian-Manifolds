# CHECKPOINT -- 2026-07-13

## Status: Text cleanup + data integrity fixes applied

### Data downloads on 2026-07-13
- Waymo: 4 real TFRecord samples (9.0 MB) from waymo-research/waymo-open-dataset GitHub
- TartanDrive2: 6 real camera images (4.3 MB) from castacks/tartanDrive2 GitHub
- RELLIS-3D: 22 real images (16.0 MB) from unmannedlab/RELLIS-3D GitHub

### v20 Experiment (July 13, 2026)
- 120 LiDAR scans from 3 datasets (40 each)
- 3-method PDE ablation in 31 seconds
- Results:

| Method              | PSNR(dB) | EdgeF1 | GeoErr(cm) |
|---------------------|----------|--------|------------|
| Sparse Raw (no PDE) | 8.7      | 0.4157 | 27.2       |
| Euclidean PDE       | 38.3     | 0.8119 | 0.9        |
| Manifold PDE        | 39.4     | 0.8119 | 0.8        |

### Per-Source Breakdown (Manifold PDE)
| Dataset        | Scans | Points/Scan | GeoErr(cm) | EdgeF1 | Slope_sigma | LiDAR       |
|----------------|-------|-------------|------------|--------|-------------|-------------|
| SemanticKITTI  | 40    | 121,352     | 0.84       | 0.7317 | 0.4095      | 64ch HDL-64E|
| nuScenes       | 40    | 34,723      | 0.68       | 0.9190 | 1.2579      | 32ch HDL-32E|
| KITTI Raw      | 40    | 121,706     | 0.92       | 0.7850 | 0.4306      | 64ch HDL-64E|

### Data Status (updated 2026-07-13)
- SemanticKITTI: 29,281 files, 19.4 GB -- LiDAR .bin scans from SemanticKITTI.org
- nuScenes v1.0-mini: 31,239 files, 9.1 GB -- LiDAR .pcd.bin scans from nuScenes.org
- KITTI Raw: 669 files, 0.8 GB (108 velodyne .bin scans) from cvlibs.net
- DSEC Event Camera: extracted semantic PNG + calibration YAML from DSEC website
- Weather: 35,088 hourly records (Berlin+PIttsburgh 2023-2024) from Open-Meteo API
- Papers: 42 PDFs (114 MB) from arXiv
- Waymo: 4 TFRecord samples (9 MB) + GitHub repo structure from waymo-research GitHub
- RELLIS-3D: 22 real scene images (16 MB) + GitHub repo tools from unmannedlab GitHub
- TartanDrive2: 6 real camera images (4.3 MB) + GitHub repo from castacks GitHub

### Missing full datasets (not feasible to download)
- Waymo Open Dataset: ~1.2TB, requires GCS authentication
- RELLIS-3D: ~350GB, Google Drive rate-limited
- TartanDrive 2.0: ~100GB, large storage required

### Deliverables
- 10 CSV tables in experiments/results_dep/
- 2 figures (ablation bar chart + visual comparison) in experiments/figures_dep/
- JSON summary: master_experiment_summary_v20.json
- Experiment log: experiment_log_v20.txt
- v20_lean.py: experiment script

### Key Bug Fixed
- v19: load_scans() filter used s[3] instead of s[2]
  causing 984 scans to load instead of 120. Fixed in v20.

### Provenance
- All metrics from .bin/.pcd.bin files
- Each TABLE row annotated with data source
- Missing datasets labeled as METADATA_ONLY
