# Data Sources and Download Instructions

This document provides detailed instructions for obtaining all datasets used in the Hyper-CAD-BEV project.

---

## Required Datasets

### 1. SemanticKITTI (19.4 GB)

**Description**: Sequential LiDAR point clouds with point-wise semantic annotations. Used as the primary benchmark for BEV perception and PDE ablation studies.

**Download**: 
- Official website: http://semantic-kitti.org/dataset.html#download
- Direct download link: http://www.semantic-kitti.org/assets/data_odometry_labels.zip
- After downloading, extract to `data/semantickitti_official/`

**Expected structure**:
```
data/semantickitti_official/
  sequences/
    00/
      velodyne/  (*.bin)
      labels/    (*.label)
    01/
      ...
```

**License**: Creative Commons BY-NC-SA 4.0

---

### 2. nuScenes v1.0-mini (5.1 GB)

**Description**: A subset of the nuScenes dataset with 10 scenes, including LiDAR, cameras, and radar data. Used for multi-sensor BEV perception across diverse urban environments.

**Download**:
- Official website: https://www.nuscenes.org/download
- Mini split: https://www.nuscenes.org/data/v1.0-mini.tgz
- Extract to `data/nuscenes/v1.0-mini/`

**Metadata already included in this repository**:
- `data/nuscenes/*.json` ? All metadata tables (sample, scene, ego_pose, calibrated_sensor, etc.)
- `data/metadata_ref/nuscenes-*.json` ? Backup copies of metadata

**License**: Custom (non-commercial, see nuScenes terms)

---

### 3. KITTI Raw (846 MB)

**Description**: Raw unsynchronized camera images and LiDAR scans from the KITTI benchmark. Used for real-world terrain validation.

**Download**:
- Official website: https://www.cvlibs.net/datasets/kitti/raw_data.php
- Use the provided download script: `python scripts/dl_kitti.py`
- Extract to `data/kitti_raw/`

**Expected structure**:
```
data/kitti_raw/
  2011_09_26/
    2011_09_26_drive_0001_sync/
      image_00/, image_01/, velodyne_points/
```

**License**: Creative Commons BY-NC-SA 3.0

---

## Optional / Reference Datasets

These datasets are **NOT required** to run the core experiments. Sample data and metadata are already included in this repository for reference purposes.

### DSEC Event Camera (318 MB)
- **Description**: Dynamic and Active-pixel Vision Sensor data for event-based perception
- **Download**: https://dsec.ifi.uzh.ch/
- **Included in repo**: Metadata files in `data/event_camera/`
- **Sample size**: 1,880 files with calibration and metadata

### Waymo Open Dataset (~1.2 TB)
- **Description**: Large-scale autonomous driving dataset from Waymo
- **Download**: https://waymo.com/open/ (requires GCS authentication)
- **Included in repo**: 39 files in `data/waymo/`, including:
  - Repository source code and tutorials
  - 4 TFRecord samples (~9.3 MB total)
  - Metadata and documentation

### RELLIS-3D (350 GB)
- **Description**: Off-road autonomous driving dataset with 3D LiDAR annotations
- **Download**: https://unmannedlab.github.io/research/RELLIS-3D
- **Included in repo**: 40 files in `data/rellis3d/`, including:
  - Repository source code (benchmarks, utils)
  - Sample images and documentation
  - 22 real sample images

### TartanDrive 2.0 (100 GB)
- **Description**: Off-road driving dataset with LiDAR, camera, and GPS data
- **Download**: https://github.com/castacks/tartan_drive_2.0
- **Included in repo**: 21 files in `data/tartandrive2/`, including:
  - Repository source code and calibration data
  - 6 sample camera images
  - Documentation and website snapshots

---

##  Weather Data

Weather metadata for robustness evaluation is included as JSON files:

- `data/weather_real/` (9 files, 1.8 MB): Hourly weather data for Berlin and Pittsburgh (2023-2024)
- `data/weather/` (5 files, 0.06 MB): Quarterly weather summaries

**Source**: Open-Meteo API (https://open-meteo.com/)

---

##  Research Paper Metadata

Reference paper metadata from ArXiv is stored in JSON format:

- `data/processed/` (34 JSON/CSV files): Scraped metadata, download logs, and integrated dataset manifests
- `data/metadata_ref/` (81 files): Paper-specific metadata, crawled leaderboard data, repository analysis

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/hanaclot/Hyper-CAD-BEV-Sparse-Query-BEV-Perception-on-Riemannian-Manifolds.git
cd Hyper-CAD-BEV-Sparse-Query-BEV-Perception-on-Riemannian-Manifolds

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download required datasets (manual)
#  - SemanticKITTI: semantic-kitti.org
#  - nuScenes v1.0-mini: nuscenes.org
#  - KITTI Raw: cvlibs.net

# 4. Run experiments
cd scripts && python v20_lean.py
```

---

## Notes

- All large binary files (LiDAR point clouds, images) are excluded from this Git repository due to size limits
- Metadata, results, figures, and source code are complete
- Contact us via GitHub Issues for any data access questions
