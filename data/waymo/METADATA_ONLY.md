# Waymo Open Dataset -- DATA STATUS

## What is present
- 4 real TFRecord samples (9.0 MB) from waymo-research/waymo-open-dataset GitHub
  - motion_data_one_example.tfrecord (1.2 MB)
  - motion_data_one_scenario.tfrecord (0.9 MB)
  - pvps_data_one_frame.tfrecord (2.1 MB)
  - womd_lidar_and_camera_data.tfrecord (4.8 MB)
- GitHub repo structure (31 files, 1.1 MB): tutorial notebooks + source code

## What is NOT present
- Full Waymo Open Dataset (~1.2TB): requires Google Cloud Storage authentication
- These 4 .tfrecord files are test samples bundled with the waymo-open-dataset repo
- They contain real motion/pvps/lidar+camera data for development/testing

## Source
- waymo-research/waymo-open-dataset GitHub repository
- Downloaded 2026-07-13 via raw.githubusercontent.com
