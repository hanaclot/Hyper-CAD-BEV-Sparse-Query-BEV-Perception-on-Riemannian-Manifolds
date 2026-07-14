# 数据来源 / Data Sources

## SemanticKITTI
- **官网**：http://semantic-kitti.org/dataset.html
- **说明**：基于 KITTI Odometry 数据集的大规模点云语义分割标注
- **所需文件**：
  - data_odometry_labels.zip：逐点语义标签
  - data_odometry_voxels.zip：体素化点云
  - data_odometry_calib.zip：传感器标定
  - data_odometry_poses.zip：位姿信息
- **下载方式**：访问官网注册账号后下载
- **大小**：约 80 GB

## nuScenes v1.0-mini
- **官网**：https://www.nuscenes.org/nuscenes#download
- **说明**：nuScenes 完整数据集的迷你子集，包含 10 个场景
- **所需文件**：
  - 1.0-mini.tgz：元数据与标注
  - 1.0-mini_blobs.tgz：传感器数据
- **下载方式**：在官网注册并同意条款后下载
- **大小**：约 4 GB

## KITTI Raw Data
- **官网**：https://www.cvlibs.net/datasets/kitti/raw_data.php
- **说明**：KITTI 原始传感器数据，包含同步和未同步图像、LiDAR 点云及标定
- **所需文件**：
  - 选择具体日期和驾驶场景的 .zip 文件
  - 例如：2011_09_26_drive_0005_sync.zip
  - devkit_raw_data.zip：开发工具包
  - 2011_09_26_calib.zip：标定文件
- **下载方式**：在官网选择所需日期/场景下载
- **大小**：每个场景约 1-2 GB

## Waymo Open Dataset
- **官网**：https://waymo.com/open/
- **GitHub**：https://github.com/waymo-research/waymo-open-dataset
- **说明**：Waymo 自动驾驶数据集，包含高分辨率 LiDAR 和摄像头数据
- **所需文件**：TFRecord 格式的感知数据
- **下载方式**：需 Google Cloud Storage 认证（gcloud auth login），然后运行：
  `ash
  gsutil -m cp -r gs://waymo_open_dataset_v_1_4_0/individual_files/<segment_name> ./data/waymo/
  `
- **大小**：约 1.2 TB（完整数据）

## RELLIS-3D
- **官网**：https://rellis-3d.tamu.edu/
- **GitHub**：https://github.com/unmannedlab/RELLIS-3D
- **说明**：越野环境下的多模态语义分割数据集
- **所需文件**：ROS bag 文件和标注
- **下载方式**：通过 Google Drive 获取（链接在 GitHub README 中）
- **大小**：约 350 GB

## TartanDrive 2.0
- **官网**：https://tartandrive.org/
- **GitHub**：https://github.com/castacks/tartanDrive2.0
- **说明**：大规模越野自动驾驶数据集，包含 LiDAR、RGB 相机及控制信号
- **所需文件**：ROS bag 文件
- **下载方式**：通过项目网站提供的下载链接
- **大小**：约 100 GB

## DSEC Event Camera
- **官网**：https://dsec.ifi.uzh.ch/
- **说明**：事件相机立体运动分割数据集
- **所需文件**：
  - 	rain_semantic_segmentation.zip
  - interlaken_00_c_events_left.zip
  - dsec_zurich_city_13_a_events_left.zip
  - interlaken_00_c_images_left.zip
  - 各场景标定文件
- **下载方式**：在官网注册账号后下载
- **大小**：约 0.3 GB（事件数据），完整数据约 1 TB

## 天气数据
- **Berlin**：包含 2023-2024 年温度、降雨量、能见度等气象数据
- **Pittsburgh**：2023-2024 年小时级气象数据
- **数据来源**：公开气象数据 API
- **存储位置**：data/weather_real/（JSON 格式）

## 论文参考
- 存储在 data/papers/ 目录下
- 包含 42 篇 arXiv PDF 论文，涵盖 BEV 感知、点云处理、Riemannian 深度学习等领域
- 通过 arXiv.org 公开获取

---

**注意**：由于 GitHub 文件大小限制（100 MB），data/ 目录中的大型数据集文件（.zip, .tgz, .tfrecord）未包含在仓库中。请按照上述说明从各数据源下载原始数据文件。

**已包含在仓库中的文件**：
- 数据集的元数据（JSON 格式）
- 标定文件（.txt）
- 处理后的数据索引（.csv, .json）
- 数据集官方文档和 README
- 天气数据（JSON 格式）
