# -*- coding: utf-8 -*-
"""
Hyper-CAD-BEV v6.5-Sparse 实验管线 - Phase 2: 提取velodyne_laser.zip真实LiDAR统计
从SemanticKITTI velodyne_laser.zip (878MB, 472帧, sequence 00) 提取真实点云统计
"""
import sys, os, json, time, zipfile, struct, io, csv
from datetime import datetime
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = r"E:\Hyper-CAD-BEV-Experiments"
VELODYNE_ZIP = os.path.join(PROJECT_ROOT, "data", "semantickitti", "velodyne_laser.zip")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

print("=" * 60)
print("Phase 2: Extracting Velodyne LiDAR Statistics from 472 frames")
print("=" * 60)

all_stats = []
frame_sample = None
total_points = 0

print(f"Opening: {VELODYNE_ZIP}")
print(f"Size: {os.path.getsize(VELODYNE_ZIP) / (1024**2):.1f} MB")

with zipfile.ZipFile(VELODYNE_ZIP, 'r') as zf:
    bin_files = sorted([n for n in zf.namelist() if n.endswith('.bin')])
    total_files = len(bin_files)
    print(f"Total .bin files: {total_files}")
    
    # Show first few names
    for fn in bin_files[:5]:
        print(f"  {fn}")
    print(f"  ... (showing first 5 of {total_files})")
    
    for idx, fname in enumerate(bin_files):
        with zf.open(fname) as f:
            raw = f.read()
        
        # Velodyne HDL-64E: each point = [x, y, z, intensity] = 4 float32
        n_points = len(raw) // 16
        pts = np.frombuffer(raw, dtype=np.float32).reshape(-1, 4)
        total_points += n_points
        
        stats = {
            "frame_id": idx,
            "n_points": n_points,
            "x_min": float(np.min(pts[:, 0])), "x_max": float(np.max(pts[:, 0])),
            "y_min": float(np.min(pts[:, 1])), "y_max": float(np.max(pts[:, 1])),
            "z_min": float(np.min(pts[:, 2])), "z_max": float(np.max(pts[:, 2])),
            "z_mean": float(np.mean(pts[:, 2])), "z_std": float(np.std(pts[:, 2])),
            "intensity_mean": float(np.mean(pts[:, 3])), "intensity_std": float(np.std(pts[:, 3])),
            "xy_density": n_points / max((pts[:, 0].max() - pts[:, 0].min()) * (pts[:, 1].max() - pts[:, 1].min()), 1e-6),
        }
        all_stats.append(stats)
        
        if idx == 0:
            frame_sample = {
                "x_sample": pts[:200, 0].tolist(),
                "y_sample": pts[:200, 1].tolist(),
                "z_sample": pts[:200, 2].tolist(),
                "intensity_sample": pts[:200, 3].tolist(),
            }
        
        if (idx + 1) % 100 == 0 or (idx + 1) == total_files:
            print(f"  Processed {idx+1}/{total_files} frames, total points: {total_points:,}")

# Compute aggregate statistics
zs = [s["z_min"] for s in all_stats]
z_maxs = [s["z_max"] for s in all_stats]
z_means = [s["z_mean"] for s in all_stats]
intensities = [s["intensity_mean"] for s in all_stats]
densities = [s["xy_density"] for s in all_stats]
n_points_all = [s["n_points"] for s in all_stats]

aggregate_stats = {
    "total_frames": total_files,
    "total_points": total_points,
    "avg_points_per_frame": float(np.mean(n_points_all)),
    "z_extent": {"min": float(np.min(zs)), "max": float(np.max(z_maxs))},
    "z_mean_of_means": float(np.mean(z_means)),
    "z_std_of_means": float(np.std(z_means)),
    "intensity_mean_of_means": float(np.mean(intensities)),
    "intensity_std": float(np.std(intensities)),
    "avg_density": float(np.mean(densities)),
    "timestamp": datetime.now().isoformat(),
}

print(f"\nAggregate Statistics:")
for k, v in aggregate_stats.items():
    if isinstance(v, dict):
        print(f"  {k}: {v}")
    else:
        print(f"  {k}: {v}")

# Save results
output = {
    "aggregate": aggregate_stats,
    "per_frame_stats": all_stats,
    "frame_0_sample": frame_sample,
}
out_path = os.path.join(PROCESSED_DIR, "velodyne_frame_stats.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"\nSaved to: {out_path}")
print("Phase 2 Complete!")
