import numpy as np, json, struct, os
from pathlib import Path

# Read all 472 LiDAR frames and extract comprehensive statistics
data_dir = Path(r"E:\Hyper-CAD-BEV-Experiments\data\semantickitti_extracted\dataset\sequences\00\velodyne")
bins = sorted(data_dir.glob("*.bin"))

all_stats = []
total_pts = 0
heights = []
intensities = []
ranges = []

for i, bf in enumerate(bins):
    pts = np.fromfile(bf, dtype=np.float32).reshape(-1, 4)  # x,y,z,intensity
    n = len(pts)
    total_pts += n
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    stat = {
        "frame": int(bf.stem),
        "points": int(n),
        "x_range": [float(x.min()), float(x.max())],
        "y_range": [float(y.min()), float(y.max())],
        "z_range": [float(z.min()), float(z.max())],
        "mean_intensity": float(pts[:, 3].mean()),
        "mean_range": float(r.mean()),
        "max_range": float(r.max()),
        "mean_height": float(z.mean()),
    }
    all_stats.append(stat)
    heights.extend(z.tolist()[:100])  # sample
    intensities.extend(pts[:, 3].tolist()[:100])
    ranges.extend(r.tolist()[:100])
    if i % 50 == 0:

# Global stats
h_arr = np.array(heights)
i_arr = np.array(intensities)
r_arr = np.array(ranges)

aggregate = {
    "total_frames": len(bins),
    "total_points": int(total_pts),
    "height_stats": {"mean": float(h_arr.mean()), "std": float(h_arr.std()), "min": float(h_arr.min()), "max": float(h_arr.max())},
    "intensity_stats": {"mean": float(i_arr.mean()), "std": float(i_arr.std())},
    "range_stats": {"mean": float(r_arr.mean()), "std": float(r_arr.std()), "max": float(r_arr.max())},
    "per_frame": all_stats,
}

out = Path(r"E:\Hyper-CAD-BEV-Experiments\data\processed\comprehensive_lidar_stats.json")
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(aggregate, f, indent=2)

# Also save as numpy
np.savez_compressed(
    Path(r"E:\Hyper-CAD-BEV-Experiments\data\processed\lidar_global_stats.npz"),
    heights=h_arr, intensities=i_arr, ranges=r_arr
)

mb = os.path.getsize(out) / 1e6
