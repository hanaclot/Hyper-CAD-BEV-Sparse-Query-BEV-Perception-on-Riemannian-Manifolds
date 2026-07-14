import sys, os, json, zipfile, numpy as np
from datetime import datetime
sys.stdout.reconfigure(encoding="utf-8")
PROJECT_ROOT = r"E:\Hyper-CAD-BEV-Experiments"
VELODYNE_ZIP = os.path.join(PROJECT_ROOT, "data", "semantickitti", "velodyne_laser.zip")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

print("Phase 2: Extracting Velodyne LiDAR Statistics")
print("=" * 60)

all_stats = []
frame_sample = None
total_points = 0
corrupted = 0

def safe_float(x):
    return float(x) if x is not None else 0.0

with zipfile.ZipFile(VELODYNE_ZIP, 'r') as zf:
    bin_files = sorted([n for n in zf.namelist() if n.endswith('.bin')])
    total_files = len(bin_files)
    print(f"Total .bin files: {total_files}")
    
    for idx, fname in enumerate(bin_files):
        try:
            with zf.open(fname) as f:
                raw = f.read()
        except:
            corrupted += 1
            if corrupted <= 5:
                print(f"  [SKIP] Corrupted: {fname}")
            continue
        
        n_points = len(raw) // 16
        pts = np.frombuffer(raw, dtype=np.float32).reshape(-1, 4)
        total_points += n_points
        
        stats = {
            "frame_id": idx, "file": fname, "n_points": int(n_points),
            "x_min": safe_float(pts[:, 0].min()), "x_max": safe_float(pts[:, 0].max()),
            "y_min": safe_float(pts[:, 1].min()), "y_max": safe_float(pts[:, 1].max()),
            "z_min": safe_float(pts[:, 2].min()), "z_max": safe_float(pts[:, 2].max()),
            "z_mean": safe_float(pts[:, 2].mean()), "z_std": safe_float(pts[:, 2].std()),
            "intensity_mean": safe_float(pts[:, 3].mean()), "intensity_std": safe_float(pts[:, 3].std()),
            "xy_density": safe_float(n_points / max((pts[:, 0].max() - pts[:, 0].min()) * (pts[:, 1].max() - pts[:, 1].min()), 1e-6)),
        }
        all_stats.append(stats)
        
        if frame_sample is None and n_points > 10000:
            frame_sample = {
                "file": fname,
                "x_all": [float(v) for v in pts[:, 0].tolist()],
                "y_all": [float(v) for v in pts[:, 1].tolist()],
                "z_all": [float(v) for v in pts[:, 2].tolist()],
                "intensity_all": [float(v) for v in pts[:, 3].tolist()],
            }
        
        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{total_files}: good={len(all_stats)}, bad={corrupted}")

good_count = len(all_stats)
zs = [s["z_min"] for s in all_stats]
z_maxs = [s["z_max"] for s in all_stats]
z_means = [s["z_mean"] for s in all_stats]
intensities = [s["intensity_mean"] for s in all_stats]
densities = [s["xy_density"] for s in all_stats]
n_points_all = [s["n_points"] for s in all_stats]

aggregate = {
    "total_files_in_zip": total_files, "good_frames": good_count, "corrupted": corrupted,
    "total_points": total_points, "avg_points_per_frame": round(float(np.mean(n_points_all)), 1),
    "z_extent_m": {"min": round(float(np.min(zs)), 3), "max": round(float(np.max(z_maxs)), 3)},
    "z_mean_of_means": round(float(np.mean(z_means)), 3),
    "z_std_of_means": round(float(np.std(z_means)), 3),
    "intensity_mean": round(float(np.mean(intensities)), 3),
    "intensity_std": round(float(np.std(intensities)), 3),
    "avg_density_pts_per_m2": round(float(np.mean(densities)), 2),
    "timestamp": datetime.now().isoformat(),
}

output = {"aggregate": aggregate, "per_frame_stats": all_stats, "frame_sample": frame_sample}
out_path = os.path.join(PROCESSED_DIR, "velodyne_frame_stats.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False)

print(f"\nAggregate Results:")
for k, v in aggregate.items():
    print(f"  {k}: {v}")
print(f"\nSaved: {out_path} ({os.path.getsize(out_path)//1024} KB)")
print("Phase 2 Complete!")
