import sys, os, json, csv, time, math
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import OrderedDict

PROJECT = Path(r"E:\Hyper-CAD-BEV-Experiments")
sys.path.insert(0, str(PROJECT / "models"))
sys.path.insert(0, str(PROJECT / "utils"))

from riemannian import RiemannianManifold
from pde_terrain import ReactionDiffusionPDE, AnisotropicDiffusionField

RESULTS_DIR = PROJECT / "experiments" / "results"
FIGURES_DIR = PROJECT / "experiments" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    t = datetime.now().strftime("%H:%M:%S")
    print(f"[{t}] {msg}")

log("=" * 60)
log("HYPER-CAD-BEV v6.5-Sparse: REAL-DATA EXPERIMENT SUITE")
log("Data: SemanticKITTI Velodyne (471 frames, 57M pts)")
log("=" * 60)

# === LOAD REAL VELODYNE DATA ===
log("Loading REAL velodyne_frame_stats.json...")
with open(PROJECT / "data" / "processed" / "velodyne_frame_stats.json", "r") as f:
    vdata = json.load(f)
agg = vdata["aggregate"]
per_frame = vdata["per_frame_stats"]
z_min_g = agg["z_extent_m"]["min"]
z_max_g = agg["z_extent_m"]["max"]
z_mean_r = agg["z_extent_m"]["mean"]
total_pts = agg["total_points"]
total_fr = agg["total_frames"]

frame_slopes = []
for pf in per_frame:
    zr = pf["z_max"] - pf["z_min"]
    frame_slopes.append(math.degrees(math.atan(zr / 5.0)))
frame_slopes = np.array(frame_slopes)
log(f"  Frames={total_fr}, Points={total_pts:,}, Z=[{z_min_g:.1f},{z_max_g:.1f}]m")
log(f"  Slope: mean={np.mean(frame_slopes):.1f}deg, max={np.max(frame_slopes):.1f}deg")
