# Quick sweep - fixed unicode
import os, sys, math, time, warnings
from pathlib import Path
import numpy as np
from scipy import ndimage

warnings.filterwarnings("ignore")
np.random.seed(42)

BEV_SIZE = 200; BEV_RANGE = 50.0; BEV_RES = BEV_RANGE * 2 / BEV_SIZE
N_TEST = 5
NOISE_SIGMA = 0.15

PROJECT = Path(r"E:\Hyper-CAD-BEV-Experiments")
DATA_ROOT = PROJECT / "data"
LEARNING_MAP = {0:0,1:0,10:1,11:2,13:5,15:3,16:5,18:4,20:5,30:6,31:7,32:8,40:9,44:10,48:11,49:12,50:13,51:14,52:0,60:0,70:15,71:16,72:17,80:18,81:19,99:0,252:1,253:7,254:7,255:8,256:5,257:5,258:7,259:7}

label_map = {}
velo_dir = DATA_ROOT / "semantickitti_official" / "dataset" / "sequences" / "00" / "velodyne"
label_dir = DATA_ROOT / "semantickitti_official" / "labels" / "dataset" / "sequences" / "00" / "labels"
if label_dir.exists():
    for lf in label_dir.glob("*.label"): label_map[lf.stem] = lf

scans = []
for bf in sorted(velo_dir.glob("*.bin"), key=lambda x: int(x.stem))[:N_TEST]:
    pts = np.fromfile(bf, dtype=np.float32).reshape(-1, 4)
    s = {"name": bf.stem, "points": pts}
    if bf.stem in label_map:
        lr = np.fromfile(label_map[bf.stem], dtype=np.uint32)
        s["labels_mapped"] = np.array([LEARNING_MAP.get(int(l & 0xFFFF), 0) for l in lr])
    scans.append(s)
print(f"Loaded {len(scans)} scans")

def project_bev(scan):
    pts = scan["points"]
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    mask = (np.abs(x) < BEV_RANGE) & (np.abs(y) < BEV_RANGE)
    x, y, z = x[mask], y[mask], z[mask]
    xi = np.clip(((x + BEV_RANGE) / BEV_RES).astype(np.int32), 0, BEV_SIZE - 1)
    yi = np.clip(((y + BEV_RANGE) / BEV_RES).astype(np.int32), 0, BEV_SIZE - 1)
    height = np.full((BEV_SIZE, BEV_SIZE), -np.inf)
    for i in range(len(xi)):
        if z[i] > height[yi[i], xi[i]]: height[yi[i], xi[i]] = z[i]
    height[~np.isfinite(height)] = 0.0
    return height

def metric_tensor(height):
    h = ndimage.gaussian_filter(height, sigma=1.0)
    hy, hx = np.gradient(h, BEV_RES)
    g11 = 1.0 + hx*hx; g12 = hx*hy; g22 = 1.0 + hy*hy
    det_g = np.maximum(g11*g22 - g12*g12, 1e-8)
    inv_det = 1.0/det_g
    return {"ginv11": g22*inv_det, "ginv12": -g12*inv_det, "ginv22": g11*inv_det, "sqrt_det": np.sqrt(det_g)}

def div_operation(fx, fy):
    df = np.zeros_like(fx)
    df[1:-1,:] = (fx[2:,:] - fx[:-2,:])/(2*BEV_RES)
    df[:,1:-1] += (fy[:,2:] - fy[:,:-2])/(2*BEV_RES)
    df[0,:] = (fx[1,:] - fx[0,:])/BEV_RES; df[-1,:] = (fx[-1,:] - fx[-2,:])/BEV_RES
    df[:,0] += (fy[:,1] - fy[:,0])/BEV_RES; df[:,-1] += (fy[:,-1] - fy[:,-2])/BEV_RES
    return df

def solve_pde(noisy, metric, D, reaction, steps, method):
    u = noisy.astype(np.float64).copy()
    sd = metric["sqrt_det"]
    g11 = metric["ginv11"]; g12 = metric["ginv12"]; g22 = metric["ginv22"]
    for _ in range(steps):
        uy, ux = np.gradient(u, BEV_RES)
        if method == "manifold":
            gx = g11*ux + g12*uy; gy = g12*ux + g22*uy
            fx = D*sd*gx; fy = D*sd*gy
            diff = div_operation(fx, fy)/(sd + 1e-8)
        elif method == "euclidean":
            diff = div_operation(D*ux, D*uy)
        else:
            diff = np.zeros_like(u)
        react = reaction*(noisy - u)
        u = u + 0.02*(diff + react)
        u = np.clip(u, 0, 1)
    return u

def geo_err(denoised, clean, mask):
    if mask.sum() < 10: return 0
    return float(np.mean(np.abs(denoised[mask] - clean[mask]))*100)

# ---- SWEEP ----
configs = [
    (0.02, 0.04, 50, "cfg_A", "D=0.02,reac=0.04,50steps"),
    (0.01, 0.03, 75, "cfg_B", "D=0.01,reac=0.03,75steps"),
    (0.03, 0.06, 40, "cfg_C", "D=0.03,reac=0.06,40steps"),
    (0.04, 0.04, 40, "cfg_D", "D=0.04,reac=0.04,40steps"),
    (0.06, 0.04, 30, "cfg_E", "D=0.06,reac=0.04,30steps"),
    (0.08, 0.04, 25, "cfg_F", "D=0.08,reac=0.04,25steps"),
    (0.05, 0.03, 30, "cfg_G", "D=0.05,reac=0.03,30steps"),
]

print("\nRunning sweep...")
best_mg, best_cfg = 999, None
for D, reaction, steps, label, desc in configs:
    t0 = time.time()
    res = {"no_pde": [], "euclidean": [], "manifold": []}
    for scan in scans:
        h = project_bev(scan)
        m = metric_tensor(h)
        h_pos = h[h>0]; h_min, h_max = h_pos.min(), h_pos.max()
        hn = (h - h_min)/(h_max - h_min + 1e-8)
        noisy = hn + np.random.randn(*hn.shape)*NOISE_SIGMA*(h>0).astype(float)
        noisy = np.clip(noisy, 0, 1)
        mask = h > 0
        res["no_pde"].append(geo_err(noisy, hn, mask))
        pe = solve_pde(noisy, m, D, reaction, steps, "euclidean")
        res["euclidean"].append(geo_err(pe, hn, mask))
        pm = solve_pde(noisy, m, D, reaction, steps, "manifold")
        res["manifold"].append(geo_err(pm, hn, mask))
    
    D_total = D*steps*0.02; char_len = math.sqrt(2*D_total)*BEV_RANGE
    Dl = D/reaction if reaction > 0 else 999
    ng = np.mean(res["no_pde"]); eg = np.mean(res["euclidean"]); mg = np.mean(res["manifold"])
    delta = eg - mg  # positive = manifold better
    elapsed = time.time() - t0
    print(f"  {label}: D={D},reac={reaction},steps={steps} | len={char_len:.1f}m, D/lambda={Dl:.3f}")
    print(f"    Noise={ng:.2f}  Euclid={eg:.2f}  Manifold={mg:.2f}  Delta_E-M={delta:+.3f} | {elapsed:.1f}s")
    if mg < best_mg: best_mg = mg; best_cfg = label

print(f"\nBest Geo: {best_cfg} = {best_mg:.2f}cm")
print("DONE")
