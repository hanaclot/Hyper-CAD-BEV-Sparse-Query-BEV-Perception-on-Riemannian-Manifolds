import urllib.request, time, json
from pathlib import Path
from datetime import datetime

DATA_ROOT = Path(r"E:\Hyper-CAD-BEV-Experiments\data")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
LOG = []
def log(msg):
    t = datetime.now().strftime("%H:%M:%S")
    print(f"[{t}] {msg}")
    LOG.append(msg)

def download(url, dest, desc, timeout=600):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 1000:
        mb = dest.stat().st_size / 1e6
        log(f"  SKIP {desc} (exists {mb:.1f}MB)")
        return dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        log(f"  DOWNLOAD {desc} from {url[:80]}...")
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
        with open(dest, "wb") as f:
            f.write(data)
        mb = len(data) / 1e6
        log(f"  DONE {desc}: {mb:.1f}MB")
        return len(data)
    except Exception as e:
        log(f"  FAIL {desc}: {str(e)[:100]}")
        return 0

total = 0
sk = DATA_ROOT / "semantickitti_official"

# Batch 1: Labels (179MB)
mb = download("http://semantic-kitti.org/assets/data_odometry_labels.zip",
              sk / "labels.zip", "SK Labels (179MB)")
total += mb
time.sleep(2)

# Batch 2: Voxels (694MB)
mb = download("http://semantic-kitti.org/assets/data_odometry_voxels.zip",
              sk / "voxels.zip", "SK Voxels (694MB)")
total += mb
time.sleep(2)

log(f"Batch 1+2 total: {total/1e6:.1f}MB")

with open(DATA_ROOT / "processed" / "batch12_log.json", "w") as f:
    json.dump({"total": total, "log": LOG}, f, indent=2)