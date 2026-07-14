import os, sys, json, time, urllib.request, urllib.error, ssl, shutil
ssl._create_default_https_context = ssl._create_unverified_context

BASE = "E:/Hyper-CAD-BEV-Experiments/data"
os.makedirs(BASE, exist_ok=True)

SOURCES = [
    # 小文件 (< 50MB) - 快速获取
    {"name":"sk_calib","url":"http://semantic-kitti.org/assets/data_odometry_calib.zip","dest":f"{BASE}/semantickitti_official/calib.zip","timeout":120},
    {"name":"sk_poses","url":"http://semantic-kitti.org/assets/data_odometry_poses.zip","dest":f"{BASE}/semantickitti_official/poses.zip","timeout":120},
    
    # KITTI raw 单序列 (~500MB-1GB)
    {"name":"kitti_raw_0926_0001","url":"https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0001_sync.zip","dest":f"{BASE}/kitti_raw/0926_0001_sync.zip","timeout":600},
    {"name":"kitti_raw_0926_0005","url":"https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0005/2011_09_26_drive_0005_sync.zip","dest":f"{BASE}/kitti_raw/0926_0005_sync.zip","timeout":600},
    {"name":"kitti_raw_0926_0009","url":"https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0009/2011_09_26_drive_0009_sync.zip","dest":f"{BASE}/kitti_raw/0926_0009_sync.zip","timeout":600},
    
    # SemanticKITTI dev kit
    {"name":"sk_devkit","url":"https://github.com/PRBonn/semantic-kitti-api/archive/refs/heads/master.zip","dest":f"{BASE}/semantickitti_official/semantic-kitti-api.zip","timeout":180},
]

def download_one(src):
    url, dest, name, to = src['url'], src['dest'], src['name'], src.get('timeout',300)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        sz = os.path.getsize(dest)
        if sz > 1000:
            print(f"[SKIP] {name}: {sz/1e6:.1f} MB")
            return sz
        else:
            try: os.remove(dest)
            except: pass
    print(f"[GET] {name}")
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0 (compatible)'})
            with urllib.request.urlopen(req, timeout=to) as r:
                total = int(r.headers.get('Content-Length',0))
                print(f"  Size: {total/1e6:.1f} MB")
                dl = 0
                with open(dest,'wb') as f:
                    while True:
                        chunk = r.read(4*1024*1024)
                        if not chunk: break
                        f.write(chunk)
                        dl += len(chunk)
                        if total and dl%(20*1024*1024) < 4*1024*1024:
                            sys.stdout.write(f"\r  {dl/1e6:.1f}/{total/1e6:.1f} MB")
                            sys.stdout.flush()
                sz = os.path.getsize(dest)
                print(f"\r[OK] {name}: {sz/1e6:.1f} MB   ")
                return sz
        except Exception as e:
            print(f"  Err: {e}")
            time.sleep(5)
    return 0

total = 0
for i, src in enumerate(SOURCES):
    print(f"\n[{i+1}/{len(SOURCES)}] {src['name']}")
    sz = download_one(src)
    total += sz

print(f"\nTotal: {total/1e9:.2f} GB")
