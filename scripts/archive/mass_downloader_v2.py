import os, sys, json, time, urllib.request, urllib.error, ssl
ssl._create_default_https_context = ssl._create_unverified_context

BASE = "E:/Hyper-CAD-BEV-Experiments/data"
WORK = "D:/HyperCAD_BEV_2026/data"
os.makedirs(BASE, exist_ok=True)
os.makedirs(WORK, exist_ok=True)

SOURCES = [
    {"name":"sk_voxels","url":"http://semantic-kitti.org/assets/data_odometry_voxels_all.zip","dest":f"{BASE}/semantickitti_official/voxels_all.zip","min":3000},
    {"name":"sk_calib","url":"http://semantic-kitti.org/assets/data_odometry_calib.zip","dest":f"{BASE}/semantickitti_official/calib.zip","min":1},
    {"name":"sk_poses","url":"http://semantic-kitti.org/assets/data_odometry_poses.zip","dest":f"{BASE}/semantickitti_official/poses.zip","min":1},
    {"name":"sk_labels_all","url":"http://semantic-kitti.org/assets/data_odometry_labels.zip","dest":f"{BASE}/semantickitti_official/labels_all.zip","min":50},
]

def download_one(src, timeout=600):
    url, dest, name, min_sz = src['url'], src['dest'], src['name'], src['min']*1024*1024
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        sz = os.path.getsize(dest)
        if sz >= min_sz:
            print(f"[SKIP] {name}: {sz/1e6:.1f} MB")
            return sz
        else:
            try:
                os.remove(dest)
            except:
                newname = dest + ".old"
                os.rename(dest, newname)
                print(f"[RENAMED] old {name}")
    print(f"[GET] {name}: {url}")
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                total = int(r.headers.get('Content-Length',0))
                print(f"  Size: {total/1e6:.1f} MB")
                dl = 0
                with open(dest+'.part','wb') as f:
                    while True:
                        chunk = r.read(8*1024*1024)
                        if not chunk: break
                        f.write(chunk)
                        dl += len(chunk)
                        if total:
                            sys.stdout.write(f"\r  {dl/1e6:.1f}/{total/1e6:.1f} MB ({100*dl/total:.0f}%)")
                            sys.stdout.flush()
                os.rename(dest+'.part', dest)
                sz = os.path.getsize(dest)
                print(f"\n[OK] {name}: {sz/1e6:.1f} MB")
                return sz
        except Exception as e:
            print(f"\n[ERR attempt {attempt+1}] {name}: {e}")
            time.sleep(10)
    print(f"[FAIL] {name}")
    return 0

total = 0
ok, fail = [], []
for i, src in enumerate(SOURCES):
    print(f"\n[{i+1}/{len(SOURCES)}]")
    sz = download_one(src)
    if sz > 0:
        total += sz
        ok.append(src['name'])
    else:
        fail.append(src['name'])

print(f"\n{'='*60}")
print(f"DONE: {total/1e9:.2f} GB from {len(ok)} sources")
if fail: print(f"FAILED: {fail}")
with open(f"{WORK}/download_v2_results.json","w") as f:
    json.dump({"total_gb":total/1e9,"ok":ok,"fail":fail},f,indent=2)
