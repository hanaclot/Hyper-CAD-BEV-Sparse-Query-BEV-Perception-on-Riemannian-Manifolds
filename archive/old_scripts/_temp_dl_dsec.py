import requests, os, time
for k in list(os.environ.keys()):
    if "proxy" in k.lower(): del os.environ[k]
s = requests.Session()
s.trust_env = False

# Check lidar_imu.zip size first (HEAD request)
url = "https://download.ifi.uzh.ch/rpg/DSEC/lidar_imu.zip"
r = s.head(url, timeout=30)
print(f"HEAD lidar_imu.zip: {r.status_code}")
print(f"Content-Length: {r.headers.get('content-length', 'unknown')}")
size_bytes = int(r.headers.get('content-length', 0))
print(f"Size: {size_bytes / (1024**3):.1f} GB")

# Try with range request to estimate speed
r2 = s.get(url, headers={"Range": "bytes=0-1048575"}, timeout=30, stream=True)
chunk = r2.content
expected = min(1048576, len(chunk))
print(f"Got first {len(chunk)} bytes, valid ZIP: {chunk[:4].hex() == '504b0304'}")

# Now try downloading the full file
print("\nStarting full download of lidar_imu.zip...")
dsec_dir = r"E:\Hyper-CAD-BEV-Experiments\data\event_camera"
local = os.path.join(dsec_dir, "lidar_imu.zip")
t0 = time.time()
r3 = s.get(url, timeout=3600, stream=True)
total = 0
with open(local, "wb") as f:
    for chunk in r3.iter_content(65536):
        f.write(chunk)
        total += len(chunk)
        elapsed = time.time() - t0
        if elapsed > 0:
            speed = total / elapsed / (1024**2)
            pct = total / size_bytes * 100 if size_bytes else 0
            eta = (size_bytes - total) / (total / elapsed) if total > 0 else 0
            print(f"\r  {total/(1024**3):.1f}GB / {size_bytes/(1024**3):.1f}GB ({pct:.1f}%) speed={speed:.1f}MB/s ETA={eta:.0f}s", end="", flush=True)
elapsed = time.time() - t0
print(f"\nDownload complete: {total} bytes in {elapsed:.0f}s ({total/elapsed/(1024**2):.2f} MB/s)")
