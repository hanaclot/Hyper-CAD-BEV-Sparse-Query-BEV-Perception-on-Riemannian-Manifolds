import requests, os, zipfile, time
for k in list(os.environ.keys()):
    if "proxy" in k.lower(): del os.environ[k]
s = requests.Session()
s.trust_env = False

dsec_dir = r"E:\Hyper-CAD-BEV-Experiments\data\event_camera"

# Download a smaller useful file: train_calibration + test_calibration + 1 small sequence
to_download = [
    ("https://download.ifi.uzh.ch/rpg/DSEC/test_coarse/test_calibration.zip", "test_calibration.zip"),
    ("https://download.ifi.uzh.ch/rpg/DSEC/semantic/train_semantic_segmentation.zip", "train_semantic_segmentation.zip"),
]

for url, fname in to_download:
    local = os.path.join(dsec_dir, fname)
    if os.path.exists(local):
        print(f"{fname}: already exists ({os.path.getsize(local)} bytes)")
        continue
    print(f"Downloading {fname}...")
    try:
        r = s.head(url, timeout=30)
        cl = r.headers.get("content-length", "unknown")
        print(f"  Content-Length: {cl}")
        r2 = s.get(url, timeout=300, stream=True)
        total = 0
        with open(local, "wb") as f:
            for chunk in r2.iter_content(65536):
                f.write(chunk)
                total += len(chunk)
        print(f"  Downloaded: {total} bytes, valid ZIP: {open(local,'rb').read(4).hex()=='504b0304'}")
    except Exception as e:
        print(f"  ERROR: {e}")

# List what we have now
print("\n=== event_camera directory contents ===")
for f in sorted(os.listdir(dsec_dir)):
    sz = os.path.getsize(os.path.join(dsec_dir, f))
    print(f"  {f}: {sz/1024:.1f} KB" if sz < 1024*1024 else f"  {f}: {sz/(1024**2):.1f} MB")
