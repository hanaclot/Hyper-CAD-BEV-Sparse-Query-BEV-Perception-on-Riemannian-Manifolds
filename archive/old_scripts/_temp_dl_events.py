import requests, os, zipfile, time
for k in list(os.environ.keys()):
    if "proxy" in k.lower(): del os.environ[k]
s = requests.Session()
s.trust_env = False

dsec_dir = r"E:\Hyper-CAD-BEV-Experiments\data\event_camera"

# Download a small sequence: interlaken_00_c events + images (these are smaller)
to_download = [
    ("https://download.ifi.uzh.ch/rpg/DSEC/train/interlaken_00_c/interlaken_00_c_events_left.zip", "interlaken_00_c_events_left.zip"),
    ("https://download.ifi.uzh.ch/rpg/DSEC/train/interlaken_00_c/interlaken_00_c_images_rectified_left.zip", "interlaken_00_c_images_left.zip"),
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
        r2 = s.get(url, timeout=1200, stream=True)
        t0 = time.time()
        total = 0
        with open(local, "wb") as f:
            for chunk in r2.iter_content(65536):
                f.write(chunk)
                total += len(chunk)
                if total % (10*65536) == 0 or total == int(cl) if cl != "unknown" else False:
                    elapsed = time.time() - t0
                    speed = total / elapsed / (1024**2) if elapsed > 0 else 0
                    pct = total / int(cl) * 100 if cl != "unknown" else 0
                    print(f"\r  {total/(1024**2):.1f}MB / {int(cl)/(1024**2):.1f}MB ({pct:.1f}%) speed={speed:.1f}MB/s", end="")
        elapsed = time.time() - t0
        print(f"\n  Downloaded: {total} bytes in {elapsed:.0f}s, valid ZIP: {open(local,'rb').read(4).hex()=='504b0304'}")
    except Exception as e:
        print(f"\n  ERROR: {e}")

print("\nFinal event_camera contents:")
for f in sorted(os.listdir(dsec_dir)):
    sz = os.path.getsize(os.path.join(dsec_dir, f))
    print(f"  {f}: {sz/(1024**2):.1f} MB")
