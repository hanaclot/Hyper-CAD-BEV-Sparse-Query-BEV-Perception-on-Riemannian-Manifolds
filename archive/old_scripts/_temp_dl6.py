import requests, os, time
for k in list(os.environ.keys()):
    if "proxy" in k.lower(): del os.environ[k]
s = requests.Session()
s.trust_env = False

# DSEC: try downloading calibration (small) + 1 small sequence
dsec_dir = r"E:\Hyper-CAD-BEV-Experiments\data\event_camera"
os.makedirs(dsec_dir, exist_ok=True)

test_urls = [
    ("https://download.ifi.uzh.ch/rpg/DSEC/train_coarse/train_calibration.zip", "train_calibration.zip"),
    ("https://download.ifi.uzh.ch/rpg/DSEC/train/interlaken_00_c/interlaken_00_c_calibration.zip", "interlaken_00_c_calibration.zip"),
]

for url, fname in test_urls:
    local = os.path.join(dsec_dir, fname)
    print(f"\nDownloading {fname}...")
    try:
        r = s.get(url, timeout=60, stream=True)
        print(f"  Status: {r.status_code}, Size: {r.headers.get('content-length', 'unknown')}")
        if r.status_code == 200:
            total = 0
            with open(local, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
                    total += len(chunk)
            print(f"  Downloaded: {total} bytes")
            # Check if valid zip
            with open(local, "rb") as f:
                header = f.read(4)
            print(f"  ZIP header: {header.hex()} (should be 504b0304)")
    except Exception as e:
        print(f"  ERROR: {e}")

# Check TartanDrive2 repos for data URLs
print("\n=== TartanDrive2 data check ===")
r = s.get("https://github.com/castacks/tartandrive2", timeout=15)
import re
# Find README content
md_match = re.search(r'<article[^>]*markdown-body[^>]*>(.*?)</article>', r.text, re.DOTALL)
if md_match:
    urls = re.findall(r'https?://[^\s"<>]+', md_match.group(0))
    for u in urls:
        if any(k in u.lower() for k in ['download', 'data', 'tartan', '.zip', '.tar', 'cmu']):
            print(f"URL: {u[:200]}")
    # Also print first 1000 chars
    text = re.sub(r'<[^>]+>', ' ', md_match.group(0))
    print("README preview:", text[:500])
