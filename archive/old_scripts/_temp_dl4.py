import requests, os, re
for k in list(os.environ.keys()):
    if "proxy" in k.lower(): del os.environ[k]
s = requests.Session()
s.trust_env = False

# DSEC download page
r = s.get("https://dsec.ifi.uzh.ch/dsec-datasets/download/", timeout=30)
print("DSEC download page:", r.status_code, len(r.text))
zips = re.findall(r"href=[\"']([^\"']*\.zip)[\"']", r.text)
if zips:
    for z in zips[:30]:
        print("ZIP:", z)
else:
    # Just print all hrefs with zip or data
    hrefs = re.findall(r'href="([^"]*)"', r.text)
    for h in hrefs:
        if "zip" in h.lower() or "lidar" in h.lower():
            print("HREF:", h[:200])

print("\n--- TartanDrive 2 ---")
for name in ["tartan_drive_2", "tartandrive2", "tartandrive", "TartanDrive2"]:
    r4 = s.get(f"https://github.com/castacks/{name}", timeout=15)
    print(f"{name}: {r4.status_code}")

print("\n--- Waymo ---")
r5 = s.get("https://waymo.com/open/", timeout=30)
print("Waymo status:", r5.status_code)
