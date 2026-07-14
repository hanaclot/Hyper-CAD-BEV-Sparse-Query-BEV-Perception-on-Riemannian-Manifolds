import requests, os, re
for k in list(os.environ.keys()):
    if "proxy" in k.lower(): del os.environ[k]
s = requests.Session()
s.trust_env = False

# Check TartanDrive2 repos that returned 200
print("=== TartanDrive 2 ===")
for name in ["tartandrive2", "TartanDrive2"]:
    r = s.get(f"https://github.com/castacks/{name}", timeout=15)
    print(f"\n{name}: {r.status_code}")
    # Find repo name
    titles = re.findall(r'<title>(.*?)</title>', r.text)
    print(f"Title: {titles}")

# Also try checking for the actual tartan_drive_2 organization
print("\n=== Other TartanDrive repos ===")
for url in ["https://github.com/tartan-dataset/tartandrive2", 
            "https://github.com/tartanair/tartandrive2",
            "https://theairlab.org/tartandrive2/"]:
    try:
        r2 = s.get(url, timeout=15)
        print(f"{url}: {r2.status_code}")
    except Exception as e:
        print(f"{url}: ERROR {e}")
