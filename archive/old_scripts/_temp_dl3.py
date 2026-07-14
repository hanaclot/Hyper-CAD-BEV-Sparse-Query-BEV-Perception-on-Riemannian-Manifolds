import requests, os
for k in list(os.environ.keys()):
    if 'proxy' in k.lower(): del os.environ[k]
# Test DSEC direct download site
s = requests.Session()
s.trust_env = False
# Try to access DSEC official site
r = s.get('https://dsec.ifi.uzh.ch/', timeout=30)
print('DSEC homepage:', r.status_code, len(r.text))
# Try to find download links
for line in r.text.split('\n'):
    if 'download' in line.lower() or 'zip' in line.lower() or '.bin' in line.lower():
        print('DSEC>', line.strip()[:200])

# Also try TartanDrive 2
print('\n--- TartanDrive 2 ---')
r2 = s.get('https://github.com/castacks/tartan_drive_2', timeout=30)
print('TartanDrive2 github:', r2.status_code)
for line in r2.text.split('\n'):
    if 'download' in line.lower() or 'data' in line.lower() or '.zip' in line.lower() or 'tartan' in line.lower():
        if any(ext in line.lower() for ext in ['http', '.zip', 'download', 'url']):
            print('TD2>', line.strip()[:200])
