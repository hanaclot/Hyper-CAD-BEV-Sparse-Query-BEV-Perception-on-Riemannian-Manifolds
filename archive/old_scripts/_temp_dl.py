import requests, os
for k in list(os.environ.keys()):
    if 'proxy' in k.lower(): del os.environ[k]

s = requests.Session()
s.trust_env = False
r = s.get('https://raw.githubusercontent.com/unmannedlab/RELLIS-3D/main/README.md', timeout=30)
print('README.md loaded:', len(r.text), 'chars')
for line in r.text.split('\n'):
    l = line.lower()
    if 'download' in l or ('http' in l and ('rellis' in l or 'data' in l or 'gdrive' in l or 'kaggle' in l)):
        print('LINK>', line.strip()[:300])
