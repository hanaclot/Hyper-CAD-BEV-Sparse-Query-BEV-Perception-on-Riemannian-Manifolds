import requests, os
for k in list(os.environ.keys()):
    if 'proxy' in k.lower(): del os.environ[k]
s = requests.Session()
s.trust_env = False
r = s.get('https://raw.githubusercontent.com/unmannedlab/RELLIS-3D/main/README.md', timeout=30)
# Find BaiDu Pan section
lines = r.text.split('\n')
for i, line in enumerate(lines):
    if 'baidu' in line.lower() or 'pan' in line.lower():
        for j in range(max(0,i-1), min(len(lines), i+10)):
            print(f'L{i}: {lines[j].strip()[:200]}')
