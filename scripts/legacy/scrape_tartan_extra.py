import urllib.request, re, os, json
# TartanDrive2 GitHub
req = urllib.request.Request('https://api.github.com/repos/castacks/TartanDrive2.0', headers={'Accept': 'application/vnd.github.v3+json', 'User-Agent': 'Codex-Agent'})
try:
    resp = urllib.request.urlopen(req, timeout=30)
    data = json.loads(resp.read().decode())
    stars = data.get('stargazers_count', 0)
    forks = data.get('forks_count', 0)
    desc = (data.get('description', '') or '')[:100]
    print('TartanDrive2 GitHub: stars=' + str(stars) + ', forks=' + str(forks) + ', desc=' + desc)
    os.makedirs('E:/HyperCAD_BEV_2026/data/tartandrive2', exist_ok=True)
    with open('E:/HyperCAD_BEV_2026/data/tartandrive2/github_metadata.json', 'w') as f:
        json.dump(data, f, indent=2)
except Exception as e:
    print('TartanDrive2 GitHub: ' + str(e))

# Also try theairlab.org/tartandrive2
req2 = urllib.request.Request('https://theairlab.org/tartandrive2/', headers={'User-Agent': 'Mozilla/5.0'})
try:
    resp2 = urllib.request.urlopen(req2, timeout=30)
    html = resp2.read().decode('utf-8', errors='replace')
    with open('E:/HyperCAD_BEV_2026/data/tartandrive2/theairlab.html', 'w', encoding='utf-8') as f:
        f.write(html)
    print('TartanDrive2 theairlab: ' + str(len(html)) + ' bytes')
    km_match = re.findall(r'([0-9]+[.,]?[0-9]*)\s*(km|kilometers|hours?|trajectories)', html, re.IGNORECASE)
    print('Stats found: ' + str(km_match[:10]))
except Exception as e:
    print('TartanDrive2 theairlab: ' + str(e))
