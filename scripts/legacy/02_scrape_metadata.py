import urllib.request, json, time, os

OUT_DIR = r'E:\Hyper-CAD-BEV-Experiments\data\processed'
os.makedirs(OUT_DIR, exist_ok=True)

results = {}

# 1. RELLIS-3D GitHub API
print('=== RELLIS-3D GitHub API ===')
try:
    req = urllib.request.Request('https://api.github.com/repos/unmannedlab/RELLIS-3D', headers={'User-Agent': 'Mozilla/5.0'})
    data = json.loads(urllib.request.urlopen(req, timeout=15).read())
    results['rellis3d'] = {'stars': data.get('stargazers_count'), 'forks': data.get('forks_count'), 'description': str(data.get('description',''))[:200]}
    print(f"Stars: {data.get('stargazers_count')}, Description: {str(data.get('description',''))[:150]}")
except Exception as e:
    results['rellis3d'] = {'error': str(e)}
    print(f'Error: {e}')

# 2. SemanticKITTI
print('\n=== SemanticKITTI Website ===')
try:
    req = urllib.request.Request('http://semantic-kitti.org/', headers={'User-Agent': 'Mozilla/5.0'})
    html = urllib.request.urlopen(req, timeout=15).read().decode('utf-8', errors='replace')
    results['semantickitti'] = {'html_size': len(html), 'has_benchmark': 'benchmark' in html.lower()}
    for kw in ['sequences', 'classes', 'labels', 'benchmark']:
        c = html.lower().count(kw)
        print(f'  {kw}: {c} occurrences')
except Exception as e:
    results['semantickitti'] = {'error': str(e)}
    print(f'Error: {e}')

# 3. TartanDrive2
print('\n=== TartanDrive2 Website ===')
try:
    req = urllib.request.Request('https://theairlab.org/TartanDrive2/', headers={'User-Agent': 'Mozilla/5.0'})
    html = urllib.request.urlopen(req, timeout=15).read().decode('utf-8', errors='replace')
    results['tartandrive2'] = {'html_size': len(html)}
    for kw in ['terrain', 'speed', 'trajectory', 'off-road', 'dataset']:
        c = html.lower().count(kw)
        print(f'  {kw}: {c} occurrences')
except Exception as e:
    results['tartandrive2'] = {'error': str(e)}
    print(f'Error: {e}')

# 4-8: arXiv papers
arxiv_urls = {
    'bevformer': 'https://arxiv.org/abs/2203.17270',
    'sparsead': 'https://arxiv.org/abs/2404.06892',
    'event_camera': 'https://arxiv.org/abs/1711.01458',
    'loihi_fusion': 'https://arxiv.org/html/2408.16096v1',
    'weather_lighting': 'https://arxiv.org/abs/2206.09907',
}
for name, url in arxiv_urls.items():
    print(f'\n=== {name}: {url} ===')
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        html = urllib.request.urlopen(req, timeout=15).read().decode('utf-8', errors='replace')
        results[name] = {'html_size': len(html), 'status': 'ok'}
        print(f'  OK, {len(html)} chars')
    except Exception as e:
        results[name] = {'error': str(e)}
        print(f'  Error: {e}')

# Save all metadata
with open(os.path.join(OUT_DIR, 'scraped_metadata.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print('\n=== All metadata saved to scraped_metadata.json ===')
