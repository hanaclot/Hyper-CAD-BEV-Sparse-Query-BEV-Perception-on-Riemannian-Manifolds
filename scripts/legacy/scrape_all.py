# -*- coding: utf-8 -*-
import os, sys, json, time, urllib.request, urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(r'E:\HyperCAD_BEV_Replication_2026')
DATA_DIR = PROJECT_ROOT / 'data'
LOG_FILE = DATA_DIR / 'processed' / 'scraping_log.json'

os.makedirs(DATA_DIR / 'processed', exist_ok=True)

log_entries = []

def log(source, status, detail=''):
    entry = {'source': source, 'status': status, 'detail': str(detail), 'time': datetime.now().isoformat()}
    log_entries.append(entry)
    print(f'[{status}] {source}: {detail}')

def safe_request(url, timeout=30):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read(), resp.getcode()
    except Exception as e:
        return None, str(e)

def scrape_rellis3d():
    target = DATA_DIR / 'rellis3d'
    os.makedirs(target, exist_ok=True)
    
    content, code = safe_request('https://api.github.com/repos/unmannedlab/RELLIS-3D')
    if content:
        (target / 'repo_metadata.json').write_bytes(content)
        log('RELLIS-3D', 'OK', f'Repo metadata, {len(content)} bytes')
    else:
        log('RELLIS-3D', 'FAIL', f'repo metadata')
    
    content, code = safe_request('https://raw.githubusercontent.com/unmannedlab/RELLIS-3D/master/README.md')
    if content:
        (target / 'README.md').write_bytes(content)
        log('RELLIS-3D', 'OK', f'README, {len(content)} bytes')
    
    content, code = safe_request('https://unmannedlab.github.io/research/RELLIS-3D')
    if content:
        (target / 'dataset_page.html').write_bytes(content)
        log('RELLIS-3D', 'OK', f'Dataset page, {len(content)} bytes')
    
    py_files = ['rellis.py','cityscapes.py','cityscapes_labels.py','config.py','loss.py','train.py','transforms.py','base_dataset.py']
    for fname in py_files:
        url = f'https://raw.githubusercontent.com/unmannedlab/RELLIS-3D/master/{fname}'
        content, code = safe_request(url)
        if content:
            (target / fname).write_bytes(content)
            log('RELLIS-3D', 'OK', f'{fname}, {len(content)} bytes')

def scrape_weather():
    target = DATA_DIR / 'weather_arxiv'
    os.makedirs(target, exist_ok=True)
    arxiv_id = '2206.09907'
    
    content, code = safe_request(f'http://export.arxiv.org/api/query?id_list={arxiv_id}&max_results=1')
    if content:
        (target / 'metadata.xml').write_bytes(content)
        log('Weather/ArXiv', 'OK', f'Metadata, {len(content)} bytes')
    
    content, code = safe_request(f'https://arxiv.org/abs/{arxiv_id}')
    if content:
        (target / 'abstract_page.html').write_bytes(content)
        log('Weather/ArXiv', 'OK', f'Abstract page, {len(content)} bytes')
    
    content, code = safe_request(f'https://arxiv.org/pdf/{arxiv_id}.pdf', timeout=60)
    if content:
        (target / 'paper.pdf').write_bytes(content)
        log('Weather/ArXiv', 'OK', f'PDF, {len(content)} bytes')

def scrape_tartandrive2():
    target = DATA_DIR / 'tartandrive2'
    os.makedirs(target, exist_ok=True)
    
    content, code = safe_request('https://theairlab.org/TartanDrive2/')
    if content:
        (target / 'index.html').write_bytes(content)
        log('TartanDrive2', 'OK', f'Main page, {len(content)} bytes')
    
    content, code = safe_request('https://api.github.com/repos/castacks/TartanDrive2')
    if content:
        (target / 'repo_metadata.json').write_bytes(content)
        log('TartanDrive2', 'OK', f'Repo metadata, {len(content)} bytes')

def scrape_bevformer():
    target = DATA_DIR / 'bevformer'
    os.makedirs(target, exist_ok=True)
    arxiv_id = '2203.17270'
    
    content, code = safe_request(f'http://export.arxiv.org/api/query?id_list={arxiv_id}&max_results=1')
    if content:
        (target / 'metadata.xml').write_bytes(content)
        log('BEVFormer', 'OK', f'Metadata, {len(content)} bytes')
    
    content, code = safe_request(f'https://arxiv.org/pdf/{arxiv_id}.pdf', timeout=60)
    if content:
        (target / 'paper.pdf').write_bytes(content)
        log('BEVFormer', 'OK', f'PDF, {len(content)} bytes')

def scrape_sparsead():
    target = DATA_DIR / 'sparsead'
    os.makedirs(target, exist_ok=True)
    arxiv_id = '2404.06892'
    
    content, code = safe_request(f'http://export.arxiv.org/api/query?id_list={arxiv_id}&max_results=1')
    if content:
        (target / 'metadata.xml').write_bytes(content)
        log('SparseAD', 'OK', f'Metadata, {len(content)} bytes')
    
    content, code = safe_request(f'https://arxiv.org/pdf/{arxiv_id}.pdf', timeout=60)
    if content:
        (target / 'paper.pdf').write_bytes(content)
        log('SparseAD', 'OK', f'PDF, {len(content)} bytes')
    
    content, code = safe_request(f'https://arxiv.org/html/{arxiv_id}v1')
    if content:
        (target / 'full_text.html').write_bytes(content)
        log('SparseAD', 'OK', f'HTML full text, {len(content)} bytes')

def scrape_event_camera():
    target = DATA_DIR / 'event_camera'
    os.makedirs(target, exist_ok=True)
    arxiv_id = '1711.01458'
    
    content, code = safe_request(f'http://export.arxiv.org/api/query?id_list={arxiv_id}&max_results=1')
    if content:
        (target / 'metadata.xml').write_bytes(content)
        log('EventCamera', 'OK', f'Metadata, {len(content)} bytes')
    
    content, code = safe_request(f'https://arxiv.org/pdf/{arxiv_id}.pdf', timeout=60)
    if content:
        (target / 'paper.pdf').write_bytes(content)
        log('EventCamera', 'OK', f'PDF, {len(content)} bytes')

def scrape_loihi():
    target = DATA_DIR / 'loihi_fusion'
    os.makedirs(target, exist_ok=True)
    arxiv_id = '2408.16096'
    
    content, code = safe_request(f'http://export.arxiv.org/api/query?id_list={arxiv_id}&max_results=1')
    if content:
        (target / 'metadata.xml').write_bytes(content)
        log('LoihiFusion', 'OK', f'Metadata, {len(content)} bytes')
    
    content, code = safe_request(f'https://arxiv.org/html/{arxiv_id}v1')
    if content:
        (target / 'full_text.html').write_bytes(content)
        log('LoihiFusion', 'OK', f'HTML full text, {len(content)} bytes')

def scrape_semantickitti():
    target = DATA_DIR / 'semantickitti'
    os.makedirs(target, exist_ok=True)
    
    content, code = safe_request('http://semantic-kitti.org/')
    if content:
        (target / 'index.html').write_bytes(content)
        log('SemanticKITTI', 'OK', f'Main page, {len(content)} bytes')
    
    content, code = safe_request('http://semantic-kitti.org/dataset.html')
    if content:
        (target / 'dataset.html').write_bytes(content)
        log('SemanticKITTI', 'OK', f'Dataset page, {len(content)} bytes')

print('='*60)
print('Hyper-CAD-BEV v6.5-Sparse Data Scraping')
print(f'Start: {datetime.now().isoformat()}')
print('='*60)

scrapers = [
    ('RELLIS-3D', scrape_rellis3d),
    ('Weather/ArXiv', scrape_weather),
    ('TartanDrive2', scrape_tartandrive2),
    ('BEVFormer', scrape_bevformer),
    ('SparseAD', scrape_sparsead),
    ('EventCamera', scrape_event_camera),
    ('LoihiFusion', scrape_loihi),
    ('SemanticKITTI', scrape_semantickitti),
]

for name, func in scrapers:
    print(f'\n--- Scraping: {name} ---')
    try:
        func()
    except Exception as e:
        log(name, 'ERROR', str(e))

with open(LOG_FILE, 'w', encoding='utf-8') as f:
    json.dump(log_entries, f, indent=2, ensure_ascii=False)

print(f'\nDone! Total: {len(log_entries)} records')
print(f'Log: {LOG_FILE}')
