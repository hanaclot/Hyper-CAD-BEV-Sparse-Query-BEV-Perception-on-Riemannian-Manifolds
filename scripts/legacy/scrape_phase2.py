# -*- coding: utf-8 -*-
# Phase 2 scraping - remaining sources
import os, json, urllib.request
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(r'E:\HyperCAD_BEV_Replication_2026\data')

def safe_get(url, timeout=40):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception as e:
        print(f'  FAIL: {url[:60]} - {e}')
        return None

results = []

# BEVFormer paper PDF
print('BEVFormer PDF...')
c = safe_get('https://arxiv.org/pdf/2203.17270.pdf', 120)
if c:
    (DATA_DIR/'bevformer'/'paper.pdf').write_bytes(c)
    results.append(('BEVFormer PDF', len(c)))
    print(f'  OK: {len(c)} bytes')

# SparseAD
print('SparseAD...')
c = safe_get('http://export.arxiv.org/api/query?id_list=2404.06892&max_results=1')
if c:
    (DATA_DIR/'sparsead'/'metadata.xml').write_bytes(c)
    results.append(('SparseAD meta', len(c)))
    print(f'  OK: {len(c)} bytes')

print('SparseAD PDF...')
c = safe_get('https://arxiv.org/pdf/2404.06892.pdf', 120)
if c:
    (DATA_DIR/'sparsead'/'paper.pdf').write_bytes(c)
    results.append(('SparseAD PDF', len(c)))
    print(f'  OK: {len(c)} bytes')

# Event camera
print('EventCamera...')
c = safe_get('http://export.arxiv.org/api/query?id_list=1711.01458&max_results=1')
if c:
    (DATA_DIR/'event_camera'/'metadata.xml').write_bytes(c)
    results.append(('EventCamera meta', len(c)))

print('EventCamera PDF...')
c = safe_get('https://arxiv.org/pdf/1711.01458.pdf', 120)
if c:
    (DATA_DIR/'event_camera'/'paper.pdf').write_bytes(c)
    results.append(('EventCamera PDF', len(c)))

# Loihi fusion
print('LoihiFusion...')
c = safe_get('http://export.arxiv.org/api/query?id_list=2408.16096&max_results=1')
if c:
    (DATA_DIR/'loihi_fusion'/'metadata.xml').write_bytes(c)
    results.append(('LoihiFusion meta', len(c)))

# SemanticKITTI
print('SemanticKITTI...')
c = safe_get('http://semantic-kitti.org/')
if c:
    (DATA_DIR/'semantickitti'/'index.html').write_bytes(c)
    results.append(('SemanticKITTI index', len(c)))

c = safe_get('http://semantic-kitti.org/dataset.html')
if c:
    (DATA_DIR/'semantickitti'/'dataset.html').write_bytes(c)
    results.append(('SemanticKITTI dataset', len(c)))

# RELLIS-3D code files
print('RELLIS-3D code...')
py_files = ['rellis.py','cityscapes.py','config.py','loss.py','train.py','transforms.py','base_dataset.py','__init__.py']
for f in py_files:
    c = safe_get(f'https://raw.githubusercontent.com/unmannedlab/RELLIS-3D/master/{f}')
    if c:
        (DATA_DIR/'rellis3d'/f).write_bytes(c)
        results.append((f'RELLIS-3D/{f}', len(c)))
        print(f'  OK: {f} ({len(c)} bytes)')

# Summary
with open(DATA_DIR/'processed'/'phase2_results.json', 'w') as f:
    json.dump(results, f)
print(f'\nDone! {len(results)} files downloaded.')
