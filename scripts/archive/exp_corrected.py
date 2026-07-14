# -*- coding: utf-8 -*-
import os, json, csv, time, math, gc
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore")

BEV_SIZE = 200; BEV_RANGE = 50.0; BEV_RES = 0.25
N_SAMPLES = 40; N_CLASSES = 20; SPARSE_RATIO = 0.25

PROJECT = Path(r"E:\Hyper-CAD-BEV-Experiments")
DATA_ROOT = PROJECT / "data"
RDIR = PROJECT / "experiments" / "results_dep"
FDIR = PROJECT / "experiments" / "figures_dep"
RDIR.mkdir(parents=True, exist_ok=True)
FDIR.mkdir(parents=True, exist_ok=True)

LEARNING_MAP = {0:0,1:0,10:1,11:2,13:5,15:3,16:5,18:4,20:5,30:6,31:7,32:8,40:9,44:10,48:11,49:12,50:13,51:14,52:0,60:0,70:15,71:16,72:17,80:18,81:19,99:0,252:1,253:7,254:7,255:8,256:5,257:5,258:7,259:7}

ll = []; _log = []; _t0 = time.time()
def log(msg):
    t = datetime.now().strftime("%H:%M:%S"); line = f"[{t}] {msg}"
    print(line); _log.append(line)

log("="*60)
log("HYPER-CAD-BEV v6.5-Sparse: CORRECTED EXPERIMENT")
log("="*60)

print("Script header loaded OK")
