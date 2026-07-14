#!/usr/bin/env python3
import os, sys, json, time, math, warnings, struct, glob, random, csv, numpy as np
from datetime import datetime
from collections import defaultdict
warnings.filterwarnings('ignore')

EXPERIMENT_VERSION = 'v9.0'
EXPERIMENT_ROOT = r'D:\HyperCAD_BEV_2026\experiments9'
E_DATA_ROOT = r'E:\Hyper-CAD-BEV-Experiments\data'
RESULTS_DIR = os.path.join(EXPERIMENT_ROOT, 'results')
FIGURES_DIR = os.path.join(EXPERIMENT_ROOT, 'figures')
for d in [RESULTS_DIR, FIGURES_DIR, os.path.join(EXPERIMENT_ROOT,'logs')]:
    os.makedirs(d, exist_ok=True)

MANIFOLD_PARAMS = {'road_width':3.5,'ridge_width':0.5,'D_diffusion_road':0.8,'D_diffusion_boundary':0.01,'gamma_reaction':0.5,'lambda_TV':0.01,'mu_sparsity':0.005,'alpha_pde':0.1,'query_budget_K':300}
SEED=42; np.random.seed(SEED); random.seed(SEED)

def load_lidar_bin(fp):
    d=np.fromfile(fp,dtype=np.float32)
    return d.reshape(-1,4)

def find_lidar_files(root):
    files=[]
    for r,_,fs in os.walk(root):
        for f in fs:
            if f.endswith('.bin'): files.append(os.path.join(r,f))
    return sorted(files)

def project_to_bev(points,grid_size=256,resolution=0.2,z_range=(-3,3)):
    x,y,z=points[:,0],points[:,1],points[:,2]
    intensity=points[:,3] if points.shape[1]>3 else np.ones(len(points))
    mask=(z>=z_range[0])&(z<=z_range[1])
    x,y,z=x[mask],y[mask],z[mask]; intensity=intensity[mask]
    if len(x)==0: return np.zeros((grid_size,grid_size)),np.zeros((grid_size,grid_size)),np.zeros((grid_size,grid_size))
    half=grid_size*resolution/2
    xi=((x+half)/resolution).astype(np.int32); yi=((y+half)/resolution).astype(np.int32)
    valid=(xi>=0)&(xi<grid_size)&(yi>=0)&(yi<grid_size)
    xi,yi,z=xi[valid],yi[valid],z[valid]; intensity=intensity[valid]
    bh=np.full((grid_size,grid_size),-10.0); bi=np.zeros((grid_size,grid_size)); bd=np.zeros((grid_size,grid_size))
    for i in range(len(xi)):
        if z[i]>bh[yi[i],xi[i]]: bh[yi[i],xi[i]]=z[i]
        bi[yi[i],xi[i]]+=intensity[i]; bd[yi[i],xi[i]]+=1
    m=bd>0; bi[m]/=bd[m]
    return bh,bi,bd

def compute_bev_semantic(bh,bi,bd,road_level=0.3,obs_thresh=0.5):
    from scipy.ndimage import uniform_filter
    H,W=bh.shape; sem=np.zeros((H,W),dtype=np.int32)
    no_points=bd==0
    hlm=uniform_filter(bh,size=9); hlv=uniform_filter(bh**2,size=9)-hlm**2
    is_low=bh<road_level; is_dense=bd>np.percentile(bd[~no_points],30); is_flat=np.sqrt(np.maximum(hlv,0))<0.15
    road=(~no_points)&is_low&is_flat; sem[road]=1
    is_tall=bh>obs_thresh; is_sparse=(bd>0)&(bd<np.percentile(bd[~no_points],50))
    obstacle=(~no_points)&is_tall&is_sparse&~road; sem[obstacle]=2
    is_mid=(~is_low)&(~is_tall); terrain=(~no_points)&is_mid&~road&~obstacle; sem[terrain]=3
    is_high_var=np.sqrt(np.maximum(hlv,0))>0.3
    veg=(~no_points)&is_tall&is_high_var&~obstacle&~road; sem[veg]=4
    return sem

print('Script v9 started successfully - models to be imported...')
