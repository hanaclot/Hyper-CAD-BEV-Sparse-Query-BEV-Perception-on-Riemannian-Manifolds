#!/usr/bin/env python3
"""
Mass data downloader for Hyper-CAD-BEV Experiments
Downloads datasets from real sources - SemanticKITTI, KITTI, RELLIS-3D, nuScenes, etc.
Target: 10GB+ of real training data
"""
import os, sys, json, time, hashlib, zipfile, tarfile, shutil, subprocess
from datetime import datetime
from pathlib import Path
import urllib.request
import urllib.error
import ssl

# Config
E_DRIVE = r"E:\Hyper-CAD-BEV-Experiments"
D_DRIVE = r"D:\HyperCAD_BEV_2026"
DATA_DIR = os.path.join(E_DRIVE, "data")
DOWNLOAD_LOG = os.path.join(D_DRIVE, "temp_workspace", "mass_download_log.json")
os.makedirs(os.path.join(D_DRIVE, "temp_workspace"), exist_ok=True)

ssl._create_default_https_context = ssl._create_unverified_context

class MassDownloader:
    def __init__(self):
        self.log = {"started": datetime.now().isoformat(), "downloads": [], "total_bytes": 0, "errors": []}
        self.session_start = time.time()
    
    def _save_log(self):
        with open(DOWNLOAD_LOG, 'w') as f:
            json.dump(self.log, f, indent=2, default=str)
    
    def _download_file(self, url, dest_path, description="", max_retries=3):
        """Download a single file with retries"""
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # Check if already downloaded
        if os.path.exists(dest_path):
            size = os.path.getsize(dest_path)
            if size > 1000:  # more than 1KB means valid
                print(f"  [SKIP] Already exists ({size/1e6:.1f}MB): {dest_path}")
                return size
        
        for attempt in range(max_retries):
            try:
                print(f"  [DOWNLOADING] {description}: {url[:80]}...")
                req = urllib.request.Request(url, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                
                with urllib.request.urlopen(req, timeout=120) as response:
                    total_size = int(response.headers.get('Content-Length', 0))
                    downloaded = 0
                    chunk_size = 1024 * 1024  # 1MB chunks
                    
                    with open(dest_path, 'wb') as f:
                        while True:
                            chunk = response.read(chunk_size)
                            if not chunk:
                                break
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                pct = downloaded * 100.0 / total_size
                                print(f"    {downloaded/1e6:.1f}/{total_size/1e6:.1f}MB ({pct:.0f}%)", end='\r')
                    
                    actual_size = os.path.getsize(dest_path)
                    print(f"    Complete: {actual_size/1e6:.1f}MB")
                    
                    self.log["downloads"].append({
                        "url": url, "dest": dest_path, "size": actual_size,
                        "description": description, "time": datetime.now().isoformat()
                    })
                    self.log["total_bytes"] += actual_size
                    self._save_log()
                    return actual_size
                    
            except Exception as e:
                print(f"    Attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    self.log["errors"].append({"url": url, "error": str(e), "description": description})
                    self._save_log()
        return 0

    def download_semantickitti(self):
        """Download SemanticKITTI dataset"""
        print("\n" + "="*60)
        print("PHASE 1: SemanticKITTI Dataset")
        print("="*60)
        
        base_url = "http://semantic-kitti.org/assets"
        dest = os.path.join(DATA_DIR, "semantickitti_official")
        
        # Already have labels.zip (35MB), velodyne_laser.zip (661MB in semantickitti/)
        # Need: voxels_all, calib, poses, more sequences
        
        files = [
            ("data_odometry_voxels_all.zip", "Voxels (all sequences, full)"),
            ("data_odometry_calib.zip", "Calibration files"),
            ("data_odometry_poses.zip", "Ground truth poses"),
            ("data_odometry_velodyne.zip", "Velodyne laser data (all sequences)"),
        ]
        
        total = 0
        for fname, desc in files:
            url = f"{base_url}/{fname}"
            path = os.path.join(dest, fname)
            size = self._download_file(url, path, f"SemanticKITTI: {desc}")
            total += size
        
        print(f"  SemanticKITTI total: {total/1e6:.1f}MB")
        return total

    def download_kitti_raw(self):
        """Download KITTI raw data drives"""
        print("\n" + "="*60)
        print("PHASE 2: KITTI Raw Data")
        print("="*60)
        
        dest = os.path.join(DATA_DIR, "kitti_raw")
        
        # KITTI raw sync+extract drives (each ~300-500MB)
        drives = [
            # City drives
            ("2011_09_26", ["0001", "0002", "0005", "0009", "0011", "0013", "0014", "0015", "0017", "0018", "0019", "0020", "0022", "0023", "0027", "0028", "0029", "0035", "0036", "0039", "0046", "0048", "0051", "0052", "0056", "0057", "0059", "0060", "0061", "0064", "0070", "0079", "0084", "0086", "0087", "0091", "0093", "0095", "0096", "0101", "0104", "0106", "0113", "0117"]),
            ("2011_09_28", ["0001", "0002", "0016", "0021", "0034", "0035", "0037", "0038", "0039", "0040", "0043", "0045", "0047", "0053", "0068", "0070", "0071", "0075", "0082", "0083", "0084", "0087", "0089", "0090", "0091", "0094", "0098", "0099", "0100", "0102", "0104", "0105", "0106", "0108", "0110", "0111", "0113"]),
            ("2011_09_29", ["0004", "0026", "0030", "0033", "0034", "0036", "0038", "0040", "0044", "0045", "0046", "0047", "0048", "0051", "0053", "0056", "0057", "0059", "0062", "0063", "0064", "0068", "0070", "0071", "0075", "0076", "0077", "0080", "0081", "0083", "0086", "0089", "0090", "0095", "0096", "0097", "0100", "0109", "0111", "0112", "0115"]),
            ("2011_09_30", ["0016", "0018", "0020", "0027", "0028", "0033", "0034", "0038"]),
            ("2011_10_03", ["0027", "0034", "0042", "0047", "0058"]),
        ]
        
        # Download just the important drives for training (first 3-4 per date)
        selected = []
        for date, drives_list in drives:
            for drive in drives_list[:4]:  # first 4 drives per date
                selected.append((date, drive))
        
        total = 0
        for date, drive in selected:
            url = f"https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/{date}_drive_{drive}/{date}_drive_{drive}_sync.zip"
            path = os.path.join(dest, f"{date}_drive_{drive}_sync.zip")
            size = self._download_file(url, path, f"KITTI Raw: {date} drive {drive}")
            total += size
            
            # Also download extract (tracklets, etc.)
            ext_url = f"https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/{date}_drive_{drive}/{date}_drive_{drive}_extract.zip"
            ext_path = os.path.join(dest, f"{date}_drive_{drive}_extract.zip")
            size2 = self._download_file(ext_url, ext_path, f"KITTI Raw: {date} drive {drive} extract")
            total += size2
        
        print(f"  KITTI Raw total: {total/1e6:.1f}MB")
        return total

    def download_kitti_odometry(self):
        """Download KITTI odometry dataset"""
        print("\n" + "="*60)
        print("PHASE 3: KITTI Odometry")
        print("="*60)
        
        dest = os.path.join(DATA_DIR, "kitti")
        base = "https://s3.eu-central-1.amazonaws.com/avg-kitti"
        
        files = [
            ("data_odometry_gray.zip", "Odometry grayscale images"),
            ("data_odometry_color.zip", "Odometry color images"),
            ("data_odometry_calib.zip", "Odometry calibration"),
            ("data_odometry_poses.zip", "Odometry ground truth poses"),
            ("data_odometry_velodyne.zip", "Odometry velodyne laser"),
            ("devkit_odometry.zip", "Odometry devkit"),
        ]
        
        total = 0
        for fname, desc in files:
            url = f"{base}/{fname}"
            path = os.path.join(dest, fname)
            size = self._download_file(url, path, f"KITTI: {desc}")
            total += size
        
        print(f"  KITTI Odometry total: {total/1e6:.1f}MB")
        return total

    def download_kitti_object(self):
        """Download KITTI object detection dataset"""
        print("\n" + "="*60)
        print("PHASE 4: KITTI Object Detection")
        print("="*60)
        
        dest = os.path.join(DATA_DIR, "kitti")
        base = "https://s3.eu-central-1.amazonaws.com/avg-kitti"
        
        files = [
            ("data_object_image_2.zip", "Object detection left color images"),
            ("data_object_image_3.zip", "Object detection right color images"),
            ("data_object_label_2.zip", "Object detection labels"),
            ("data_object_velodyne.zip", "Object detection velodyne"),
            ("data_object_calib.zip", "Object detection calibration"),
            ("devkit_object.zip", "Object detection devkit"),
        ]
        
        total = 0
        for fname, desc in files:
            url = f"{base}/{fname}"
            path = os.path.join(dest, fname)
            size = self._download_file(url, path, f"KITTI Object: {desc}")
            total += size
        
        print(f"  KITTI Object total: {total/1e6:.1f}MB")
        return total

    def download_rellis3d_data(self):
        """Download RELLIS-3D actual dataset files"""
        print("\n" + "="*60)
        print("PHASE 5: RELLIS-3D Dataset")
        print("="*60)
        
        dest = os.path.join(DATA_DIR, "rellis3d_data")
        
        # RELLIS-3D dataset is hosted on Google Drive
        # Try the official download links from the GitHub repo
        urls = [
            # Main dataset
            ("https://drive.google.com/uc?export=download&id=1SfQEhcfBSF6UOz7YAFOUnJQvDP1XJTHn", "RELLIS-3D dataset full"),
            # Alternative hosting
            ("https://github.com/unmannedlab/RELLIS-3D/archive/refs/heads/master.zip", "RELLIS-3D GitHub repo archive"),
        ]
        
        total = 0
        for url, desc in urls:
            fname = url.split('/')[-1].split('?')[0] or f"rellis3d_{hashlib.md5(url.encode()).hexdigest()[:8]}.zip"
            path = os.path.join(dest, fname)
            size = self._download_file(url, path, desc)
            total += size
        
        print(f"  RELLIS-3D total: {total/1e6:.1f}MB")
        return total

    def download_tartandrive2(self):
        """Download TartanDrive2 dataset samples"""
        print("\n" + "="*60)
        print("PHASE 6: TartanDrive2 Dataset")
        print("="*60)
        
        dest = os.path.join(DATA_DIR, "tartandrive2_data")
        
        # TartanDrive2 available via HuggingFace and direct links
        urls = [
            ("https://github.com/castacks/TartanDrive2/archive/refs/heads/main.zip", "TartanDrive2 GitHub repo"),
        ]
        
        # Also try direct dataset links from the AirLab
        airlab_base = "https://theairlab.org/TartanDrive2"
        additional = [
            (f"{airlab_base}/assets/sample_rosbag.tar.gz", "TartanDrive2 sample rosbag"),
        ]
        urls.extend(additional)
        
        total = 0
        for url, desc in urls:
            fname = url.split('/')[-1].split('?')[0]
            if not fname.endswith(('.zip', '.tar.gz', '.tgz', '.tar')):
                fname = f"tartandrive2_{hashlib.md5(url.encode()).hexdigest()[:8]}.zip"
            path = os.path.join(dest, fname)
            size = self._download_file(url, path, desc)
            total += size
        
        print(f"  TartanDrive2 total: {total/1e6:.1f}MB")
        return total

    def download_nuscenes_mini(self):
        """Download nuScenes mini dataset"""
        print("\n" + "="*60)
        print("PHASE 7: nuScenes Dataset")
        print("="*60)
        
        dest = os.path.join(DATA_DIR, "nuscenes")
        
        # nuScenes mini (v1.0-mini) - publicly available without registration
        urls = [
            ("https://www.nuscenes.org/data/v1.0-mini.tgz", "nuScenes v1.0-mini"),
            ("https://www.nuscenes.org/data/v1.0-trainval_meta.tgz", "nuScenes v1.0-trainval metadata"),
        ]
        
        total = 0
        for url, desc in urls:
            fname = url.split('/')[-1]
            path = os.path.join(dest, fname)
            size = self._download_file(url, path, desc)
            total += size
        
        print(f"  nuScenes total: {total/1e6:.1f}MB")
        return total

    def download_waymo_samples(self):
        """Download Waymo Open Dataset samples"""
        print("\n" + "="*60)
        print("PHASE 8: Waymo Open Dataset")
        print("="*60)
        
        dest = os.path.join(DATA_DIR, "waymo")
        
        # Waymo perception dataset tutorials and samples
        urls = [
            ("https://github.com/waymo-research/waymo-open-dataset/archive/refs/heads/master.zip", "Waymo Open Dataset toolkit"),
            ("https://waymo.com/open/data/perception/v1.4.3/waymo_open_dataset_v_1_4_3_individual_files_tutorial.tar", "Waymo tutorial data"),
        ]
        
        total = 0
        for url, desc in urls:
            fname = url.split('/')[-1].split('?')[0]
            path = os.path.join(dest, fname)
            size = self._download_file(url, path, desc)
            total += size
        
        print(f"  Waymo total: {total/1e6:.1f}MB")
        return total

    def download_event_camera_data(self):
        """Download event camera datasets"""
        print("\n" + "="*60)
        print("PHASE 9: Event Camera Datasets")
        print("="*60)
        
        dest = os.path.join(DATA_DIR, "event_camera")
        
        # DDD17 event camera driving dataset
        # MVSEC dataset samples
        urls = [
            ("https://github.com/SensorsINI/DDD17/archive/refs/heads/master.zip", "DDD17 event camera dataset repo"),
            ("https://github.com/HKBU-HPML/MVSEC/archive/refs/heads/master.zip", "MVSEC event camera dataset tools"),
            ("https://daniilidis-group.github.io/mvsec/download/mvsec_data.zip", "MVSEC sample data"),
        ]
        
        total = 0
        for url, desc in urls:
            fname = url.split('/')[-1].split('?')[0]
            if not fname.endswith(('.zip', '.tar.gz', '.tgz', '.tar')):
                fname = f"event_{hashlib.md5(url.encode()).hexdigest()[:8]}.zip"
            path = os.path.join(dest, fname)
            size = self._download_file(url, path, desc)
            total += size
        
        print(f"  Event Camera total: {total/1e6:.1f}MB")
        return total

    def download_github_repos(self):
        """Download key GitHub repositories for reference"""
        print("\n" + "="*60)
        print("PHASE 10: Reference GitHub Repositories")
        print("="*60)
        
        dest = os.path.join(DATA_DIR, "github_repos")
        
        repos = [
            ("BEVFormer", "fundamentalvision/BEVFormer"),
            ("BEVDet", "HuangJunJie2017/BEVDet"),
            ("Sparse4D", "linxc06700/Sparse4D"),
            ("SparseBEV", "Megvii-BaseDetection/SparseBEV"),
            ("SparseAD", "opendrivelab/SparseAD"),
            ("MMDetection3D", "open-mmlab/mmdetection3d"),
            ("OpenPCDet", "open-mmlab/OpenPCDet"),
            ("Loihi2-tutorials", "intel-nrc-ecosystem/intel-nrc-loihi2-tutorials"),
        ]
        
        total = 0
        for name, repo_path in repos:
            url = f"https://github.com/{repo_path}/archive/refs/heads/main.zip"
            path = os.path.join(dest, f"{name}.zip")
            size = self._download_file(url, path, f"GitHub: {name}")
            total += size
        
        print(f"  GitHub repos total: {total/1e6:.1f}MB")
        return total

    def download_arxiv_supplementary(self):
        """Download arxiv paper PDFs with supplementary"""
        print("\n" + "="*60)
        print("PHASE 11: ArXiv Papers with Supplementary")
        print("="*60)
        
        dest = os.path.join(DATA_DIR, "arxiv_supplementary")
        
        papers = [
            ("2203.17270", "BEVFormer"),
            ("2404.06892", "SparseAD"),
            ("1711.01458", "Event Camera Event-based Vision"),
            ("2408.16096", "Loihi Sensor Fusion"),
            ("2206.09907", "Weather Robustness Off-road"),
            ("2311.11722", "Sparse4D"),
            ("2308.09244", "SparseBEV"),
            ("2112.11790", "BEVDet"),
            ("2205.02833", "BEVerse"),
            ("2211.12786", "PolarBEV"),
            ("2305.09910", "FB-BEV"),
        ]
        
        total = 0
        for arxiv_id, name in papers:
            url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            path = os.path.join(dest, f"{arxiv_id}_{name}.pdf")
            size = self._download_file(url, path, f"ArXiv: {name} ({arxiv_id})")
            total += size
        
        print(f"  ArXiv papers total: {total/1e6:.1f}MB")
        return total

    def extract_archives(self):
        """Extract downloaded archives"""
        print("\n" + "="*60)
        print("EXTRACTING ARCHIVES")
        print("="*60)
        
        for root, dirs, files in os.walk(DATA_DIR):
            for f in files:
                if f.endswith('.zip'):
                    fpath = os.path.join(root, f)
                    extract_dir = fpath.replace('.zip', '')
                    if not os.path.exists(extract_dir):
                        try:
                            print(f"  Extracting: {fpath}")
                            with zipfile.ZipFile(fpath, 'r') as zf:
                                zf.extractall(extract_dir)
                            print(f"    -> {extract_dir}")
                        except Exception as e:
                            print(f"    Error: {e}")
                elif f.endswith('.tar.gz') or f.endswith('.tgz'):
                    fpath = os.path.join(root, f)
                    extract_dir = fpath.replace('.tar.gz', '').replace('.tgz', '')
                    if not os.path.exists(extract_dir):
                        try:
                            print(f"  Extracting: {fpath}")
                            with tarfile.open(fpath, 'r:gz') as tf:
                                tf.extractall(extract_dir)
                            print(f"    -> {extract_dir}")
                        except Exception as e:
                            print(f"    Error: {e}")

    def run_all(self):
        """Run all download phases"""
        print("="*60)
        print("MASS DATA DOWNLOAD FOR HYPER-CAD-BEV EXPERIMENTS")
        print("="*60)
        print(f"Target: 10GB+ real training data")
        print(f"E-Drive: {E_DRIVE}")
        print(f"D-Workspace: {D_DRIVE}")
        print("="*60)
        
        phases = [
            ("SemanticKITTI", self.download_semantickitti),
            ("KITTI Raw", self.download_kitti_raw),
            ("KITTI Odometry", self.download_kitti_odometry),
            ("KITTI Object", self.download_kitti_object),
            ("RELLIS-3D Data", self.download_rellis3d_data),
            ("TartanDrive2", self.download_tartandrive2),
            ("nuScenes", self.download_nuscenes_mini),
            ("Waymo", self.download_waymo_samples),
            ("Event Camera", self.download_event_camera_data),
            ("GitHub Repos", self.download_github_repos),
            ("ArXiv Papers", self.download_arxiv_supplementary),
        ]
        
        grand_total = 0
        for name, fn in phases:
            try:
                size = fn()
                grand_total += size
                print(f"  [OK] {name}: {size/1e6:.1f}MB")
            except Exception as e:
                print(f"  [FAIL] {name}: {e}")
                self.log["errors"].append({"phase": name, "error": str(e)})
        
        # Extract archives
        try:
            self.extract_archives()
        except Exception as e:
            print(f"  [FAIL] Extraction: {e}")
        
        # Final summary
        elapsed = time.time() - self.session_start
        final_size = 0
        for root, dirs, files in os.walk(DATA_DIR):
            for f in files:
                fp = os.path.join(root, f)
                if os.path.exists(fp):
                    final_size += os.path.getsize(fp)
        
        print("\n" + "="*60)
        print("DOWNLOAD COMPLETE")
        print("="*60)
        print(f"  Downloaded: {grand_total/1e6:.1f} MB")
        print(f"  Total data on disk: {final_size/1e9:.2f} GB")
        print(f"  Elapsed: {elapsed/60:.1f} minutes")
        print(f"  Errors: {len(self.log['errors'])}")
        
        self.log["completed"] = datetime.now().isoformat()
        self.log["final_size_bytes"] = final_size
        self.log["elapsed_seconds"] = elapsed
        self._save_log()
        
        return grand_total

if __name__ == "__main__":
    downloader = MassDownloader()
    downloader.run_all()