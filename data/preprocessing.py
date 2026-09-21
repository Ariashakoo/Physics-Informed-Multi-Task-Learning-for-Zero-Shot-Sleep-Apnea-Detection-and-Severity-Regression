import os
import re
import glob
import pickle
import warnings
from pathlib import Path
import numpy as np
import mne
import wfdb
from tqdm import tqdm
from utils.signal_processing import compute_cpc

warnings.filterwarnings("ignore")

PRECOMP_DIR = "/kaggle/working/precomputed"
os.makedirs(PRECOMP_DIR, exist_ok=True)

def process_apnea_ecg():
    target_dir = "/kaggle/working/apnea_ecg_clean"
    os.makedirs(target_dir, exist_ok=True)
    
    canonical_pattern = re.compile(r"^[abcx]\d{2}\.(hea|dat|apn)$", re.IGNORECASE)
    for p in Path("/kaggle/input").rglob("*"):
        if p.is_file() and canonical_pattern.match(p.name):
            dest = Path(target_dir) / p.name.lower()
            if not dest.exists(): os.symlink(p, dest)
                
    valid_records = sorted(list({f.stem for f in Path(target_dir).glob("*.hea") 
                                 if (Path(target_dir) / f"{f.stem}.apn").exists() 
                                 and (Path(target_dir) / f"{f.stem}.dat").exists()}))
    
    print(f"[*] Extracting {len(valid_records)} Apnea-ECG records...")
    for rec in tqdm(valid_records, desc="Apnea-ECG"):
        out_path = os.path.join(PRECOMP_DIR, f"apnea_ecg_{rec}.pkl")
        if os.path.exists(out_path): continue
            
        record_path = os.path.join(target_dir, rec)
        record = wfdb.rdrecord(record_path)
        ecg = record.p_signal[:, 0]
        fs = record.fs
        
        ann = wfdb.rdann(record_path, extension='apn')
        labels = [1 if sym == 'A' else 0 for sym in ann.symbol]
        
        samples_per_min = 60 * fs
        n_minutes = min(len(labels), len(ecg) // samples_per_min)
        
        segments, minute_labels = [], []
        for i in range(n_minutes):
            seg = ecg[i * samples_per_min : (i + 1) * samples_per_min]
            if np.isnan(seg).any() or np.isinf(seg).any(): continue
            seg = (seg - np.mean(seg)) / (np.std(seg) + 1e-8)
            segments.append(seg)
            minute_labels.append(labels[i])
            
        if len(segments) == 0: continue
        segments = np.array(segments, dtype=np.float32)
        minute_labels = np.array(minute_labels, dtype=np.int64)
        ahi = float(np.sum(minute_labels) / (len(minute_labels) / 60.0))
        
        cpc_features = []
        for s in segments:
            c = compute_cpc(s, fs=100)
            cpc_features.append(np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0))
            
        with open(out_path, "wb") as f:
            pickle.dump({
                "dataset": "apnea_ecg", "patient_id": rec,
                "segments": segments, "labels": minute_labels,
                "ahi": ahi, "cpc": np.array(cpc_features, dtype=np.float32)
            }, f)

def process_ucddb():
    bridge_dir = "/kaggle/working/ucddb_edf_bridge"
    os.makedirs(bridge_dir, exist_ok=True)
    records = {}
    
    for f in Path("/kaggle/input").rglob("ucddb*"):
        match = re.search(r"(ucddb\d{3})", f.name.lower())
        if not match: continue
        stem = match.group(1)
        if stem not in records: records[stem] = {"signal": None, "annotation": None}
            
        fname_lower = f.name.lower()
        if fname_lower.endswith(".rec"): records[stem]["signal"] = str(f)
        elif fname_lower.endswith("_lifecard.edf") and records[stem]["signal"] is None:
            records[stem]["signal"] = str(f)
        if "_respevt" in fname_lower: records[stem]["annotation"] = str(f)

    valid_records = {}
    for stem, paths in records.items():
        if paths["signal"] is not None:
            dest_sig = os.path.join(bridge_dir, f"{stem}.edf")
            if not os.path.exists(dest_sig): os.symlink(paths["signal"], dest_sig)
            valid_records[stem] = {"signal": dest_sig, "annotation": paths["annotation"]}
            
    print(f"[*] Extracting {len(valid_records)} UCDDB records...")
    import scipy.signal as signal
    for stem, paths in tqdm(sorted(valid_records.items()), desc="UCDDB"):
        out_path = os.path.join(PRECOMP_DIR, f"ucddb_{stem}.pkl")
        if os.path.exists(out_path): continue
            
        raw = mne.io.read_raw_edf(paths["signal"], preload=True, verbose=False)
        ecg_ch = next((ch for ch in raw.ch_names if "ECG" in ch.upper() or "EKG" in ch.upper()), raw.ch_names[0])
        orig_fs = int(raw.info["sfreq"])
        raw_ecg = raw.get_data(picks=[ecg_ch])[0]
        
        target_fs = 100
        resampled_ecg = signal.resample(raw_ecg, int(len(raw_ecg) * (target_fs / orig_fs)))
        samples_per_min = 60 * target_fs
        total_minutes = len(resampled_ecg) // samples_per_min
        
        minute_labels = np.zeros(total_minutes, dtype=np.int64)
        rec_start_sec = raw.info["meas_date"].hour * 3600 + raw.info["meas_date"].minute * 60 + raw.info["meas_date"].second if raw.info.get("meas_date") else None
        
        if paths["annotation"] and os.path.exists(paths["annotation"]):
            with open(paths["annotation"], "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    match = re.search(r"(\d{1,2}):(\d{2}):(\d{2})", line.strip())
                    if not match or not any(k in line.upper() for k in ["APNEA", "HYP", "A-O", "A-C", "A-M", "H-O", "H-C"]): continue
                    event_sec = int(match.group(1)) * 3600 + int(match.group(2)) * 60 + int(match.group(3))
                    
                    duration = 10.0
                    for token in line.strip().split():
                        try:
                            if 5.0 <= float(token) <= 240.0: duration = float(token); break
                        except ValueError: pass
                            
                    if event_sec >= total_minutes * 60 and rec_start_sec is not None:
                        adjusted = (event_sec - rec_start_sec) % 86400
                        if adjusted < total_minutes * 60: event_sec = adjusted
                            
                    min_start = int(event_sec // 60)
                    min_end = int((event_sec + duration) // 60)
                    for m_idx in range(min_start, min(min_end + 1, total_minutes)):
                        minute_labels[m_idx] = 1
                        
        segments = []
        for i in range(total_minutes):
            seg = resampled_ecg[i * samples_per_min : (i + 1) * samples_per_min]
            seg = (seg - np.mean(seg)) / (np.std(seg) + 1e-8)
            segments.append(seg)
            
        segments = np.array(segments, dtype=np.float32)
        ahi = float(np.sum(minute_labels) / (total_minutes / 60.0))
        
        cpc_features = []
        for s in segments:
            c = compute_cpc(s, fs=100)
            cpc_features.append(np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0))
            
        with open(out_path, "wb") as f:
            pickle.dump({
                "dataset": "ucddb", "patient_id": stem,
                "segments": segments, "labels": minute_labels,
                "ahi": ahi, "cpc": np.array(cpc_features, dtype=np.float32)
            }, f)

def clean_data():
    pkl_files = glob.glob(os.path.join(PRECOMP_DIR, "*.pkl"))
    purged_duplicates = 0
    valid_files = []

    for p in pkl_files:
        fname = os.path.basename(p)
        if re.match(r"^apnea_ecg_[abc]\d{2}r\.pkl$", fname, re.IGNORECASE):
            os.remove(p)
            purged_duplicates += 1
        else:
            valid_files.append(p)

    print(f"[*] Purged {purged_duplicates} duplicate '*r' records.")

    total_dropped_segments = 0
    for p in sorted(valid_files):
        with open(p, "rb") as f:
            data = pickle.load(f)
            
        segs = data["segments"]
        labels = data["labels"]
        cpc = data["cpc"]
        
        if np.isnan(cpc).any() or np.isinf(cpc).any():
            cpc = np.nan_to_num(cpc, nan=0.0, posinf=0.0, neginf=0.0)
            data["cpc"] = cpc
            
        nan_mask = np.isnan(segs).any(axis=1) | np.isinf(segs).any(axis=1)
        
        if np.any(nan_mask):
            valid_mask = ~nan_mask
            dropped = int(np.sum(nan_mask))
            total_dropped_segments += dropped
            
            data["segments"] = segs[valid_mask]
            data["labels"] = labels[valid_mask]
            data["cpc"] = cpc[valid_mask]
            
            valid_minutes = len(data["labels"])
            if valid_minutes > 0:
                data["ahi"] = float(np.sum(data["labels"]) / (valid_minutes / 60.0))
            else:
                data["ahi"] = 0.0
                
        with open(p, "wb") as f:
            pickle.dump(data, f)

    print(f"[*] Dropped {total_dropped_segments} corrupted (lead-off) epochs across all files.")

if __name__ == "__main__":
    existing_files = glob.glob(os.path.join(PRECOMP_DIR, "*.pkl"))
    if len(existing_files) < 95:
        print("[!] Fresh environment detected. Auto-generating all features...")
        process_apnea_ecg()
        process_ucddb()
        print("[✓] All 95 patient feature files generated successfully.")
    else:
        print(f"[✓] Found {len(existing_files)} precomputed files. Skipping extraction to save time.")
    clean_data()