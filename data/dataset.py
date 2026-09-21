import os
import glob
import re
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.signal_processing import compute_rwave_slope_edr, ecg_to_scalogram

class UniversalSleepDataset(Dataset):
    def __init__(self, pkl_file_list, model_type="pimtl"):
        self.model_type = model_type
        self.samples = []
        
        for p in pkl_file_list:
            with open(p, "rb") as f:
                d = pickle.load(f)
            p_id = d["patient_id"]
            for i in range(len(d["labels"])):
                self.samples.append({
                    "ecg": d["segments"][i],
                    "label": d["labels"][i],
                    "ahi": d["ahi"],
                    "cpc": d["cpc"][i],
                    "patient_id": p_id
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        ecg = torch.tensor(item["ecg"], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(item["label"], dtype=torch.long)
        ahi = torch.tensor(item["ahi"], dtype=torch.float32)
        cpc = torch.tensor(item["cpc"], dtype=torch.float32)

        if self.model_type == 'pimtl':
            return ecg, label, ahi, cpc

        elif self.model_type == 'pase_mst':
            raw = item["ecg"]
            edr = compute_rwave_slope_edr(raw, fs=100)
            ra = torch.tensor(edr, dtype=torch.float32).unsqueeze(0)
            rri = torch.tensor(edr, dtype=torch.float32).unsqueeze(0)
            rrid = torch.tensor(np.gradient(edr), dtype=torch.float32).unsqueeze(0)
            cpc_seq = torch.full((1, 6000), fill_value=float(item["cpc"][2]), dtype=torch.float32)
            return ecg, ra, rri, rrid, cpc_seq, label

        elif self.model_type == 'dream':
            scalogram = ecg_to_scalogram(item["ecg"], target_size=(224, 224))
            return torch.tensor(scalogram, dtype=torch.float32), label

        elif self.model_type == 'cnn_bigru':
            return ecg, label

        elif self.model_type == 'cnn_transformer_lstm':
            edr = compute_rwave_slope_edr(item["ecg"], fs=100)
            feat = torch.tensor(np.stack([edr, np.gradient(edr)]), dtype=torch.float32)
            return feat, label

def get_benchmark_loaders(precomputed_dir="/kaggle/working/precomputed", model_type="pimtl", batch_size=64):
    all_files = sorted(glob.glob(os.path.join(precomputed_dir, "*.pkl")))
    
    apnea_train_files = [p for p in all_files if re.search(r"apnea_ecg_[abc]\d{2}\.pkl$", p)]
    apnea_test_files  = [p for p in all_files if re.search(r"apnea_ecg_x\d{2}\.pkl$", p)]
    ucddb_test_files  = [p for p in all_files if "ucddb_" in p]

    train_ds = UniversalSleepDataset(apnea_train_files, model_type=model_type)
    val_ds   = UniversalSleepDataset(apnea_test_files, model_type=model_type)
    ucddb_ds = UniversalSleepDataset(ucddb_test_files, model_type=model_type)

    train_labels = [s["label"] for s in train_ds.samples]
    n_neg = sum(1 for l in train_labels if l == 0)
    n_pos = sum(1 for l in train_labels if l == 1)
    class_weights = [1.0, n_neg / max(n_pos, 1)]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    ucddb_loader = DataLoader(ucddb_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, ucddb_loader, class_weights