import os
import re
import glob
import time
import copy
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, mean_absolute_error

from data.dataset import UniversalSleepDataset
from models.architectures import build_model_and_criterion

@torch.no_grad()
def evaluate_cohort(model, loader, model_type, device):
    model.eval()
    all_preds, all_targets = [], []
    all_ahi_preds, all_ahi_true = [], []
    
    for batch in loader:
        batch = [b.to(device) for b in batch]
        if model_type == 'pimtl':
            ecg, target, ahi, _ = batch
            logits, ahi_pred, _ = model(ecg)
            all_ahi_preds.extend(ahi_pred.view(-1).cpu().numpy())
            all_ahi_true.extend(ahi.view(-1).cpu().numpy())
        elif model_type == 'pase_mst':
            logits = model(*batch[:5])
            target = batch[5]
        else:
            logits = model(batch[0])
            target = batch[1]
            
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(target.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = accuracy_score(all_targets, all_preds) * 100.0
    macro_f1 = f1_score(all_targets, all_preds, average='macro') * 100.0
    sens = recall_score(all_targets, all_preds, pos_label=1, zero_division=0) * 100.0
    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds, labels=[0, 1]).ravel()
    spec = (tn / (tn + fp)) * 100.0 if (tn + fp) > 0 else 0.0

    res = {
        "Acc": acc,
        "Macro_F1": macro_f1,
        "Sens": sens,
        "Spec": spec
    }
    if model_type == 'pimtl' and len(all_ahi_preds) > 0:
        res["AHI_MAE"] = mean_absolute_error(all_ahi_true, all_ahi_preds)
    return res

if __name__ == "__main__":
    precomputed_dir = "/kaggle/working/precomputed"
    all_files = sorted(glob.glob(os.path.join(precomputed_dir, "*.pkl")))

    apnea_train_files = [p for p in all_files if re.search(r"apnea_ecg_[abc]\d{2}\.pkl$", p)]
    apnea_test_files  = [p for p in all_files if re.search(r"apnea_ecg_x\d{2}\.pkl$", p)]
    ucddb_test_files  = [p for p in all_files if "ucddb_" in p]

    train_meta_ds = UniversalSleepDataset(apnea_train_files, model_type="pimtl")
    train_labels = [s["label"] for s in train_meta_ds.samples]
    n_neg = sum(1 for l in train_labels if l == 0)
    n_pos = sum(1 for l in train_labels if l == 1)
    class_weights = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float32)
    del train_meta_ds

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    models_to_run = ['pimtl', 'pase_mst', 'cnn_transformer_lstm']
    epochs = 30
    batch_size = 64
    benchmark_results = {}

    for m_type in models_to_run:
        train_ds = UniversalSleepDataset(apnea_train_files, model_type=m_type)
        test_ds  = UniversalSleepDataset(apnea_test_files,  model_type=m_type)
        ucddb_ds = UniversalSleepDataset(ucddb_test_files,  model_type=m_type)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        ucddb_loader = DataLoader(ucddb_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        model, criterion = build_model_and_criterion(m_type, class_weights, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_macro_f1 = -1.0
        best_weights = None
        best_test_metrics = None

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs} [{m_type}]", leave=False)
            for batch in pbar:
                batch = [b.to(device) for b in batch]
                optimizer.zero_grad()

                if m_type == 'pimtl':
                    ecg, label, ahi, cpc = batch
                    apnea_logits, ahi_pred, cpc_pred = model(ecg)
                    loss = criterion(apnea_logits, ahi_pred, cpc_pred, label, ahi, cpc)
                elif m_type == 'pase_mst':
                    logits = model(*batch[:5])
                    loss = criterion(logits, batch[5])
                else:
                    inputs, label = batch[0], batch[1]
                    logits = model(inputs)
                    loss = criterion(logits, label)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

            scheduler.step()
            test_metrics = evaluate_cohort(model, test_loader, m_type, device)
            
            if test_metrics['Macro_F1'] > best_macro_f1:
                best_macro_f1 = test_metrics['Macro_F1']
                best_weights = copy.deepcopy(model.state_dict())
                best_test_metrics = test_metrics

        if best_weights is not None:
            model.load_state_dict(best_weights)
            torch.save(best_weights, f"/kaggle/working/best_{m_type}.pth")
            
        ucddb_metrics = evaluate_cohort(model, ucddb_loader, m_type, device)
        benchmark_results[m_type] = {"apnea_test": best_test_metrics, "ucddb_test": ucddb_metrics}

        del model, optimizer, scheduler, train_loader, test_loader, ucddb_loader, train_ds, test_ds, ucddb_ds
        torch.cuda.empty_cache()