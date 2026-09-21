import os
import glob
import re
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, confusion_matrix
from scipy.stats import pearsonr

from data.dataset import UniversalSleepDataset
from models.architectures import build_model_and_criterion

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 14,
    'legend.fontsize': 11, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'figure.dpi': 300, 'savefig.dpi': 300, 'font.family': 'serif'
})

plot_dir = "/kaggle/working/paper_plots"
os.makedirs(plot_dir, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

plot_models = ['pimtl', 'pase_mst', 'cnn_transformer_lstm']
model_names = {'pimtl': 'PIMTL-CPC', 'pase_mst': 'PASE-MST', 'cnn_transformer_lstm': 'CNN-Trans-LSTM'}
colors = {'pimtl': '#d62728', 'pase_mst': '#1f77b4', 'cnn_transformer_lstm': '#2ca02c'}

precomputed_dir = "/kaggle/working/precomputed"
all_files = sorted(glob.glob(os.path.join(precomputed_dir, "*.pkl")))
apnea_test_files  = [p for p in all_files if re.search(r"apnea_ecg_x\d{2}\.pkl$", p)]
ucddb_test_files  = [p for p in all_files if "ucddb_" in p]

def gather_predictions(m_type):
    test_ds  = UniversalSleepDataset(apnea_test_files,  model_type=m_type)
    ucddb_ds = UniversalSleepDataset(ucddb_test_files,  model_type=m_type)
    
    t_loader = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=2)
    u_loader = DataLoader(ucddb_ds, batch_size=64, shuffle=False, num_workers=2)

    model, _ = build_model_and_criterion(m_type, device=device)
    model.load_state_dict(torch.load(f"/kaggle/working/best_{m_type}.pth", map_location=device, weights_only=True))
    model.eval()
    
    def run_inference(loader):
        y_true, y_probs, y_preds, ahi_true, ahi_preds = [], [], [], [], []
        with torch.no_grad():
            for batch in loader:
                batch = [b.to(device) for b in batch]
                if m_type == 'pimtl':
                    ecg, target, ahi, _ = batch
                    logits, ahi_p, _ = model(ecg)
                    ahi_true.extend(ahi.cpu().numpy())
                    ahi_preds.extend(ahi_p.view(-1).cpu().numpy())
                elif m_type == 'pase_mst':
                    logits = model(*batch[:5])
                    target = batch[5]
                elif m_type == 'cnn_transformer_lstm':
                    logits = model(batch[0])
                    target = batch[1]
                    
                probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = logits.argmax(dim=1).cpu().numpy()
                y_probs.extend(probs); y_preds.extend(preds); y_true.extend(target.cpu().numpy())
        return np.array(y_true), np.array(y_probs), np.array(y_preds), np.array(ahi_true), np.array(ahi_preds)

    y_t_ap, y_p_ap, y_pr_ap, ahi_t_ap, ahi_p_ap = run_inference(t_loader)
    y_t_uc, y_p_uc, y_pr_uc, ahi_t_uc, ahi_p_uc = run_inference(u_loader)
    
    return {
        'apnea': {'true': y_t_ap, 'prob': y_p_ap, 'pred': y_pr_ap, 'ahi_t': ahi_t_ap, 'ahi_p': ahi_p_ap},
        'ucddb': {'true': y_t_uc, 'prob': y_p_uc, 'pred': y_pr_uc, 'ahi_t': ahi_t_uc, 'ahi_p': ahi_p_uc}
    }

if __name__ == "__main__":
    results = {'apnea': {}, 'ucddb': {}}
    for m in plot_models:
        results['apnea'][m] = gather_predictions(m)['apnea']
        results['ucddb'][m] = gather_predictions(m)['ucddb']

    # Fig 1: ROC Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for idx, (cohort_key, title) in enumerate([('apnea', 'Apnea-ECG Test Set'), ('ucddb', 'UCDDB Zero-Shot Transfer')]):
        ax = axes[idx]
        for m in plot_models:
            fpr, tpr, _ = roc_curve(results[cohort_key][m]['true'], results[cohort_key][m]['prob'])
            ax.plot(fpr, tpr, color=colors[m], lw=2.5 if m == 'pimtl' else 1.5, label=f"{model_names[m]} (AUC = {auc(fpr, tpr):.3f})")
        ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        ax.set(xlim=[0.0, 1.0], ylim=[0.0, 1.05], xlabel='False Positive Rate', ylabel='True Positive Rate', title=title)
        ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/Fig1_ROC_Curves.pdf", bbox_inches='tight')

    # Fig 2: Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for idx, (cohort_key, title) in enumerate([('apnea', 'PIMTL-CPC: Apnea-ECG'), ('ucddb', 'PIMTL-CPC: UCDDB')]):
        y_true, y_pred = results[cohort_key]['pimtl']['true'], results[cohort_key]['pimtl']['pred']
        cm_norm = confusion_matrix(y_true, y_pred).astype('float') / confusion_matrix(y_true, y_pred).sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues', ax=axes[idx], xticklabels=['Normal', 'Apnea'], yticklabels=['Normal', 'Apnea'])
        axes[idx].set(title=title, ylabel='True State', xlabel='Prediction')
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/Fig2_Confusion_Matrices.pdf", bbox_inches='tight')