import numpy as np
import pywt
import torch
import torch.nn.functional as F
from scipy import signal

def compute_cpc(ecg, fs=100, min_rpeaks=10):
    import neurokit2 as nk
    try:
        _, rpeaks = nk.ecg_peaks(ecg, sampling_rate=fs)
        r_locs = rpeaks["ECG_R_Peaks"]
        if len(r_locs) < min_rpeaks:
            return 0.0, 0.0, 0.0
    except Exception:
        return 0.0, 0.0, 0.0

    rri = np.diff(r_locs) / fs
    rri_time = r_locs[1:] / fs
    edr = [np.ptp(ecg[max(0, loc - 10):min(len(ecg), loc + 10)]) for loc in r_locs]
    edr = np.array(edr)
    edr_time = r_locs / fs

    fs_interp = 4.0
    t_start = max(rri_time[0], edr_time[0])
    t_end = min(rri_time[-1], edr_time[-1])
    if t_end <= t_start:
        return 0.0, 0.0, 0.0

    t_uniform = np.arange(t_start, t_end, 1.0 / fs_interp)
    if len(t_uniform) < 16:
        return 0.0, 0.0, 0.0

    rri_interp = signal.detrend(np.interp(t_uniform, rri_time, rri))
    edr_interp = signal.detrend(np.interp(t_uniform, edr_time, edr))

    nperseg = min(len(rri_interp), 256)
    f, coh = signal.coherence(rri_interp, edr_interp, fs=fs_interp, nperseg=nperseg)
    _, pxy = signal.csd(rri_interp, edr_interp, fs=fs_interp, nperseg=nperseg)

    cpc_spectrum = coh * np.abs(pxy)
    lf_mask = (f >= 0.01) & (f < 0.1)
    hf_mask = (f >= 0.1) & (f < 0.4)

    cpc_lf = float(np.mean(cpc_spectrum[lf_mask])) if np.count_nonzero(lf_mask) > 0 else 0.0
    cpc_hf = float(np.mean(cpc_spectrum[hf_mask])) if np.count_nonzero(hf_mask) > 0 else 0.0
    e_lfc = cpc_lf / (cpc_hf + 1e-8)
    return cpc_lf, cpc_hf, e_lfc

def compute_rwave_slope_edr(ecg, fs=100, target_len=6000):
    import neurokit2 as nk
    try:
        _, rpeaks = nk.ecg_peaks(ecg, sampling_rate=fs)
        r_locs = rpeaks["ECG_R_Peaks"]
        if len(r_locs) < 2:
            return np.zeros(target_len, dtype=np.float32)
    except Exception:
        return np.zeros(target_len, dtype=np.float32)

    slopes = []
    for loc in r_locs:
        seg = ecg[max(0, loc - 15):min(len(ecg), loc + 5)]
        slopes.append(np.max(np.diff(seg)) if len(seg) > 1 else 0.0)

    t_beats = r_locs / fs
    t_target = np.linspace(0, (target_len - 1) / fs, target_len)
    return np.interp(t_target, t_beats, slopes, left=slopes[0], right=slopes[-1]).astype(np.float32)

def ecg_to_scalogram(ecg_segment, scales=None, target_size=(224, 224)):
    if scales is None:
        scales = np.arange(1, 65)
        
    signal_1d = np.asarray(ecg_segment, dtype=np.float32).flatten()
    cwtmatr, _ = pywt.cwt(signal_1d, scales, 'morl')
    mag = np.log1p(np.abs(cwtmatr))
    
    tensor_img = torch.tensor(mag, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor_img, size=target_size, mode='bilinear', align_corners=False)
    return resized.squeeze(0).numpy()