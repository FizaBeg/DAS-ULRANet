"""Synthetic DAS benchmark generator + preprocessing (detrend -> channel z-score -> DWT denoise)."""
import numpy as np, pandas as pd, pywt

BASE = dict(fs=200, W=512, S=64, dx=1.0, noise_std=1.0, cm_std=0.5, drift_amp=1.5,
            traffic_p=0.6, traffic_amp=2.0, spike_p=0.15, bad_ch_p=0.05)
SITES = {"A": dict(BASE),
         "B": dict(BASE, noise_std=1.6, cm_std=0.9, drift_amp=2.5, traffic_p=0.9, traffic_amp=3.0),
         "C": dict(BASE, noise_std=0.8, cm_std=0.3, drift_amp=0.8, traffic_p=0.2, spike_p=0.3, bad_ch_p=0.15)}

def bandlimited(rng, n, fs, f_lo, f_hi, size=None):
    shape = (n,) if size is None else (n, size)
    F = np.fft.rfft(rng.standard_normal(shape), axis=0)
    f = np.fft.rfftfreq(n, 1 / fs)
    m = ((f >= f_lo) & (f <= f_hi)).astype(float)
    F = F * (m if size is None else m[:, None])
    y = np.fft.irfft(F, n=n, axis=0)
    return y / (y.std() + 1e-12)

def ricker(t, f0):
    a = (np.pi * f0 * t) ** 2
    return (1 - 2 * a) * np.exp(-a)

def smooth_profile(rng, S, width):
    k = np.exp(-0.5 * (np.arange(-3 * width, 3 * width + 1) / width) ** 2)
    g = np.convolve(rng.standard_normal(S + 6 * width), k, mode="valid")[:S]
    return (g - g.mean()) / (g.std() + 1e-12)

def gen_window(rng, cfg, leak=False, snr_db=None):
    W, S, fs = cfg["W"], cfg["S"], cfg["fs"]
    t = np.arange(W) / fs; ch = np.arange(S)
    X = cfg["noise_std"] * rng.standard_normal((W, S))
    X *= np.clip(1 + 0.2 * rng.standard_normal(S), 0.5, None)[None, :]
    X += cfg["cm_std"] * bandlimited(rng, W, fs, 0.5, 20)[:, None]
    X += (cfg["drift_amp"] * np.sin(2 * np.pi * rng.uniform(0.02, 0.1) * t + rng.uniform(0, 2 * np.pi))[:, None]
          * (1 + 0.5 * smooth_profile(rng, S, 20))[None, :])
    if rng.random() < cfg["traffic_p"]:
        for _ in range(rng.integers(1, 4)):
            v, d = rng.uniform(10, 60), rng.choice([-1, 1])
            t0, c0, f0 = rng.uniform(0, W / fs), rng.uniform(0, S), rng.uniform(4, 15)
            tc = t0 + d * (ch - c0) * cfg["dx"] / v
            X += (cfg["traffic_amp"] * rng.uniform(0.5, 1.5) * ricker(t[:, None] - tc[None, :], f0)
                  * np.exp(-(((ch - c0) / (S / 2)) ** 2))[None, :])
    if rng.random() < cfg["spike_p"]:
        for _ in range(rng.integers(1, 4)):
            X[rng.integers(0, W), rng.integers(0, S)] += rng.choice([-1, 1]) * rng.uniform(5, 10) * cfg["noise_std"]
    X[:, rng.random(S) < cfg["bad_ch_p"]] *= 3
    info = dict(kind="none", snr_db=np.nan)
    if leak:
        kind = rng.choice(["lowfreq", "burst", "both"], p=[0.4, 0.3, 0.3])
        c0, sig, onset = rng.uniform(0.15 * S, 0.85 * S), rng.uniform(3, 12), rng.uniform(0.1, 0.6) * W / fs
        g = np.exp(-0.5 * ((ch - c0) / sig) ** 2)
        tt = np.clip(t - onset, 0, None); on = (t >= onset).astype(float)
        L = np.zeros((W, S))
        if kind in ("lowfreq", "both"):
            env = on * (1 - np.exp(-tt / 0.2)) * np.exp(-tt / (3 * rng.uniform(0.5, 2.0)))
            L += (env * np.sin(2 * np.pi * rng.uniform(0.7, 5.0) * tt + rng.uniform(0, 2 * np.pi)))[:, None] * g[None, :]
        if kind in ("burst", "both"):
            L += 0.7 * bandlimited(rng, W, fs, 20, 80, size=S) * (on * np.exp(-tt / rng.uniform(0.3, 1.0)))[:, None] * g[None, :]
        act = np.abs(L) > 0.1 * np.abs(L).max()
        snr = rng.uniform(-5, 15) if snr_db is None else snr_db
        X += L * (cfg["noise_std"] * 10 ** (snr / 20) / np.sqrt((L[act] ** 2).mean()))
        info = dict(kind=kind, snr_db=snr)
    return X.astype(np.float32), info

def make_dataset(site, n_norm, n_leak, seed):
    rng, cfg = np.random.default_rng(seed), SITES[site]
    flags = np.r_[np.zeros(n_norm, int), np.ones(n_leak, int)]; rng.shuffle(flags)
    Xs, meta = [], []
    for y in flags:
        x, info = gen_window(rng, cfg, bool(y)); Xs.append(x); meta.append(dict(label=int(y), site=site, **info))
    return np.stack(Xs), pd.DataFrame(meta)

# ---------------- preprocessing ----------------
def detrend_window(X):
    t = np.arange(X.shape[1], dtype=np.float32); t -= t.mean()
    Xm = X - X.mean(axis=1, keepdims=True)
    slope = (Xm * t[None, :, None]).sum(axis=1, keepdims=True) / (t ** 2).sum()
    return Xm - slope * t[None, :, None]

DWT_K = 0.15   # chosen on DEV sets (dwt_sweep.py): thr = k * sigma * sqrt(2 ln W)

def dwt_denoise(X, wavelet="db4", level=4, mode="soft", k=DWT_K):
    """channel-wise soft-threshold DWT denoising, universal threshold, sigma from MAD of finest details.
    X: (N, W, S) -> same shape"""
    Xt = np.moveaxis(X, 1, -1)                              # (N,S,W)
    coeffs = pywt.wavedec(Xt, wavelet, level=level, axis=-1)
    sigma = np.median(np.abs(coeffs[-1]), axis=-1, keepdims=True) / 0.6745
    thr = k * sigma * np.sqrt(2 * np.log(Xt.shape[-1]))
    den = [coeffs[0]] + [pywt.threshold(c, thr, mode=mode) for c in coeffs[1:]]
    out = pywt.waverec(den, wavelet, axis=-1)[..., :Xt.shape[-1]]
    return np.moveaxis(out, -1, 1).astype(np.float32)

class Prep:
    """fit channel statistics on source training data; apply to any site (zero-shot uses source stats)."""
    def __init__(self, dwt=True, k=DWT_K): self.dwt, self.k = dwt, k
    def fit(self, Xtr):
        D = detrend_window(Xtr)
        self.mu = D.mean(axis=(0, 1), keepdims=True); self.sd = D.std(axis=(0, 1), keepdims=True) + 1e-8
        return self
    def __call__(self, X):
        Z = (detrend_window(X) - self.mu) / self.sd
        return dwt_denoise(Z, k=self.k) if self.dwt else Z.astype(np.float32)

# ---------------- pseudo-anomalies for unsupervised training ----------------
def inject_pseudo(Z, rng, fs=200, kinds=("bump", "step", "noise")):
    """Z: (B,W,S) preprocessed normal windows. Adds generic transient anomalies (families deliberately different
    from the test leak generator: gaussian-windowed bump, smooth step, band-limited noise burst)."""
    B, W, S = Z.shape; t = np.arange(W) / fs; ch = np.arange(S); out = Z.copy()
    for i in range(B):
        kind = rng.choice(list(kinds)); c0 = rng.uniform(0.1 * S, 0.9 * S)
        if kind == "bump":
            sc = rng.uniform(2, 15); g = np.exp(-0.5 * ((ch - c0) / sc) ** 2)
            t0, tau, f = rng.uniform(0.3, 2.2), rng.uniform(0.2, 0.8), rng.uniform(0.5, 8)
            A = (np.exp(-0.5 * ((t - t0) / tau) ** 2) * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)))[:, None] * g[None, :]
        elif kind == "step":
            w = rng.uniform(5, 30); g = 1 / (1 + np.exp(-(ch - c0) / 3)) * 1 / (1 + np.exp((ch - c0 - w) / 3))
            t0, wd = rng.uniform(0.3, 2.0), rng.uniform(0.3, 1.0)
            A = (1 / (1 + np.exp(-(t - t0) / (wd / 4))))[:, None] * g[None, :]
        else:
            w = rng.uniform(5, 30); g = 1 / (1 + np.exp(-(ch - c0) / 3)) * 1 / (1 + np.exp((ch - c0 - w) / 3))
            t0, tau = rng.uniform(0.3, 2.0), rng.uniform(0.2, 0.8)
            A = bandlimited(rng, W, fs, 15, 90, size=S) * np.exp(-0.5 * ((t - t0) / tau) ** 2)[:, None] * g[None, :]
        act = np.abs(A) > 0.1 * np.abs(A).max()
        out[i] += (A * (10 ** (rng.uniform(-5, 15) / 20) / np.sqrt((A[act] ** 2).mean()))).astype(np.float32)
    return out
