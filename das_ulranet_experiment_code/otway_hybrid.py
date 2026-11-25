"""Hybrid Otway benchmark: REAL DAS background (Otway, 5 wells x 16 SEG-Y files) + SYNTHETIC leak injection,
plus a separate test on REAL microseismic events.  Same leak physics, window size and metrics as the synthetic benchmark.

  python otway_hybrid.py --data data --inspect                      # list files / wells / shapes only
  python otway_hybrid.py --data data --folds 0 --seeds 0 --jobs main        # smoke test
  python otway_hybrid.py --data data --folds 0,1,2,3,4 --seeds 0,1,2        # full run (default jobs below)

Protocol (leave-wells-out; 5 folds, every fold holds out 2 wells):
  train / val : real background of the 3 remaining wells (no leaks, no labels)
  test_A      : held-out well #1  - real background windows (normal) vs. same-well windows with an injected synthetic leak
  test_C      : held-out well #2  - same construction (second, unseen well)
  test_B      : held-out well #1  - real background (normal) vs. REAL microseismic-event windows (first 2.5 s of each file)
  dev_A/dev_B : labelled dev sets built from the training wells only (used only to choose the score-fusion weight)
Pre-processing: detrend -> per-channel robust z-score (median/MAD of the window) -> clip +-10 -> [leak injection] -> DWT (gen.DWT_K).
"""
import os, re, sys, glob, json, time, argparse, numpy as np, pandas as pd
from scipy.signal import decimate

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data"); ap.add_argument("--inspect", action="store_true")
ap.add_argument("--folds", default="0,1,2,3,4"); ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--jobs", default="main,nodwt,cnnae,lstmae,pca,supcnn_pa")
ap.add_argument("--threads", type=int, default=4); ap.add_argument("--max-files", type=int, default=0)
ap.add_argument("--cache", default="otway_windows.npz")
a = ap.parse_args()

FS_OUT, W, S, EV_END, BG_START = 200, 512, 64, 0, 2.0      # background starts at t = 2.0 s (after the event)

# ------------------------------------------------------------------ reading
def find_files(root):
    fs = [f for f in glob.glob(os.path.join(root, "**", "*"), recursive=True)
          if f.lower().endswith((".sgy", ".segy")) and "__MACOSX" not in f and "/._" not in f.replace("\\", "/")]
    return sorted(fs)

def read_segy(path):
    """-> array (channels, time) float32 and sampling rate in Hz."""
    try:
        import segyio
        with segyio.open(path, ignore_geometry=True) as f:
            arr = np.asarray(f.trace.raw[:], dtype=np.float32); dt = f.bin[segyio.BinField.Interval] * 1e-6
        fs = 1.0 / dt if dt > 0 else 1000.0
        if not (200 <= fs <= 20000): fs = 1000.0              # implausible header value -> Otway DAS is 1 kHz
    except ImportError:
        try:
            import obspy
            st = obspy.read(path, format="SEGY", unpack_trace_headers=False)
            arr = np.stack([tr.data for tr in st]).astype(np.float32); fs = float(st[0].stats.sampling_rate)
        except ImportError:
            sys.exit("Need segyio or obspy:  pip install segyio")
    if arr.shape[1] < arr.shape[0]: arr = arr.T          # time must be the longer axis; rows = channels
    return arr, fs

def well_of(path, i):
    """file names are event_<event>_<well>_strain_geom.sgy : the SECOND number is the well (same size per well = same channel count)."""
    m = re.search(r"event[_-]?(\d+)[_-](\d+)", os.path.basename(path), re.I)
    return int(m.group(2)) if m else None

# ------------------------------------------------------------------ windows from one file
def windows_from(arr, fs):
    """arr (C,T) -> bg (n,W,S) from t>=2 s, ev (n,W,S) from t<2.56 s; both at 200 Hz, non-overlapping 64-channel groups."""
    q = int(round(fs / FS_OUT)); x = decimate(arr, q, axis=1, zero_phase=True).astype(np.float32) if q > 1 else arr
    i0 = int(BG_START * FS_OUT); ng = x.shape[0] // S
    if x.shape[1] < i0 + W: return None, None
    bg = np.stack([x[g * S:(g + 1) * S, i0:i0 + W].T for g in range(ng)])
    ev = np.stack([x[g * S:(g + 1) * S, 0:W].T for g in range(ng)])
    return bg, ev

def load_all(root, max_files=0):
    files = find_files(root)
    if max_files: files = files[:max_files]
    if not files: sys.exit(f"No SEG-Y files found under '{root}'. Pass --data <folder>.")
    wells = [well_of(f, i) for i, f in enumerate(files)]
    if any(w is None for w in wells):                      # fallback: consecutive blocks of 16 files
        print("WARNING: could not parse well from file names; using consecutive blocks of 16 files.")
        wells = [i // 16 for i in range(len(files))]
    bgs, evs, wid = [], [], []
    for f, w in zip(files, wells):
        try:
            if os.path.getsize(f) < 100_000: print("skip (header only / empty):", os.path.basename(f)); continue
            arr, fs = read_segy(f); b, e = windows_from(arr, fs)
        except (IndexError, ValueError, RuntimeError) as err:
            print("skip (unreadable):", os.path.basename(f), "-", err); continue
        if b is None: print("skip (too short):", f); continue
        bgs.append(b); evs.append(e); wid += [w] * len(b)
    return np.concatenate(bgs), np.concatenate(evs), np.array(wid), files, wells

# ------------------------------------------------------------------ preprocessing + leak injection
import gen
def robust_norm(X):
    Xd = gen.detrend_window(X.astype(np.float32))
    med = np.median(Xd, axis=1, keepdims=True); sig = 1.4826 * np.median(np.abs(Xd - med), axis=1, keepdims=True)
    sig = np.maximum(sig, 0.1 * np.median(sig, axis=2, keepdims=True) + 1e-12)
    return np.clip((Xd - med) / sig, -10, 10).astype(np.float32)

def make_leak(rng, snr_db, fs=FS_OUT):
    """same leak physics as gen.gen_window (low-freq oscillation and/or 20-80 Hz burst, Gaussian footprint), noise std = 1."""
    t = np.arange(W) / fs; ch = np.arange(S); kind = rng.choice(["lowfreq", "burst", "both"], p=[0.4, 0.3, 0.3])
    c0, sig, onset = rng.uniform(0.15 * S, 0.85 * S), rng.uniform(3, 12), rng.uniform(0.1, 0.6) * W / fs
    g = np.exp(-0.5 * ((ch - c0) / sig) ** 2); tt = np.clip(t - onset, 0, None); on = (t >= onset).astype(float); L = np.zeros((W, S))
    if kind in ("lowfreq", "both"):
        env = on * (1 - np.exp(-tt / 0.2)) * np.exp(-tt / (3 * rng.uniform(0.5, 2.0)))
        L += (env * np.sin(2 * np.pi * rng.uniform(0.7, 5.0) * tt + rng.uniform(0, 2 * np.pi)))[:, None] * g[None, :]
    if kind in ("burst", "both"):
        L += 0.7 * gen.bandlimited(rng, W, fs, 20, 80, size=S) * (on * np.exp(-tt / rng.uniform(0.3, 1.0)))[:, None] * g[None, :]
    act = np.abs(L) > 0.1 * np.abs(L).max()
    return (L * (10 ** (snr_db / 20) / np.sqrt((L[act] ** 2).mean()))).astype(np.float32), kind

def with_leaks(Zn, rng):
    out = Zn.copy(); snrs, kinds = [], []
    for i in range(len(out)):
        snr = rng.uniform(-5, 15); L, k = make_leak(rng, snr); out[i] += L; snrs.append(snr); kinds.append(k)
    return out, snrs, kinds

def labelled(Znorm, Zanom, rng, real_events=False):
    """normal windows + anomalous windows -> (X, meta). Zanom is already anomalous (real events) or gets leaks injected."""
    if real_events: A, snr, kind = Zanom, [np.nan] * len(Zanom), ["event"] * len(Zanom)
    else:           A, snr, kind = with_leaks(Zanom, rng)
    X = np.concatenate([Znorm, A]); y = np.r_[np.zeros(len(Znorm), int), np.ones(len(A), int)]
    meta = pd.DataFrame(dict(label=y, snr_db=np.r_[[np.nan] * len(Znorm), snr], kind=np.r_[["none"] * len(Znorm), kind]))
    p = rng.permutation(len(X)); return X[p], meta.iloc[p].reset_index(drop=True)

def split_well(Zbg, rng, n_cal=100):
    """background windows of one well -> calib (normal), test-normal, to-inject (disjoint)."""
    p = rng.permutation(len(Zbg)); n_cal = min(n_cal, len(Zbg) // 4); rest = p[n_cal:]; h = len(rest) // 2
    return Zbg[p[:n_cal]], Zbg[rest[:h]], Zbg[rest[h:]]

def build_fold(Zbg, Zev, wid, wells_sorted, k, seed=123):
    wa, wc = wells_sorted[k % len(wells_sorted)], wells_sorted[(k + 1) % len(wells_sorted)]
    rng = np.random.default_rng(seed + k); tr_mask = ~np.isin(wid, [wa, wc])
    Ztr_all = Zbg[tr_mask]; p = rng.permutation(len(Ztr_all)); nv = int(0.25 * len(p))
    Zval, Ztr = Ztr_all[p[:nv]], Ztr_all[p[nv:]]
    h = len(Zval) // 2
    D = {"train": (Ztr, None), "val": (Zval, None)}
    D["dev_A"] = labelled(Zval[:h], Zval[h:], rng)                                   # injected leaks, training wells only
    ev_tr = Zev[tr_mask]; D["dev_B"] = labelled(Zval[:h], ev_tr[rng.permutation(len(ev_tr))[:h]], rng, real_events=True)
    cal_a, nrm_a, inj_a = split_well(Zbg[wid == wa], rng); cal_c, nrm_c, inj_c = split_well(Zbg[wid == wc], rng)
    D["calib_A"], D["calib_B"], D["calib_C"] = (cal_a, None), (cal_a, None), (cal_c, None)
    D["test_A"] = labelled(nrm_a, inj_a, rng)
    ev_a = Zev[wid == wa]; D["test_B"] = labelled(nrm_a, ev_a[rng.permutation(len(ev_a))[:len(nrm_a)]], rng, real_events=True)
    D["test_C"] = labelled(nrm_c, inj_c, rng)
    D["sup"] = D["train"]
    return D, (wa, wc)

def to_prep(D, dwt):
    return {k: ((gen.dwt_denoise(v[0]) if dwt else v[0]), v[1]) for k, v in D.items()}

# ------------------------------------------------------------------ main
if __name__ == "__main__":
    if os.path.exists(a.cache) and not a.inspect:
        z = np.load(a.cache); Zbg, Zev, wid = z["bg"], z["ev"], z["wid"]; print("loaded cache", Zbg.shape)
    else:
        t0 = time.time(); bg, ev, wid, files, wells = load_all(a.data, a.max_files)
        print(f"{len(files)} files, wells {sorted(set(wells))}, windows per well: "
              f"{ {int(w): int((wid == w).sum()) for w in sorted(set(wid))} }, raw window array {bg.shape}, {time.time() - t0:.0f}s")
        if a.inspect:
            arr, fs = read_segy(files[0]); print("first file:", files[0], "-> (channels, time) =", arr.shape, "fs =", fs)
            sys.exit(0)
        Zbg, Zev = robust_norm(bg), robust_norm(ev); np.savez(a.cache, bg=Zbg, ev=Zev, wid=wid)
    print("normalised: bg std %.2f, ev std %.2f (should be ~1)" % (Zbg.std(), Zev.std()))

    import torch; torch.set_num_threads(a.threads)
    import run; run.E = int(os.environ.get("EPOCHS", 12)); print("epochs:", run.E, flush=True)
    wells_sorted = sorted(set(wid.tolist())); os.makedirs("results_otway", exist_ok=True)
    for k in [int(x) for x in a.folds.split(",")]:
        D0, (wa, wc) = build_fold(Zbg, Zev, wid, wells_sorted, k)
        D = {True: to_prep(D0, True), False: to_prep(D0, False)}
        print(f"fold {k}: test wells A/B={wa}, C={wc}; train {len(D0['train'][0])}, val {len(D0['val'][0])}, "
              f"test_A {len(D0['test_A'][0])}, test_B {len(D0['test_B'][0])}, test_C {len(D0['test_C'][0])}", flush=True)
        for seed in [int(x) for x in a.seeds.split(",")]:
            for name in a.jobs.split(","):
                f = f"results_otway/{name}_f{k}_s{seed}.json"
                if os.path.exists(f): continue
                r = run.job(name, seed, D); r["fold"] = k; json.dump(r, open(f, "w"))
                print(f"  fold {k} {name:9s} seed {seed}  AUROC  A(inj) {r['results']['A']['auroc']:.3f}  "
                      f"B(real events) {r['results']['B']['auroc']:.3f}  C(inj, 2nd well) {r['results']['C']['auroc']:.3f}  ({r['secs']:.0f}s)", flush=True)

    # ---- summary table
    R = {}
    for f in glob.glob("results_otway/*.json"):
        r = json.load(open(f)); R.setdefault(r["name"], []).append(r)
    L = ["| Method | n runs | AUROC A (injected leak) | AUROC C (2nd held-out well) | AUROC B (real microseismic events) | F1 A (recal. thr) | FAR A (recal. thr) |", "|---|---|---|---|---|---|---|"]
    ms = lambda v: f"{np.mean(v):.3f} ± {np.std(v, ddof=1) if len(v) > 1 else 0:.3f}"
    for n in ["pca", "cnnae", "lstmae", "supcnn_pa", "nodwt", "main"]:
        if n not in R: continue
        g = lambda s, *kk: [__import__("functools").reduce(lambda d, key: d[key], kk, r["results"][s]) for r in R[n]]
        L.append(f"| {n} | {len(R[n])} | {ms(g('A','auroc'))} | {ms(g('C','auroc'))} | {ms(g('B','auroc'))} | {ms(g('A','recal','f1'))} | {ms(g('A','recal','far'))} |")
    open("results_otway.md", "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L))