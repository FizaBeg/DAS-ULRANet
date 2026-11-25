"""
Config selection is done on a labelled DEV set (new seeds, never used for the final numbers);
the final numbers are read on the untouched TEST sets A/B/C.

Run in Jupyter (inside the experiment folder):
    %env EPOCHS=12
    %run dwt_sweep.py --seeds 0,1,2
Writes dwt_sweep.csv and prints a table sorted by DEV score.
"""
import os, argparse, time, numpy as np, pandas as pd, pywt

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
SEEDS = [int(s) for s in a.seeds.split(",")]

import torch
torch.set_num_threads(a.threads)
import gen, run            # run.py only trains/evaluates when executed as a script, so importing is safe
run.E = int(os.environ.get("EPOCHS", 12))     # to not get module cache problem
print("epochs:", run.E, flush=True)

def dwt_cfg(X, wavelet, level, mode, k):
    """channel-wise DWT denoising. X: (N, W, S). Threshold = k * sigma * sqrt(2 ln W), sigma from MAD of finest details."""
    Xt = np.moveaxis(X, 1, -1)
    c = pywt.wavedec(Xt, wavelet, level=level, axis=-1)
    sigma = np.median(np.abs(c[-1]), axis=-1, keepdims=True) / 0.6745
    thr = k * sigma * np.sqrt(2 * np.log(Xt.shape[-1]))
    den = [c[0]] + [pywt.threshold(x, thr, mode=mode) for x in c[1:]]
    out = pywt.waverec(den, wavelet, axis=-1)[..., :Xt.shape[-1]]
    return np.moveaxis(out, -1, 1).astype(np.float32)


CONFIGS = {                                         # name: (wavelet, level, mode, k);  None = no DWT
    "none":              None,
    "db4_L4_soft_k1.0":  ("db4", 4, "soft", 1.0),    # current setting
    "db4_L4_soft_k0.5":  ("db4", 4, "soft", 0.5),
    "db4_L4_soft_k0.25": ("db4", 4, "soft", 0.25),
    "db4_L4_hard_k1.0":  ("db4", 4, "hard", 1.0),
    "db4_L4_hard_k0.5":  ("db4", 4, "hard", 0.5),
    "db4_L2_soft_k1.0":  ("db4", 2, "soft", 1.0),
    "db4_L3_soft_k1.0":  ("db4", 3, "soft", 1.0),
    "sym4_L4_soft_k0.5": ("sym4", 4, "soft", 0.5),
    "haar_L4_soft_k0.5": ("haar", 4, "soft", 0.5),
    "db4_L4_soft_k0.10": ("db4", 4, "soft", 0.10),
    "db4_L4_soft_k0.15": ("db4", 4, "soft", 0.15),
    "db4_L4_hard_k0.25": ("db4", 4, "hard", 0.25),
    "sym4_L4_soft_k0.25": ("sym4", 4, "soft", 0.25),
}

# ---------- raw data: same seeds as build_data.py, plus two new labelled DEV sets ----------
t0 = time.time()
Xtr, _ = gen.make_dataset("A", 500, 0, seed=1)
Xval, _ = gen.make_dataset("A", 150, 0, seed=2)
RAW = {"val": (Xval, None)}
for i, s in enumerate("ABC"):
    RAW[f"test_{s}"] = gen.make_dataset(s, 250, 250, seed=10 + i)
RAW["dev_A"] = gen.make_dataset("A", 250, 250, seed=40)
RAW["dev_B"] = gen.make_dataset("B", 250, 250, seed=41)
P = gen.Prep(dwt=False).fit(Xtr)                    # channel statistics from source training data (shared by all configs)
print("raw data ready in %.0f s" % (time.time() - t0), flush=True)


def prep(X, cfg):
    Z = P(X)
    return Z if cfg is None else dwt_cfg(Z, *cfg)


VC = dict(attn="mdlra", r=16, lam=(1, .2, .2))      # the full model
rows = []
for name, cfg in CONFIGS.items():
    Ztr = prep(Xtr, cfg)
    Zs = {k: (prep(v[0], cfg), v[1]) for k, v in RAW.items()}
    for sd in SEEDS:
        t1 = time.time()
        m, _ = run.train_ulra(Ztr, VC, sd)
        rv = run.run_model(m, Zs["val"][0])[0]
        mu, sg = rv.mean(), rv.std() + 1e-9

        def score(Z):
            r, z, ev = run.run_model(m, Z)
            return 0.5 * run.sig((r - mu) / sg) + 0.5 * (ev[:, 1] + 1) / (ev.sum(1) + 2)

        th = np.quantile(score(Zs["val"][0]), .99)  # label-free threshold from validation-normal windows
        row = dict(config=name, seed=sd)
        for k in ("dev_A", "dev_B", "test_A", "test_B", "test_C"):
            Z, meta = Zs[k]
            y = meta.label.values
            s = score(Z)
            row[k] = run.auroc(y, s)
            if k == "test_A":
                pr = run.prf(y, s, th)
                row["F1_A"], row["FAR_A"] = pr["f1"], pr["far"]
        rows.append(row)
        print("%-18s seed %d | dev_A %.3f dev_B %.3f | test_A %.3f (%.0f s)" %
              (name, sd, row["dev_A"], row["dev_B"], row["test_A"], time.time() - t1), flush=True)

df = pd.DataFrame(rows)
df.to_csv("dwt_sweep.csv", index=False)
g = df.groupby("config").mean(numeric_only=True).drop(columns="seed")
g["dev_mean"] = (g.dev_A + g.dev_B) / 2
g = g.sort_values("dev_mean", ascending=False)
print("\nSorted by DEV (mean of dev_A and dev_B). Choose the config from the dev_mean column, then read the TEST columns.\n")
print(g.round(3).to_string())
