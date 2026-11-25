import json, glob, numpy as np, collections, math, sys
R = collections.defaultdict(list)
for f in sorted(glob.glob("results/*.json")):
    r = json.load(open(f)); R[r["name"]].append(r)
LAT = json.load(open("latency.json")) if len(glob.glob("latency.json")) else {}
def ms(vals, p=3):
    v = np.array([x for x in vals if x is not None], float)
    if len(v) == 0: return "n/a"
    return f"{v.mean():.{p}f} ± {v.std(ddof=1) if len(v) > 1 else 0:.{p}f}" if len(v) > 1 else f"{v.mean():.{p}f}"
def get(name, site, key, sub=None, sub2=None):
    out = []
    for r in R[name]:
        x = r["results"][site][key]
        if sub: x = x[sub]
        if sub2: x = x[sub2]
        out.append(x)
    return out
def hm_se(auc, n1=250, n0=250):
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    return math.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n0 - 1) * (q2 - auc ** 2)) / (n1 * n0))
NAMES = [("pca", "PCA (linear AE, k=20)"), ("cnnae", "CNN-AE (unsup. adaptation of CNN)"), ("lstmae", "BiLSTM-AE (unsup. adaptation of BiLSTM)"),
         ("supcnn", "1-D CNN, supervised (label reference)"),
         ("pca_nd", "PCA, no DWT"), ("cnnae_nd", "CNN-AE, no DWT"), ("lstmae_nd", "BiLSTM-AE, no DWT"), ("supcnn_nd", "1-D CNN supervised, no DWT"), ("supcnn_pa", "1-D CNN trained on pseudo-anomalies (no real labels)"),
         ("main", "DAS-ULRANet (ours, full)"), ("nodwt", "DAS-ULRANet, no DWT")]
L = []
L.append("# DAS-ULRANet: experiment results (synthetic benchmark)\n")
L.append("All numbers are produced by `run.py` in this folder. Values: mean ± std over seeds (n shown). Threshold θ = 99th percentile of validation-normal scores (label-free). Test sets: 250 normal + 250 leak windows per site.\n")
L.append("## Table 1. In-domain performance (Site A)\n")
L.append("| Method | n | AUROC | AUPRC | Precision | Recall | F1 | FAR |\n|---|---|---|---|---|---|---|---|")
for k, n in NAMES:
    if not R[k]: continue
    L.append(f"| {n} | {len(R[k])} | {ms(get(k,'A','auroc'))} | {ms(get(k,'A','auprc'))} | {ms(get(k,'A','src','precision'))} | {ms(get(k,'A','src','recall'))} | {ms(get(k,'A','src','f1'))} | {ms(get(k,'A','src','far'))} |")
if R["main"]:
    a = np.mean(get("main", "A", "auroc")); L.append(f"\nApprox. 95% CI half-width of a single AUROC at this test size (Hanley–McNeil): ±{1.96*hm_se(a):.3f}. Differences smaller than this are not resolvable.\n")
L.append("## Table 2. AUROC by leak SNR bin (Site A)\n")
bins = ["-5..0", "0..5", "5..10", "10..15"]
L.append("| Method | " + " | ".join(b + " dB" for b in bins) + " |\n|---|" + "---|" * len(bins))
for k, n in NAMES:
    if R[k]: L.append(f"| {n} | " + " | ".join(ms(get(k, 'A', 'snr_bins', b)) for b in bins) + " |")
L.append("\n## Table 3. Zero-shot transfer: trained on Site A only, tested on B (noisier) and C (quiet, more bad channels)\n")
L.append("Normalisation statistics from Site A. 'Source θ' = threshold from Site A validation; 'Recalibrated θ' = 99th pct of 100 normal target windows.\n")
L.append("| Method | Site | AUROC | F1 (source θ) | FAR (source θ) | Recall (source θ) | F1 (recal. θ) | FAR (recal. θ) | Recall (recal. θ) |\n|---|---|---|---|---|---|---|---|---|")
for k, n in NAMES:
    if not R[k]: continue
    for s in "ABC":
        L.append(f"| {n} | {s} | {ms(get(k,s,'auroc'))} | {ms(get(k,s,'src','f1'))} | {ms(get(k,s,'src','far'))} | {ms(get(k,s,'src','recall'))} | {ms(get(k,s,'recal','f1'))} | {ms(get(k,s,'recal','far'))} | {ms(get(k,s,'recal','recall'))} |")
L.append("\n## Table 4. Ablation (Site A; B/C AUROC zero-shot)\n")
AB = [("main", "Full model (MDLRA, r=16, all loss terms, DWT)"), ("noattn", "No attention"), ("fullattn", "Dual-axis FULL softmax attention"),
      ("temporal", "MDLRA temporal branch only"), ("spatial", "MDLRA spatial branch only"), ("nocls", "No contrastive term (λ2=0)"),
      ("nounc", "No evidential term (λ3=0; recon-score only)"), ("reconly", "Reconstruction loss only"), ("nodwt", "No DWT denoising"), ("dwtuniv", "DWT, universal threshold (k=1.0)"),
      ("pa_step", "Pseudo-anomalies: step family only"), ("pa_bump", "Pseudo-anomalies: bump family only"), ("pa_noise", "Pseudo-anomalies: noise-burst family only"),
      ("r4", "Rank r=4"), ("r64", "Rank r=64")]
L.append("| Variant | n | AUROC A | F1 A | FAR A | AUROC B | AUROC C | Params |\n|---|---|---|---|---|---|---|---|")
for k, n in AB:
    if R[k]: L.append(f"| {n} | {len(R[k])} | {ms(get(k,'A','auroc'))} | {ms(get(k,'A','src','f1'))} | {ms(get(k,'A','src','far'))} | {ms(get(k,'B','auroc'))} | {ms(get(k,'C','auroc'))} | {R[k][0]['params']:,} |")
L.append("\n## Table 5. Score components and uncertainty (full model, Site A)\n")
if R["main"]:
    comp = [r["extra"]["components_auroc_A"] for r in R["main"]]
    L.append("| Score | AUROC |\n|---|---|")
    for key, nm in [("recon", "Reconstruction error only"), ("evidential_p", "Evidential leak probability only"), ("latent_dist", "Latent distance to normal centre")]:
        L.append(f"| {nm} | {ms([c.get(key) for c in comp])} |")
    L.append(f"| Combined score (used) | {ms(get('main','A','auroc'))} |")
    ex = [r["extra"] for r in R["main"]]
    L.append(f"\nECE of evidential probability: {ms([e['ece_p'] for e in ex])}. Mean vacuity (epistemic uncertainty): normal windows {ms([e['vacuity_normal'] for e in ex])}, leak windows {ms([e['vacuity_leak'] for e in ex])}. AUROC of vacuity for flagging misclassified windows: {ms([e['unc_err_auroc'] for e in ex])}.")
if LAT:
    L.append("\n## Table 6. Cost (CPU, 1 thread, batch 1; includes no preprocessing unless stated)\n")
    L.append("| Model | Params | Latency (ms/window) |\n|---|---|---|")
    for k, v in LAT.items():
        if k != "preprocessing_ms": L.append(f"| {k} | {v['params']:,} | {v['ms']:.1f} |")
    if "preprocessing_ms" in LAT: L.append(f"\nPreprocessing (detrend + z-score + DWT) per window: {LAT['preprocessing_ms']:.2f} ms.")
open("results.md", "w", encoding="utf-8").write("\n".join(L)); print("\n".join(L))
