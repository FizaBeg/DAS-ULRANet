# DAS-ULRANet: experiment results (synthetic benchmark)

All numbers are produced by `run.py` in this folder. Values: mean ± std over seeds (n shown). Threshold θ = 99th percentile of validation-normal scores (label-free). Test sets: 250 normal + 250 leak windows per site.

## Table 1. In-domain performance (Site A)

| Method | n | AUROC | AUPRC | Precision | Recall | F1 | FAR |
|---|---|---|---|---|---|---|---|
| PCA (linear AE, k=20) | 1 | 0.836 | 0.866 | 0.966 | 0.448 | 0.612 | 0.016 |
| CNN-AE (unsup. adaptation of CNN) | 3 | 0.827 ± 0.002 | 0.857 ± 0.001 | 0.961 ± 0.009 | 0.428 ± 0.011 | 0.592 ± 0.009 | 0.017 ± 0.005 |
| BiLSTM-AE (unsup. adaptation of BiLSTM) | 3 | 0.843 ± 0.000 | 0.872 ± 0.000 | 0.967 ± 0.000 | 0.472 ± 0.000 | 0.634 ± 0.000 | 0.016 ± 0.000 |
| 1-D CNN, supervised (label reference) | 3 | 0.895 ± 0.013 | 0.923 ± 0.010 | 0.956 ± 0.027 | 0.684 ± 0.034 | 0.797 ± 0.018 | 0.032 ± 0.022 |
| DAS-ULRANet (ours, full) | 5 | 0.924 ± 0.009 | 0.946 ± 0.007 | 0.984 ± 0.007 | 0.802 ± 0.023 | 0.884 ± 0.014 | 0.013 ± 0.006 |

Approx. 95% CI half-width of a single AUROC at this test size (Hanley–McNeil): ±0.024. Differences smaller than this are not resolvable.

## Table 2. AUROC by leak SNR bin (Site A)

| Method | -5..0 dB | 0..5 dB | 5..10 dB | 10..15 dB |
|---|---|---|---|---|
| PCA (linear AE, k=20) | 0.634 | 0.796 | 0.966 | 0.992 |
| CNN-AE (unsup. adaptation of CNN) | 0.623 ± 0.004 | 0.774 ± 0.003 | 0.965 ± 0.001 | 0.991 ± 0.000 |
| BiLSTM-AE (unsup. adaptation of BiLSTM) | 0.645 ± 0.000 | 0.812 ± 0.000 | 0.966 ± 0.000 | 0.990 ± 0.000 |
| 1-D CNN, supervised (label reference) | 0.826 ± 0.029 | 0.905 ± 0.015 | 0.916 ± 0.007 | 0.946 ± 0.028 |
| DAS-ULRANet (ours, full) | 0.762 ± 0.029 | 0.953 ± 0.010 | 1.000 ± 0.000 | 1.000 ± 0.000 |

## Table 3. Zero-shot transfer: trained on Site A only, tested on B (noisier) and C (quiet, more bad channels)

Normalisation statistics from Site A. 'Source θ' = threshold from Site A validation; 'Recalibrated θ' = 99th pct of 100 normal target windows.

| Method | Site | AUROC | F1 (source θ) | FAR (source θ) | Recall (source θ) | F1 (recal. θ) | FAR (recal. θ) | Recall (recal. θ) |
|---|---|---|---|---|---|---|---|---|
| PCA (linear AE, k=20) | A | 0.836 | 0.612 | 0.016 | 0.448 | 0.670 | 0.048 | 0.528 |
| PCA (linear AE, k=20) | B | 0.809 | 0.667 | 1.000 | 1.000 | 0.553 | 0.004 | 0.384 |
| PCA (linear AE, k=20) | C | 0.788 | 0.497 | 0.004 | 0.332 | 0.506 | 0.004 | 0.340 |
| CNN-AE (unsup. adaptation of CNN) | A | 0.827 ± 0.002 | 0.592 ± 0.009 | 0.017 ± 0.005 | 0.428 ± 0.011 | 0.637 ± 0.017 | 0.036 ± 0.004 | 0.484 ± 0.021 |
| CNN-AE (unsup. adaptation of CNN) | B | 0.799 ± 0.002 | 0.667 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.598 ± 0.004 | 0.020 ± 0.000 | 0.435 ± 0.005 |
| CNN-AE (unsup. adaptation of CNN) | C | 0.780 ± 0.001 | 0.483 ± 0.008 | 0.004 ± 0.000 | 0.320 ± 0.007 | 0.485 ± 0.003 | 0.004 ± 0.000 | 0.321 ± 0.002 |
| BiLSTM-AE (unsup. adaptation of BiLSTM) | A | 0.843 ± 0.000 | 0.634 ± 0.000 | 0.016 ± 0.000 | 0.472 ± 0.000 | 0.698 ± 0.000 | 0.052 ± 0.000 | 0.564 ± 0.000 |
| BiLSTM-AE (unsup. adaptation of BiLSTM) | B | 0.811 ± 0.000 | 0.667 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.628 ± 0.000 | 0.032 ± 0.000 | 0.472 ± 0.000 |
| BiLSTM-AE (unsup. adaptation of BiLSTM) | C | 0.795 ± 0.000 | 0.501 ± 0.000 | 0.004 ± 0.000 | 0.336 ± 0.000 | 0.532 ± 0.000 | 0.004 ± 0.000 | 0.364 ± 0.000 |
| 1-D CNN, supervised (label reference) | A | 0.895 ± 0.013 | 0.797 ± 0.018 | 0.032 ± 0.022 | 0.684 ± 0.034 | 0.801 ± 0.024 | 0.027 ± 0.002 | 0.687 ± 0.034 |
| 1-D CNN, supervised (label reference) | B | 0.869 ± 0.014 | 0.784 ± 0.021 | 0.303 ± 0.089 | 0.839 ± 0.028 | 0.771 ± 0.024 | 0.020 ± 0.014 | 0.640 ± 0.028 |
| 1-D CNN, supervised (label reference) | C | 0.880 ± 0.016 | 0.759 ± 0.033 | 0.020 ± 0.011 | 0.625 ± 0.051 | 0.745 ± 0.030 | 0.013 ± 0.013 | 0.603 ± 0.047 |
| DAS-ULRANet (ours, full) | A | 0.924 ± 0.009 | 0.884 ± 0.014 | 0.013 ± 0.006 | 0.802 ± 0.023 | 0.870 ± 0.012 | 0.057 ± 0.009 | 0.814 ± 0.024 |
| DAS-ULRANet (ours, full) | B | 0.914 ± 0.097 | 0.667 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.661 ± 0.283 | 0.009 ± 0.009 | 0.542 ± 0.270 |
| DAS-ULRANet (ours, full) | C | 0.860 ± 0.007 | 0.781 ± 0.010 | 0.019 ± 0.010 | 0.653 ± 0.014 | 0.751 ± 0.047 | 0.008 ± 0.010 | 0.609 ± 0.065 |

## Table 4. Ablation (Site A; B/C AUROC zero-shot)

| Variant | n | AUROC A | F1 A | FAR A | AUROC B | AUROC C | Params |
|---|---|---|---|---|---|---|---|
| Full model (MDLRA, r=16, all loss terms, DWT) | 5 | 0.924 ± 0.009 | 0.884 ± 0.014 | 0.013 ± 0.006 | 0.914 ± 0.097 | 0.860 ± 0.007 | 45,364 |
| No attention | 5 | 0.911 ± 0.009 | 0.852 ± 0.016 | 0.014 ± 0.002 | 0.922 ± 0.017 | 0.845 ± 0.011 | 30,516 |
| Dual-axis FULL softmax attention | 5 | 0.921 ± 0.005 | 0.882 ± 0.010 | 0.010 ± 0.004 | 0.773 ± 0.322 | 0.855 ± 0.012 | 45,364 |
| MDLRA temporal branch only | 5 | 0.914 ± 0.009 | 0.867 ± 0.013 | 0.017 ± 0.003 | 0.938 ± 0.029 | 0.850 ± 0.009 | 45,364 |
| MDLRA spatial branch only | 5 | 0.927 ± 0.015 | 0.889 ± 0.019 | 0.013 ± 0.002 | 0.904 ± 0.073 | 0.869 ± 0.021 | 45,364 |
| No contrastive term (λ2=0) | 5 | 0.925 ± 0.008 | 0.885 ± 0.011 | 0.012 ± 0.005 | 0.943 ± 0.016 | 0.860 ± 0.016 | 45,364 |
| No evidential term (λ3=0; recon-score only) | 5 | 0.847 ± 0.006 | 0.643 ± 0.021 | 0.017 ± 0.002 | 0.813 ± 0.003 | 0.799 ± 0.007 | 45,364 |
| Reconstruction loss only | 5 | 0.843 ± 0.000 | 0.634 ± 0.000 | 0.016 ± 0.000 | 0.811 ± 0.000 | 0.795 ± 0.000 | 45,364 |
| No DWT denoising | 5 | 0.938 ± 0.013 | 0.898 ± 0.022 | 0.011 ± 0.005 | 0.921 ± 0.033 | 0.863 ± 0.011 | 45,364 |
| DWT, universal threshold (k=1.0) | 5 | 0.890 ± 0.008 | 0.789 ± 0.022 | 0.044 ± 0.005 | 0.859 ± 0.014 | 0.867 ± 0.004 | 45,364 |
| Rank r=4 | 5 | 0.926 ± 0.009 | 0.891 ± 0.012 | 0.013 ± 0.002 | 0.894 ± 0.090 | 0.861 ± 0.012 | 45,364 |
| Rank r=64 | 5 | 0.925 ± 0.010 | 0.882 ± 0.016 | 0.014 ± 0.002 | 0.951 ± 0.018 | 0.855 ± 0.009 | 45,364 |

## Table 5. Score components and uncertainty (full model, Site A)

| Score | AUROC |
|---|---|
| Reconstruction error only | 0.849 ± 0.006 |
| Evidential leak probability only | 0.960 ± 0.011 |
| Latent distance to normal centre | 0.926 ± 0.020 |
| Combined score (used) | 0.924 ± 0.009 |

ECE of evidential probability: 0.079 ± 0.009. Mean vacuity (epistemic uncertainty): normal windows 0.191 ± 0.018, leak windows 0.163 ± 0.025. AUROC of vacuity for flagging misclassified windows: 0.862 ± 0.027.

## Table 6. Cost (CPU, 1 thread, batch 1; includes no preprocessing unless stated)

| Model | Params | Latency (ms/window) |
|---|---|---|
| DAS-ULRANet (MDLRA r=16) | 45,364 | 16.1 |
| DAS-ULRANet (MDLRA r=4) | 45,364 | 16.5 |
| DAS-ULRANet (MDLRA r=64) | 45,364 | 17.8 |
| Dual-axis full softmax attention | 45,364 | 15.1 |
| No attention | 30,516 | 10.6 |
| CNN-AE | 230,336 | 1.9 |
| BiLSTM-AE | 205,408 | 1.8 |

Preprocessing (detrend + z-score + DWT) per window: 1.27 ms.