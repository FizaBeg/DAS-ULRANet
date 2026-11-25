# DAS-ULRANet: experiment results (synthetic benchmark)

All numbers are produced by `run.py` in this folder. Values: mean ± std over seeds (n shown). Threshold θ = 99th percentile of validation-normal scores (label-free). Test sets: 250 normal + 250 leak windows per site.

## Table 1. In-domain performance (Site A)

| Method | n | AUROC | AUPRC | Precision | Recall | F1 | FAR |
|---|---|---|---|---|---|---|---|
| PCA (linear AE, k=20) | 1 | 0.836 | 0.866 | 0.966 | 0.448 | 0.612 | 0.016 |
| CNN-AE (unsup. adaptation of CNN) | 3 | 0.827 ± 0.002 | 0.857 ± 0.001 | 0.961 ± 0.009 | 0.428 ± 0.011 | 0.592 ± 0.009 | 0.017 ± 0.005 |
| BiLSTM-AE (unsup. adaptation of BiLSTM) | 3 | 0.843 ± 0.000 | 0.872 ± 0.000 | 0.967 ± 0.000 | 0.472 ± 0.000 | 0.634 ± 0.000 | 0.016 ± 0.000 |
| 1-D CNN, supervised (label reference) | 3 | 0.895 ± 0.013 | 0.923 ± 0.010 | 0.956 ± 0.027 | 0.684 ± 0.034 | 0.797 ± 0.018 | 0.032 ± 0.022 |
| PCA, no DWT | 1 | 0.799 | 0.829 | 0.955 | 0.340 | 0.501 | 0.016 |
| CNN-AE, no DWT | 3 | 0.795 ± 0.002 | 0.826 ± 0.001 | 0.963 ± 0.005 | 0.343 ± 0.008 | 0.505 ± 0.008 | 0.013 ± 0.002 |
| BiLSTM-AE, no DWT | 3 | 0.801 ± 0.000 | 0.831 ± 0.000 | 0.957 ± 0.000 | 0.357 ± 0.002 | 0.520 ± 0.002 | 0.016 ± 0.000 |
| 1-D CNN supervised, no DWT | 3 | 0.863 ± 0.016 | 0.898 ± 0.011 | 0.967 ± 0.007 | 0.595 ± 0.015 | 0.737 ± 0.014 | 0.020 ± 0.004 |
| 1-D CNN trained on pseudo-anomalies (no real labels) | 3 | 0.913 ± 0.013 | 0.934 ± 0.012 | 0.982 ± 0.019 | 0.684 ± 0.057 | 0.805 ± 0.036 | 0.013 ± 0.014 |
| DAS-ULRANet (ours, full) | 5 | 0.965 ± 0.012 | 0.975 ± 0.008 | 0.983 ± 0.014 | 0.866 ± 0.020 | 0.921 ± 0.012 | 0.015 ± 0.012 |
| DAS-ULRANet, no DWT | 5 | 0.962 ± 0.016 | 0.973 ± 0.011 | 0.984 ± 0.013 | 0.865 ± 0.032 | 0.920 ± 0.018 | 0.014 ± 0.012 |

Approx. 95% CI half-width of a single AUROC at this test size (Hanley–McNeil): ±0.017. Differences smaller than this are not resolvable.

## Table 2. AUROC by leak SNR bin (Site A)

| Method | -5..0 dB | 0..5 dB | 5..10 dB | 10..15 dB |
|---|---|---|---|---|
| PCA (linear AE, k=20) | 0.634 | 0.796 | 0.966 | 0.992 |
| CNN-AE (unsup. adaptation of CNN) | 0.623 ± 0.004 | 0.774 ± 0.003 | 0.965 ± 0.001 | 0.991 ± 0.000 |
| BiLSTM-AE (unsup. adaptation of BiLSTM) | 0.645 ± 0.000 | 0.812 ± 0.000 | 0.966 ± 0.000 | 0.990 ± 0.000 |
| 1-D CNN, supervised (label reference) | 0.826 ± 0.029 | 0.905 ± 0.015 | 0.916 ± 0.007 | 0.946 ± 0.028 |
| PCA, no DWT | 0.603 | 0.717 | 0.946 | 0.983 |
| CNN-AE, no DWT | 0.601 ± 0.002 | 0.706 ± 0.004 | 0.944 ± 0.001 | 0.984 ± 0.000 |
| BiLSTM-AE, no DWT | 0.606 ± 0.000 | 0.721 ± 0.000 | 0.945 ± 0.000 | 0.982 ± 0.000 |
| 1-D CNN supervised, no DWT | 0.762 ± 0.008 | 0.883 ± 0.024 | 0.894 ± 0.022 | 0.926 ± 0.032 |
| 1-D CNN trained on pseudo-anomalies (no real labels) | 0.805 ± 0.032 | 0.911 ± 0.012 | 0.962 ± 0.010 | 0.992 ± 0.005 |
| DAS-ULRANet (ours, full) | 0.877 ± 0.039 | 0.991 ± 0.010 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| DAS-ULRANet, no DWT | 0.862 ± 0.059 | 0.996 ± 0.003 | 1.000 ± 0.000 | 1.000 ± 0.000 |

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
| PCA, no DWT | A | 0.799 | 0.501 | 0.016 | 0.340 | 0.571 | 0.040 | 0.416 |
| PCA, no DWT | B | 0.769 | 0.667 | 1.000 | 1.000 | 0.522 | 0.008 | 0.356 |
| PCA, no DWT | C | 0.741 | 0.431 | 0.004 | 0.276 | 0.465 | 0.004 | 0.304 |
| CNN-AE, no DWT | A | 0.795 ± 0.002 | 0.505 ± 0.008 | 0.013 ± 0.002 | 0.343 ± 0.008 | 0.565 ± 0.006 | 0.037 ± 0.002 | 0.408 ± 0.007 |
| CNN-AE, no DWT | B | 0.767 ± 0.001 | 0.667 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.536 ± 0.010 | 0.009 ± 0.002 | 0.369 ± 0.010 |
| CNN-AE, no DWT | C | 0.739 ± 0.000 | 0.425 ± 0.008 | 0.004 ± 0.000 | 0.271 ± 0.006 | 0.451 ± 0.008 | 0.004 ± 0.000 | 0.292 ± 0.007 |
| BiLSTM-AE, no DWT | A | 0.801 ± 0.000 | 0.520 ± 0.002 | 0.016 ± 0.000 | 0.357 ± 0.002 | 0.565 ± 0.000 | 0.036 ± 0.000 | 0.408 ± 0.000 |
| BiLSTM-AE, no DWT | B | 0.769 ± 0.000 | 0.667 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.561 ± 0.002 | 0.012 ± 0.000 | 0.395 ± 0.002 |
| BiLSTM-AE, no DWT | C | 0.742 ± 0.000 | 0.433 ± 0.003 | 0.004 ± 0.000 | 0.277 ± 0.002 | 0.465 ± 0.000 | 0.004 ± 0.000 | 0.304 ± 0.000 |
| 1-D CNN supervised, no DWT | A | 0.863 ± 0.016 | 0.737 ± 0.014 | 0.020 ± 0.004 | 0.595 ± 0.015 | 0.738 ± 0.007 | 0.021 ± 0.014 | 0.597 ± 0.016 |
| 1-D CNN supervised, no DWT | B | 0.840 ± 0.021 | 0.742 ± 0.020 | 0.192 ± 0.160 | 0.707 ± 0.126 | 0.665 ± 0.024 | 0.016 ± 0.017 | 0.507 ± 0.036 |
| 1-D CNN supervised, no DWT | C | 0.864 ± 0.019 | 0.707 ± 0.035 | 0.011 ± 0.006 | 0.553 ± 0.039 | 0.683 ± 0.041 | 0.003 ± 0.002 | 0.521 ± 0.048 |
| 1-D CNN trained on pseudo-anomalies (no real labels) | A | 0.913 ± 0.013 | 0.805 ± 0.036 | 0.013 ± 0.014 | 0.684 ± 0.057 | 0.808 ± 0.034 | 0.017 ± 0.008 | 0.691 ± 0.054 |
| 1-D CNN trained on pseudo-anomalies (no real labels) | B | 0.919 ± 0.009 | 0.691 ± 0.015 | 0.873 ± 0.068 | 0.988 ± 0.011 | 0.807 ± 0.025 | 0.004 ± 0.000 | 0.680 ± 0.036 |
| 1-D CNN trained on pseudo-anomalies (no real labels) | C | 0.895 ± 0.012 | 0.744 ± 0.057 | 0.003 ± 0.002 | 0.596 ± 0.075 | 0.786 ± 0.036 | 0.016 ± 0.007 | 0.659 ± 0.050 |
| DAS-ULRANet (ours, full) | A | 0.965 ± 0.012 | 0.921 ± 0.012 | 0.015 ± 0.012 | 0.866 ± 0.020 | 0.917 ± 0.004 | 0.012 ± 0.012 | 0.857 ± 0.007 |
| DAS-ULRANet (ours, full) | B | 0.954 ± 0.006 | 0.667 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.758 ± 0.113 | 0.006 ± 0.008 | 0.625 ± 0.154 |
| DAS-ULRANet (ours, full) | C | 0.877 ± 0.024 | 0.804 ± 0.041 | 0.126 ± 0.149 | 0.752 ± 0.034 | 0.809 ± 0.019 | 0.005 ± 0.007 | 0.682 ± 0.028 |
| DAS-ULRANet, no DWT | A | 0.962 ± 0.016 | 0.920 ± 0.018 | 0.014 ± 0.012 | 0.865 ± 0.032 | 0.918 ± 0.022 | 0.019 ± 0.009 | 0.866 ± 0.036 |
| DAS-ULRANet, no DWT | B | 0.866 ± 0.108 | 0.667 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.580 ± 0.269 | 0.010 ± 0.008 | 0.456 ± 0.285 |
| DAS-ULRANet, no DWT | C | 0.870 ± 0.030 | 0.801 ± 0.038 | 0.090 ± 0.088 | 0.726 ± 0.049 | 0.793 ± 0.024 | 0.010 ± 0.008 | 0.664 ± 0.038 |

## Table 4. Ablation (Site A; B/C AUROC zero-shot)

| Variant | n | AUROC A | F1 A | FAR A | AUROC B | AUROC C | Params |
|---|---|---|---|---|---|---|---|
| Full model (MDLRA, r=16, all loss terms, DWT) | 5 | 0.965 ± 0.012 | 0.921 ± 0.012 | 0.015 ± 0.012 | 0.954 ± 0.006 | 0.877 ± 0.024 | 45,364 |
| No attention | 5 | 0.949 ± 0.010 | 0.880 ± 0.010 | 0.012 ± 0.004 | 0.916 ± 0.019 | 0.856 ± 0.009 | 30,516 |
| Dual-axis FULL softmax attention | 5 | 0.944 ± 0.052 | 0.873 ± 0.120 | 0.020 ± 0.010 | 0.889 ± 0.054 | 0.864 ± 0.039 | 45,364 |
| MDLRA temporal branch only | 5 | 0.952 ± 0.010 | 0.909 ± 0.012 | 0.013 ± 0.010 | 0.946 ± 0.012 | 0.859 ± 0.023 | 45,364 |
| MDLRA spatial branch only | 5 | 0.967 ± 0.009 | 0.923 ± 0.013 | 0.022 ± 0.016 | 0.909 ± 0.046 | 0.879 ± 0.025 | 45,364 |
| No contrastive term (λ2=0) | 5 | 0.967 ± 0.013 | 0.919 ± 0.016 | 0.006 ± 0.007 | 0.923 ± 0.014 | 0.859 ± 0.016 | 45,364 |
| No evidential term (λ3=0; recon-score only) | 5 | 0.847 ± 0.006 | 0.643 ± 0.021 | 0.017 ± 0.002 | 0.813 ± 0.003 | 0.799 ± 0.007 | 45,364 |
| Reconstruction loss only | 5 | 0.843 ± 0.000 | 0.634 ± 0.000 | 0.016 ± 0.000 | 0.811 ± 0.000 | 0.795 ± 0.000 | 45,364 |
| No DWT denoising | 5 | 0.962 ± 0.016 | 0.920 ± 0.018 | 0.014 ± 0.012 | 0.866 ± 0.108 | 0.870 ± 0.030 | 45,364 |
| DWT, universal threshold (k=1.0) | 5 | 0.889 ± 0.014 | 0.779 ± 0.038 | 0.043 ± 0.004 | 0.865 ± 0.005 | 0.868 ± 0.012 | 45,364 |
| Pseudo-anomalies: step family only | 3 | 0.843 ± 0.000 | 0.634 ± 0.001 | 0.017 ± 0.002 | 0.811 ± 0.000 | 0.795 ± 0.000 | 45,364 |
| Pseudo-anomalies: bump family only | 3 | 0.904 ± 0.012 | 0.817 ± 0.017 | 0.016 ± 0.000 | 0.824 ± 0.012 | 0.847 ± 0.007 | 45,364 |
| Pseudo-anomalies: noise-burst family only | 3 | 0.848 ± 0.007 | 0.649 ± 0.027 | 0.017 ± 0.002 | 0.812 ± 0.000 | 0.802 ± 0.011 | 45,364 |
| Rank r=4 | 5 | 0.937 ± 0.047 | 0.865 ± 0.111 | 0.012 ± 0.005 | 0.925 ± 0.065 | 0.852 ± 0.027 | 45,364 |
| Rank r=64 | 5 | 0.969 ± 0.007 | 0.928 ± 0.008 | 0.014 ± 0.012 | 0.915 ± 0.076 | 0.880 ± 0.016 | 45,364 |

## Table 5. Score components and uncertainty (full model, Site A)

| Score | AUROC |
|---|---|
| Reconstruction error only | 0.852 ± 0.010 |
| Evidential leak probability only | 0.965 ± 0.012 |
| Latent distance to normal centre | 0.929 ± 0.014 |
| Combined score (used) | 0.965 ± 0.012 |

ECE of evidential probability: 0.085 ± 0.011. Mean vacuity (epistemic uncertainty): normal windows 0.197 ± 0.015, leak windows 0.168 ± 0.011. AUROC of vacuity for flagging misclassified windows: 0.811 ± 0.041.

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