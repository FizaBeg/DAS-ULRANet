"""Significance numbers for the paper.   %run stats_report.py        (writes stats_report.md)
 * synthetic benchmark (results/*.json): ULRANet ('main') vs every other method, per site, over seeds  -> Welch t-test + bootstrap CI of the mean AUROC difference
 * Otway hybrid (results_otway/*.json): same, but PAIRED over (fold, seed)                          -> paired t-test + bootstrap CI
Small n (3-5 seeds): read the CI first; a p-value from n=5 is a rough guide only."""
import json, glob, collections, numpy as np
from scipy import stats

rng = np.random.default_rng(0)
def boot_ci(d, n=5000):
    d = np.asarray(d, float); m = [rng.choice(d, len(d)).mean() for _ in range(n)]; return np.percentile(m, [2.5, 97.5])

def load(pattern, key):
    R = collections.defaultdict(dict)
    for f in glob.glob(pattern):
        r = json.load(open(f)); R[r["name"]][key(r)] = {s: r["results"][s]["auroc"] for s in "ABC"}
    return R

def compare(R, ref, paired, sites, lines, title):
    lines += [f"\n### {title}\n", "| vs | site | mean diff (ref - other) | 95% bootstrap CI | p | n |", "|---|---|---|---|---|---|"]
    for other in sorted(R):
        if other == ref or other not in R or ref not in R: continue
        for s in sites:
            if paired:
                keys = sorted(set(R[ref]) & set(R[other]))
                x = np.array([R[ref][k][s] for k in keys]); y = np.array([R[other][k][s] for k in keys]); ok = ~(np.isnan(x) | np.isnan(y))
                x, y = x[ok], y[ok]
                if len(x) < 3: continue
                d = x - y; p = stats.ttest_rel(x, y).pvalue; ci = boot_ci(d); n = len(x)
            else:
                x = np.array([v[s] for v in R[ref].values()]); y = np.array([v[s] for v in R[other].values()])
                if len(x) < 2 or len(y) < 2: continue
                d = np.array([a - b for a in x for b in y]); p = stats.ttest_ind(x, y, equal_var=False).pvalue
                ci = np.percentile([rng.choice(x, len(x)).mean() - rng.choice(y, len(y)).mean() for _ in range(5000)], [2.5, 97.5]); n = f"{len(x)}/{len(y)}"
            lines.append(f"| {other} | {s} | {d.mean():+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | {p:.3g} | {n} |")

lines = ["# Significance report (AUROC differences, reference = DAS-ULRANet 'main')"]
Rs = load("results/*.json", key=lambda r: r["seed"])
if Rs: compare(Rs, "main", False, "AC", lines, "Synthetic benchmark: A = in-domain, B/C = zero-shot sites (Welch, over seeds)")
if Rs: compare(Rs, "main", False, "B", lines, "Synthetic benchmark: Site B")
Ro = load("results_otway/*.json", key=lambda r: (r["fold"], r["seed"]))
if Ro: compare(Ro, "main", True, "AC", lines, "Otway hybrid: A = injected leak (held-out well), C = second held-out well (paired over fold x seed)")
if Ro: compare(Ro, "main", True, "B", lines, "Otway hybrid: B = real microseismic events (not leaks)")
open("stats_report.md", "w", encoding="utf-8").write("\n".join(lines) + "\n"); print("\n".join(lines))
