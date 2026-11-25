import sys, json, time, pickle, argparse, os, numpy as np, torch, torch.nn.functional as F
import gen, models

DEV = torch.device(os.environ.get('DEVICE') or ('cuda' if torch.cuda.is_available() else 'cpu'))
print('device:', DEV, torch.cuda.get_device_name(0) if DEV.type == 'cuda' else '', flush=True)

E, BS = int(os.environ.get("EPOCHS", 12)), 16
VARIANTS = {
 "main":     dict(attn="mdlra", r=16, lam=(1, .2, .2), dwt=True),
 "noattn":   dict(attn="none",  r=16, lam=(1, .2, .2), dwt=True),
 "fullattn": dict(attn="full",  r=16, lam=(1, .2, .2), dwt=True),
 "temporal": dict(attn="temporal", r=16, lam=(1, .2, .2), dwt=True),
 "spatial":  dict(attn="spatial",  r=16, lam=(1, .2, .2), dwt=True),
 "nocls":    dict(attn="mdlra", r=16, lam=(1, 0, .2), dwt=True),
 "nounc":    dict(attn="mdlra", r=16, lam=(1, .2, 0), dwt=True),
 "reconly":  dict(attn="mdlra", r=16, lam=(1, 0, 0), dwt=True),
 "r4":       dict(attn="mdlra", r=4,  lam=(1, .2, .2), dwt=True),
 "r64":      dict(attn="mdlra", r=64, lam=(1, .2, .2), dwt=True),
 "nodwt":    dict(attn="mdlra", r=16, lam=(1, .2, .2), dwt=False),
 "pa_step":  dict(attn="mdlra", r=16, lam=(1, .2, .2), dwt=True, kinds=("step",)),
 "pa_bump":  dict(attn="mdlra", r=16, lam=(1, .2, .2), dwt=True, kinds=("bump",)),
 "pa_noise": dict(attn="mdlra", r=16, lam=(1, .2, .2), dwt=True, kinds=("noise",)),
 "dwtuniv":  dict(attn="mdlra", r=16, lam=(1, .2, .2), dwt="univ"),
}
BASELINES = ["cnnae", "lstmae", "supcnn", "pca"]

# ---------------- metrics ----------------
def auroc(y, s):
    y = np.asarray(y); r = np.empty(len(s)); r[np.argsort(s, kind="stable")] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def auprc(y, s):
    o = np.argsort(-s); y = np.asarray(y)[o]; tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())
def prf(y, s, th):
    p = s >= th; y = np.asarray(y).astype(bool)
    tp, fp, fn, tn = (p & y).sum(), (p & ~y).sum(), (~p & y).sum(), (~p & ~y).sum()
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    return dict(precision=float(pr), recall=float(rc), f1=float(2 * pr * rc / max(pr + rc, 1e-9)), far=float(fp / max(fp + tn, 1)))
def ece(y, p, nb=10):
    b = np.minimum((p * nb).astype(int), nb - 1); e = 0
    for k in range(nb):
        m = b == k
        if m.any(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)
def sig(x): return 1 / (1 + np.exp(-x))

def site_metrics(y, s, th_src, th_cal, meta):
    out = dict(auroc=auroc(y, s), auprc=auprc(y, s), src=prf(y, s, th_src), recal=prf(y, s, th_cal))
    snr = meta.snr_db.values; bins = {}
    for lo, hi in [(-5, 0), (0, 5), (5, 10), (10, 15.01)]:
        sel = (y == 0) | ((snr >= lo) & (snr < hi)); bins[f"{lo}..{int(hi)}"] = auroc(y[sel], s[sel])
    out["snr_bins"] = bins; return out

# ---------------- losses ----------------
def kl_dir(alpha):
    K = alpha.shape[1]; S = alpha.sum(1, keepdim=True)
    return (torch.lgamma(S.squeeze(1)) - torch.lgamma(alpha).sum(1) - torch.lgamma(torch.tensor(float(K), device=alpha.device)) +
            ((alpha - 1) * (torch.digamma(alpha) - torch.digamma(S))).sum(1))
def evid_loss(ev, y, eta):
    alpha = ev + 1; S = alpha.sum(1, keepdim=True); p = alpha / S; Y = F.one_hot(y, 2).float()
    mse = ((Y - p) ** 2).sum(1) + (p * (1 - p) / (S + 1)).sum(1)
    at = Y + (1 - Y) * alpha
    return (mse + eta * kl_dir(at)).mean()

@torch.no_grad()
def run_model(model, Z, bs=50):
    model.eval(); recs, zs, evs = [], [], []
    for i in range(0, len(Z), bs):
        x = torch.from_numpy(Z[i:i + bs]).to(DEV); y, z, ev = model(x)
        recs.append(((y - x) ** 2).mean((1, 2)).cpu().numpy()); zs.append(z.cpu().numpy())
        evs.append(ev.cpu().numpy() if ev is not None else None)
    return np.concatenate(recs), np.concatenate(zs), (np.concatenate(evs) if evs[0] is not None else None)

# ---------------- training ----------------
def train_ulra(Ztr, vcfg, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    W, S = Ztr.shape[1:]; m = models.ULRANet(W, S, r=vcfg["r"], attn=vcfg["attn"]).to(DEV)
    l1, l2, l3 = vcfg["lam"]; use_pa = (l2 > 0 or l3 > 0)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=E * (len(Ztr) // BS))
    Zt = torch.from_numpy(Ztr); hist = []
    for ep in range(E):
        m.train(); perm = rng.permutation(len(Ztr)); tot = 0; nb = 0; eta = min(1.0, ep / 5)
        for i in range(0, len(perm) - BS + 1, BS):
            xn = Zt[perm[i:i + BS]].clone(); y = torch.zeros(BS, dtype=torch.long)
            if use_pa:
                xn[:BS // 2] = torch.from_numpy(gen.inject_pseudo(xn[:BS // 2].numpy(), rng, kinds=vcfg.get("kinds", ("bump", "step", "noise")))); y[:BS // 2] = 1
            xn = xn.to(DEV); y = y.to(DEV)
            rec, z, ev = m(xn); normal = (y == 0)
            loss = l1 * ((rec[normal] - xn[normal]) ** 2).mean()
            if l2 > 0:
                zn = z[normal]
                if m.c_init.item() == 0: m.c.copy_(zn.detach().mean(0)); m.c_init.fill_(1)
                else: m.c.mul_(.9).add_(.1 * zn.detach().mean(0))
                dn = ((zn - m.c) ** 2).mean(1); da = torch.sqrt(((z[~normal] - m.c) ** 2).mean(1) + 1e-8)
                loss = loss + l2 * (dn.mean() + (F.relu(1.0 - da) ** 2).mean())
            if l3 > 0: loss = loss + l3 * evid_loss(ev, y, eta)
            if not torch.isfinite(loss): sched.step(); continue          # NaN guard: skip bad batch
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.); opt.step(); sched.step()
            tot += loss.item(); nb += 1
        hist.append(tot / max(nb, 1))
    return m, hist

def train_ae(kind, Ztr, seed, sup_data=None):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    W, S = (Ztr if Ztr is not None else sup_data[0]).shape[1:]
    m = (models.CNNAE(W, S, sup=(kind == "supcnn")) if kind in ("cnnae", "supcnn") else models.BiLSTMAE(W, S)).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3); hist = []
    X, Y = (Ztr, None) if sup_data is None else sup_data
    Xt = torch.from_numpy(X); Yt = None if Y is None else torch.from_numpy(Y).float()
    for ep in range(E):
        m.train(); perm = rng.permutation(len(X)); tot = nb = 0
        for i in range(0, len(perm) - BS + 1, BS):
            idx = perm[i:i + BS]; x = Xt[idx].to(DEV); rec, z, lg = m(x)
            loss = F.binary_cross_entropy_with_logits(lg, Yt[idx].to(DEV)) if Yt is not None else ((rec - x) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); nb += 1
        hist.append(tot / max(nb, 1))
    return m, hist

def eval_all(score_fn, data, th_pct=99):
    sv = score_fn(data["val"][0]); th_src = np.quantile(sv, th_pct / 100); res = {}
    for s in "ABC":
        Xte, meta = data[f"test_{s}"]; y = meta.label.values; sc = score_fn(Xte)
        th_cal = np.quantile(score_fn(data[f"calib_{s}"][0]), th_pct / 100)
        res[s] = site_metrics(y, sc, th_src, th_cal, meta)
    return res

def job(name, seed, D):
    t0 = time.time()
    base = name[:-3] if name.endswith("_nd") else name      # "_nd" = baseline on data WITHOUT DWT
    key = False if name.endswith("_nd") else True
    if name in VARIANTS:
        vc = VARIANTS[name]; data = D[vc["dwt"]]; Ztr = data["train"][0]
        m, hist = train_ulra(Ztr, vc, seed)
        rv, _, _ = run_model(m, data["val"][0]); mu, sd = rv.mean(), rv.std() + 1e-9
        combined = vc["lam"][2] > 0
        cache = {}
        fw = {"w": 0.5}                                         # weight of the reconstruction score
        def parts(X):
            k = id(X)
            if k not in cache:
                r, z, ev = run_model(m, X); cache[k] = (r, z, ev)
            return cache[k]
        def score_fn(X):
            r, z, ev = parts(X); s_rec = sig((r - mu) / sd)
            if not combined: return s_rec
            p = ev[:, 1] / ev.sum(1) if False else (ev[:, 1] + 1) / (ev.sum(1) + 2)
            return fw["w"] * s_rec + (1 - fw["w"]) * p
        if combined and "dev_A" in data:
            best = None
            for w in (0.0, 0.25, 0.5, 0.75, 1.0):
                fw["w"] = w
                a_dev = np.mean([auroc(data[k][1].label.values, score_fn(data[k][0])) for k in ("dev_A", "dev_B")])
                if best is None or a_dev > best[0]: best = (a_dev, w)
            fw["w"] = best[1]
        res = eval_all(score_fn, data)
        extra = {"fusion_w": fw["w"]}
        if name in ("main", "r4", "r64"):                      # score-component diagnostics + uncertainty (site A)
            Xte, meta = data["test_A"]; y = meta.label.values; r, z, ev = parts(Xte)
            comp = dict(recon=auroc(y, sig((r - mu) / sd)))
            if combined:
                p = (ev[:, 1] + 1) / (ev.sum(1) + 2); vac = 2 / (ev.sum(1) + 2)
                comp["evidential_p"] = auroc(y, p); comp["latent_dist"] = auroc(y, np.sqrt(((z - m.c.cpu().numpy()) ** 2).mean(1)))
                s = score_fn(Xte); th = np.quantile(score_fn(data["val"][0]), .99); pred = (s >= th).astype(int)
                err = (pred != y).astype(int)
                extra = dict(ece_p=ece(y, p), vacuity_normal=float(vac[y == 0].mean()), vacuity_leak=float(vac[y == 1].mean()),
                             unc_err_auroc=auroc(err, vac) if 0 < err.sum() < len(err) else None)
            extra["components_auroc_A"] = comp; extra["fusion_w"] = fw["w"]
        params = sum(p.numel() for p in m.parameters())
    elif base in ("cnnae", "lstmae"):
        data = D[key]; m, hist = train_ae(base, data["train"][0], seed)
        rv, _, _ = run_model(m, data["val"][0]); mu, sd = rv.mean(), rv.std() + 1e-9
        res = eval_all(lambda X: sig((run_model(m, X)[0] - mu) / sd), data); extra = {}; params = sum(p.numel() for p in m.parameters())
    elif base in ("supcnn", "supcnn_pa"):
        data = D[key]
        if base == "supcnn_pa":      # same supervision as ULRANet: normal + injected pseudo-anomalies, no real leak labels
            Ztr = data["train"][0]; Xs = np.concatenate([Ztr, gen.inject_pseudo(Ztr, np.random.default_rng(seed + 100))])
            ys = np.concatenate([np.zeros(len(Ztr)), np.ones(len(Ztr))]).astype(np.float32)
        else:
            Xs, ms = data["sup"]; ys = ms.label.values.astype(np.float32)
        m, hist = train_ae("supcnn", None, seed, sup_data=(Xs, ys))
        @torch.no_grad()
        def sc(X):
            m.eval(); return np.concatenate([torch.sigmoid(m(torch.from_numpy(X[i:i + 50]).to(DEV))[2]).cpu().numpy() for i in range(0, len(X), 50)])
        res = eval_all(sc, data); extra = {}; params = sum(p.numel() for p in m.parameters())
    elif base == "pca":
        data = D[key]; Z = data["train"][0].reshape(len(data["train"][0]), -1); zm = Z.mean(0); Zc = Z - zm
        w, U = np.linalg.eigh(Zc @ Zc.T); top = np.argsort(w)[::-1][:20]; V = Zc.T @ U[:, top] / np.sqrt(w[top])
        def sc(X):
            z = X.reshape(len(X), -1) - zm; return ((z - (z @ V) @ V.T) ** 2).mean(1)
        res = eval_all(sc, data); hist = []; extra = {}; params = int(V.size)
    return dict(name=name, seed=seed, epochs=E, hist=hist, results=res, extra=extra, params=int(params), secs=time.time() - t0)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", required=True); ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args(); torch.set_num_threads(a.threads); os.makedirs("results", exist_ok=True)
    D = pickle.load(open("data_cache.pkl", "rb"))
    for j in a.jobs.split(","):
        name, seed = j.split(":"); f = f"results/{name}_{seed}.json"
        if os.path.exists(f): continue
        r = job(name, int(seed), D); json.dump(r, open(f, "w"))
        print(name, seed, "done in %.0fs" % r["secs"], "A auroc %.3f" % r["results"]["A"]["auroc"], flush=True)
