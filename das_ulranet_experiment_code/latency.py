import time, json, numpy as np, torch, models, gen
torch.set_num_threads(1)
x = torch.randn(1, 512, 64); out = {}
cfgs = {"DAS-ULRANet (MDLRA r=16)": lambda: models.ULRANet(r=16), "DAS-ULRANet (MDLRA r=4)": lambda: models.ULRANet(r=4),
        "DAS-ULRANet (MDLRA r=64)": lambda: models.ULRANet(r=64), "Dual-axis full softmax attention": lambda: models.ULRANet(attn="full"),
        "No attention": lambda: models.ULRANet(attn="none"), "CNN-AE": lambda: models.CNNAE(512, 64), "BiLSTM-AE": lambda: models.BiLSTMAE(512, 64)}
for k, f in cfgs.items():
    m = f().eval(); ts = []
    with torch.no_grad():
        for _ in range(10): m(x)
        for _ in range(100):
            t = time.perf_counter(); m(x); ts.append(time.perf_counter() - t)
    out[k] = dict(params=sum(p.numel() for p in m.parameters()), ms=float(np.median(ts) * 1e3))
    print(k, out[k], flush=True)
X, _ = gen.make_dataset("A", 20, 0, seed=99); P = gen.Prep().fit(X); ts = []
for i in range(20):
    t = time.perf_counter(); P(X[i:i + 1]); ts.append(time.perf_counter() - t)
out["preprocessing_ms"] = float(np.median(ts) * 1e3); print("prep ms", out["preprocessing_ms"])
json.dump(out, open("latency.json", "w"))
