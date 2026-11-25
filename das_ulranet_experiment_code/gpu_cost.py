import time, json, numpy as np, torch, models
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def sync():
    if dev.type == "cuda": torch.cuda.synchronize()
def timeit(f, n=50, warm=10):
    for _ in range(warm): f()
    sync(); ts = []
    for _ in range(n):
        t = time.perf_counter(); f(); sync(); ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1e3)
print("device:", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "")
cfgs = {"DAS-ULRANet (MDLRA r=16)": lambda: models.ULRANet(r=16), "DAS-ULRANet (MDLRA r=4)": lambda: models.ULRANet(r=4),
        "DAS-ULRANet (MDLRA r=64)": lambda: models.ULRANet(r=64), "Dual-axis full softmax attention": lambda: models.ULRANet(attn="full"),
        "No attention": lambda: models.ULRANet(attn="none"), "CNN-AE": lambda: models.CNNAE(512, 64), "BiLSTM-AE": lambda: models.BiLSTMAE(512, 64)}
out = {"device": str(dev), "latency": {}, "scaling": []}
for k, f in cfgs.items():
    m = f().to(dev).eval(); x1 = torch.randn(1, 512, 64, device=dev); x64 = torch.randn(64, 512, 64, device=dev)
    with torch.no_grad():
        b1 = timeit(lambda: m(x1)); b64 = timeit(lambda: m(x64), n=20, warm=3) / 64
    out["latency"][k] = dict(params=sum(p.numel() for p in m.parameters()), ms_batch1=b1, ms_per_window_batch64=b64); print(k, out["latency"][k], flush=True)
d, T = 32, 32
for S in (64, 256, 1024, 2048, 4096):
    row = dict(S=S)
    for name, mode in (("lowrank_r16", "mdlra"), ("full_softmax", "full")):
        try:
            m = models.MDRLA(d, 16, mode).to(dev).eval(); H = torch.randn(1, S, T, d, device=dev)
            if dev.type == "cuda": torch.cuda.reset_peak_memory_stats()
            with torch.no_grad(): row[name] = round(timeit(lambda: m(H), n=10, warm=3), 2)
            if dev.type == "cuda": row[name + "_peakMB"] = round(torch.cuda.max_memory_allocated() / 2**20)
        except Exception as e:
            row[name] = "failed: " + type(e).__name__
            if dev.type == "cuda": torch.cuda.empty_cache()
    out["scaling"].append(row); print(row, flush=True)
json.dump(out, open("gpu_cost.json", "w"), indent=1)
