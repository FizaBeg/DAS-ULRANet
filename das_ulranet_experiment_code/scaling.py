import time, json, torch, models
torch.set_num_threads(1); torch.manual_seed(0); out = []
d, T = 32, 32
for S in (64, 256, 1024, 2048, 4096):
    row = dict(S=S)
    for name, mode in (("lowrank_r16", "mdlra"), ("full_softmax", "full")):
        m = models.MDRLA(d, 16, mode).eval(); H = torch.randn(1, S, T, d)
        try:
            with torch.no_grad():
                m(H); ts = []
                for _ in range(3):
                    t = time.perf_counter(); m(H); ts.append(time.perf_counter() - t)
            row[name] = round(sorted(ts)[1] * 1e3, 1)
        except Exception as e:
            row[name] = "failed: " + type(e).__name__
    out.append(row); print(row, flush=True)
json.dump(out, open("scaling.json", "w"))
