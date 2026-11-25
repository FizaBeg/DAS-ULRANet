import pickle, json, torch, run
torch.set_num_threads(1); run.E = 40
D = pickle.load(open("data_cache.pkl", "rb"))
for s in range(3):
    r = run.job("supcnn", s, D); r["name"] = "supcnn40"; r["epochs"] = 40
    json.dump(r, open(f"results/supcnn40_{s}.json", "w")); print("supcnn40", s, round(r["results"]["A"]["auroc"], 3), flush=True)
