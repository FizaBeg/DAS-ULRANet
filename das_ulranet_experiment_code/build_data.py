import numpy as np, pickle, gen, time
t=time.time(); D={}
Xtr,mtr = gen.make_dataset("A",500,0,seed=1); Xval,mval = gen.make_dataset("A",150,0,seed=2)
Xsup,msup = gen.make_dataset("A",300,300,seed=7)
D["train"]=(Xtr,mtr); D["val"]=(Xval,mval); D["sup"]=(Xsup,msup)
D["dev_A"]=gen.make_dataset("A",250,250,seed=40); D["dev_B"]=gen.make_dataset("B",250,250,seed=41)
for i,s in enumerate("ABC"):
    D[f"test_{s}"]=gen.make_dataset(s,250,250,seed=10+i)
    D[f"calib_{s}"]=gen.make_dataset(s,100,0,seed=20+i)
print("raw generated",time.time()-t)
out={}
for key, kw in ((True, dict(dwt=True)), (False, dict(dwt=False)), ("univ", dict(dwt=True, k=1.0))):
    P=gen.Prep(**kw).fit(Xtr)
    out[key]={k:(P(v[0]),v[1]) for k,v in D.items()}
    print("prep", key, time.time()-t)
pickle.dump(out,open("data_cache.pkl","wb"),protocol=4)
print({k:v[0].shape for k,v in out[True].items()})
print(out[True]["test_A"][1].groupby("label").snr_db.describe())
