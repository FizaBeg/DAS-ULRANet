import math, torch, torch.nn as nn, torch.nn.functional as F

def sinus_pe(n, d):
    pos = torch.arange(n).float()[:, None]; i = torch.arange(0, d, 2).float()[None, :]
    pe = torch.zeros(n, d); pe[:, 0::2] = torch.sin(pos / 10000 ** (i / d)); pe[:, 1::2] = torch.cos(pos / 10000 ** (i / d))
    return pe

class LowRankAttn(nn.Module):
    """softmax-kernel approximated by positive random features of rank r; cost O(n r d)."""
    def __init__(self, d, r, seed=0):
        super().__init__()
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("Om", torch.randn(r, d, generator=g)); self.r = r; self.d = d
    def phi(self, x, is_query):
        x = x * self.d ** -0.25
        proj = x @ self.Om.t()                                   # (...,n,r)
        proj = proj - (x ** 2).sum(-1, keepdim=True) / 2
        proj = proj - (proj.amax(-1, keepdim=True) if is_query else proj.amax((-1, -2), keepdim=True))
        return torch.exp(proj) / math.sqrt(self.r) + 1e-6
    def forward(self, z):                                        # z: (M,n,d)
        Q, K, V = self.phi(self.q(z), True), self.phi(self.k(z), False), self.v(z)
        kv = K.transpose(-1, -2) @ V                             # (M,r,d)
        den = Q @ K.sum(-2).unsqueeze(-1)                        # (M,n,1)
        return self.o((Q @ kv) / (den + 1e-6))

class FullAttn(nn.Module):
    def __init__(self, d, heads=1):
        super().__init__(); self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4)); self.d = d
    def forward(self, z):
        a = torch.softmax(self.q(z) @ self.k(z).transpose(-1, -2) / math.sqrt(self.d), -1)
        return self.o(a @ self.v(z))

class MDRLA(nn.Module):
    """dual-axis attention on H:(B,S,T,d). mode in {mdlra, full, temporal, spatial}"""
    def __init__(self, d, r, mode="mdlra"):
        super().__init__(); self.mode = mode
        mk = (lambda s: LowRankAttn(d, r, s)) if mode in ("mdlra", "temporal", "spatial") else (lambda s: FullAttn(d))
        self.t_att, self.s_att = mk(1), mk(2)
        self.gate = nn.Linear(2 * d, d); self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
    def forward(self, H):
        B, S, T, d = H.shape
        At = self.t_att(H.reshape(B * S, T, d)).reshape(B, S, T, d) if self.mode != "spatial" else torch.zeros_like(H)
        As = self.s_att(H.permute(0, 2, 1, 3).reshape(B * T, S, d)).reshape(B, T, S, d).permute(0, 2, 1, 3) if self.mode != "temporal" else torch.zeros_like(H)
        g = torch.sigmoid(self.gate(torch.cat([At, As], -1)))
        M = self.ln1(H + g * At + (1 - g) * As)
        return self.ln2(M + self.ffn(M))

class ULRANet(nn.Module):
    def __init__(self, W=512, S=128, d=32, dz=32, r=16, attn="mdlra"):
        super().__init__(); self.W, self.S, self.d, self.attn = W, S, d, attn
        st = (4, 2, 2); self.T = W // 16
        self.cnn = nn.Sequential(nn.Conv1d(1, 16, 7, st[0], 3), nn.BatchNorm1d(16), nn.GELU(),
                                 nn.Conv1d(16, 32, 5, st[1], 2), nn.BatchNorm1d(32), nn.GELU(),
                                 nn.Conv1d(32, d, 3, st[2], 1), nn.BatchNorm1d(d), nn.GELU())
        self.enc_lstm = nn.LSTM(d, d // 2, batch_first=True, bidirectional=True)
        self.att = MDRLA(d, r, attn) if attn != "none" else None
        self.pool_w = nn.Linear(d, 1); self.to_z = nn.Linear(2 * d, dz)
        self.dz_in = nn.Linear(dz, d); self.dec_pre = nn.Sequential(nn.Linear(d, d), nn.GELU())
        self.dec_lstm = nn.LSTM(d, d // 2, batch_first=True, bidirectional=True)
        self.up = nn.Sequential(nn.ConvTranspose1d(d, 32, 4, 2, 1), nn.BatchNorm1d(32), nn.GELU(),
                                nn.ConvTranspose1d(32, 16, 4, 2, 1), nn.BatchNorm1d(16), nn.GELU(),
                                nn.ConvTranspose1d(16, 1, 6, 4, 1))
        self.head = nn.Sequential(nn.Linear(dz, 32), nn.GELU(), nn.Linear(32, 2))
        self.register_buffer("c", torch.zeros(dz)); self.register_buffer("c_init", torch.zeros(1))
        self.register_buffer("pe_t", sinus_pe(self.T, d)); self.register_buffer("pe_s", sinus_pe(4096, d))
    def encode(self, x):                                       # x: (B,W,S)
        B = x.shape[0]; S = x.shape[2]
        h = self.cnn(x.permute(0, 2, 1).reshape(B * S, 1, self.W))      # (BS,d,T)
        h = h.permute(0, 2, 1); h, _ = self.enc_lstm(h)                  # (BS,T,d)
        h = h.reshape(B, S, self.T, self.d) + self.pe_t[None, None] + self.pe_s[:S][None, :, None, :]
        if self.att is not None: h = self.att(h)
        m = h.mean(2)                                                    # (B,S,d)
        w = torch.softmax(self.pool_w(m), 1)
        z = self.to_z(torch.cat([(w * m).sum(1), h.amax((1, 2))], -1))
        return z
    def decode(self, z, S):
        B = z.shape[0]
        base = self.dz_in(z)[:, None, None, :] + self.pe_s[:S][None, :, None, :] + self.pe_t[None, None]
        h = self.dec_pre(base).reshape(B * S, self.T, self.d)
        h, _ = self.dec_lstm(h)
        y = self.up(h.permute(0, 2, 1))                                  # (BS,1,W)
        return y.reshape(B, S, self.W).permute(0, 2, 1)
    def forward(self, x):
        z = self.encode(x); return self.decode(z, x.shape[2]), z, F.softplus(self.head(z))

# ---------------- baselines ----------------
class CNNAE(nn.Module):
    """1-D CNN autoencoder, channels as input maps (unsupervised adaptation of a CNN); sup=True adds a classifier head."""
    def __init__(self, W=512, S=128, dz=32, sup=False):
        super().__init__(); self.W = W
        self.enc = nn.Sequential(nn.Conv1d(S, 64, 7, 2, 3), nn.BatchNorm1d(64), nn.GELU(),
                                 nn.Conv1d(64, 64, 5, 2, 2), nn.BatchNorm1d(64), nn.GELU(),
                                 nn.Conv1d(64, 32, 3, 2, 1), nn.BatchNorm1d(32), nn.GELU())
        self.fc = nn.Linear(32 * (W // 8), dz); self.fc2 = nn.Linear(dz, 32 * (W // 8))
        self.dec = nn.Sequential(nn.ConvTranspose1d(32, 64, 4, 2, 1), nn.BatchNorm1d(64), nn.GELU(),
                                 nn.ConvTranspose1d(64, 64, 4, 2, 1), nn.BatchNorm1d(64), nn.GELU(),
                                 nn.ConvTranspose1d(64, S, 4, 2, 1))
        self.cls = nn.Linear(dz, 1) if sup else None
    def forward(self, x):
        h = self.enc(x.permute(0, 2, 1)); z = self.fc(h.flatten(1))
        y = self.dec(self.fc2(z).reshape(-1, 32, self.W // 8)).permute(0, 2, 1)
        return y, z, (self.cls(z).squeeze(-1) if self.cls is not None else None)

class BiLSTMAE(nn.Module):
    def __init__(self, W=512, S=128, dz=32, hid=64, pool=8):
        super().__init__(); self.pool, self.T = pool, W // pool
        self.enc = nn.LSTM(S, hid, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hid, dz); self.fc2 = nn.Linear(dz, hid)
        self.dec = nn.LSTM(hid, hid, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hid, S * pool); self.S = S
    def forward(self, x):
        B = x.shape[0]; xp = F.avg_pool1d(x.permute(0, 2, 1), self.pool).permute(0, 2, 1)   # (B,T,S)
        h, _ = self.enc(xp); z = self.fc(h.mean(1))
        d, _ = self.dec(self.fc2(z)[:, None, :].expand(-1, self.T, -1))
        y = self.out(d).reshape(B, self.T, self.pool, self.S).reshape(B, self.T * self.pool, self.S)
        return y, z, None
