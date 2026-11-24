"""05・08で共通に使う特徴量の作成。基準日以前の購買履歴だけから作る(未来の情報は使わない)。"""
import pandas as pd, numpy as np

def make_build(p):
    def build(cutoff, horizon_days=180):
        c = pd.Timestamp(cutoff); end = c + pd.Timedelta(days=horizon_days)
        hist = p[p.InvoiceDate < c]; win = hist[hist.InvoiceDate >= c - pd.Timedelta(days=180)]
        hb = hist[~hist.is_cancel]; wb = win[~win.is_cancel]; wc = win[win.is_cancel]
        X = pd.DataFrame({
            "recency": (c - hb.groupby("CustomerID").InvoiceDate.max()).dt.days,
            "freq180": wb.groupby("CustomerID").Invoice.nunique(),
            "sales180": wb.groupby("CustomerID").Sales.sum(),
            "items180": wb.groupby("CustomerID").StockCode.nunique(),
            "return_rate180": (-wc.groupby("CustomerID").Sales.sum()),
            "is_new": (hb.groupby("CustomerID").InvoiceDate.min() >= c - pd.Timedelta(days=180)).astype(int),
            "is_uk": hb.groupby("CustomerID").Country.agg(lambda s: int(s.mode().iat[0] == "United Kingdom")),
        })
        X = X[X.recency.notna()].copy()          # 基準日以前に購入実績のある顧客のみ
        X[["freq180", "sales180", "items180", "return_rate180"]] = X[["freq180", "sales180", "items180", "return_rate180"]].fillna(0)
        X["return_rate180"] = (X.return_rate180 / X.sales180.where(X.sales180 > 0)).fillna(0).clip(0, 1)
        X["aov180"] = (X.sales180 / X.freq180.where(X.freq180 > 0)).fillna(0)
        X = X.drop(columns="sales180")           # freq×aov と多重共線性になるため特徴量から除外
        fb = p[(p.InvoiceDate >= c) & (p.InvoiceDate < end) & ~p.is_cancel]
        y = pd.Series(X.index.isin(fb.CustomerID).astype(int), index=X.index, name="repurchase")
        fs = fb.groupby("CustomerID").Sales.sum().reindex(X.index).fillna(0)
        return X, y, fs
    return build

def tf(X):   # 回数・金額は対数変換
    Z = X.copy()
    for c in ["freq180", "items180", "aov180"]: Z[c] = np.log1p(Z[c].clip(lower=0))
    return Z


# ---- BG/NBD (Fader, Hardie & Lee, 2005) と Gamma-Gamma (Fader & Hardie, 2013)
from scipy.optimize import minimize
from scipy.special import gammaln

def cbt(p_buy, cutoff, idx=None):
    """顧客ごとの (x, t_x, T)。x=購入日数-1(同日の複数伝票は1回)、t_x=初回〜最終購入の週数、T=初回購入〜基準日の週数。基準日より前のデータのみ使う"""
    c = pd.Timestamp(cutoff)
    h = p_buy[p_buy.InvoiceDate < c].assign(d=lambda x: x.InvoiceDate.dt.normalize())
    g = h.groupby("CustomerID").d.agg(["min", "max", "nunique"])
    if idx is not None: g = g.loc[idx]
    out = pd.DataFrame({"x": (g["nunique"] - 1).astype(float), "t_x": (g["max"] - g["min"]).dt.days / 7, "T": (c - g["min"]).dt.days / 7}, index=g.index)
    assert (out.t_x <= out["T"]).all()
    return out

def _ll(params, x, tx, T):
    r, a_, aa, bb = np.exp(params)
    ln_a1 = gammaln(r + x) - gammaln(r) + r * np.log(a_)
    ln_a2 = gammaln(aa + bb) + gammaln(bb + x) - gammaln(bb) - gammaln(aa + bb + x)
    ln_a3 = -(r + x) * np.log(a_ + T)
    ln_a4 = np.where(x > 0, np.log(aa) - np.log(bb + np.maximum(x, 1) - 1) - (r + x) * np.log(a_ + tx), -np.inf)
    return ln_a1 + ln_a2 + np.logaddexp(ln_a3, ln_a4)

def fit_bgnbd(d):
    x, tx, T = d.x.values, d.t_x.values, d["T"].values
    best = None
    for init in ([0., 0., 0., 0.], [0., 1., 0., 1.], [-0.5, 2., -1., 0.5]):
        res = minimize(lambda th: -_ll(th, x, tx, T).sum(), init, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-8})
        if best is None or res.fun < best.fun: best = res
    return dict(zip(["r", "alpha", "a", "b"], np.exp(best.x))), best

def predict_bgnbd(par, d, days=180):
    r, al, a, b = par["r"], par["alpha"], par["a"], par["b"]; x, tx, T = d.x.values, d.t_x.values, d["T"].values; H = days / 7
    p_alive = np.where(x > 0, 1 / (1 + (a / (b + np.maximum(x, 1) - 1)) * ((al + T) / (al + tx)) ** (r + x)), 1.0)
    return p_alive * (1 - ((al + T) / (al + T + H)) ** (r + x))      # 基準日から180日以内に1回以上購入する確率
