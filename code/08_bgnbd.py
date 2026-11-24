"""08 再購入予測の主モデル: BG/NBD(+Gamma-Gamma)。
BG/NBD(Fader, Hardie & Lee, 2005)は、顧客ごとの購入回数・最終購入時期・観測期間の3つから、購入頻度と離脱確率の個人差を確率分布で表す。
購入額は Gamma-Gamma(Fader & Hardie, 2013)で推定する。実装は scipy による最尤推定。
- パラメータは基準日以前の購入履歴だけで推定する(評価期間の情報は使わない。学習用の別時点は不要)
- 評価: 05と同じ基準日(2011-06-01)・同じ顧客・同じ目的変数(以降180日に再購入したか)。05のロジスティック回帰・単純ルールと比較する
- 安定性: 基準日を5時点に変えて、単純ルールとのAUC差を確認する
前提: 05 の実行後(tables/13_test_scores.csv を使う)。"""
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import gammaln
from sklearn.metrics import roc_auc_score
from config import *
setup_plot()
from model_common import cbt, fit_bgnbd, predict_bgnbd

CUTOFF = pd.Timestamp("2011-06-01")
df = pd.read_pickle(WORK_DIR / "clean.pkl")
p = df[df.is_product & df.CustomerID.notna() & ~df.is_cancel].copy(); p["CustomerID"] = p.CustomerID.astype(int)
te = pd.read_csv(TABLES / "13_test_scores.csv").set_index("CustomerID").rename(columns={"prob": "prob_lr"})

# ---- BG/NBD
d = cbt(p, CUTOFF, te.index)
par, best = fit_bgnbd(d)
print("BG/NBD", {k: round(v, 3) for k, v in par.items()}, f"NLL={best.fun:.1f} converged={best.success}")
te["p_bgnbd"] = predict_bgnbd(par, d)

# ---- Gamma-Gamma(1注文あたり金額)。2回以上購入した顧客で推定し、全顧客に適用
hist = p[p.InvoiceDate < CUTOFF]
inv = hist.groupby(["CustomerID", "Invoice"]).Sales.sum().reset_index(); inv = inv[inv.Sales > 0]
gm = inv.groupby("CustomerID").Sales.agg(["mean", "size"]).reindex(te.index)
fit = gm[gm["size"] >= 2].dropna()
def nll_gg(th):
    pp, q, v = np.exp(th); xx, m = fit["size"].values.astype(float), fit["mean"].values
    ll = (gammaln(pp * xx + q) - gammaln(pp * xx) - gammaln(q) + q * np.log(v) + (pp * xx - 1) * np.log(m) + pp * xx * np.log(xx) - (pp * xx + q) * np.log(v + xx * m))
    return -ll.sum()
rg = minimize(nll_gg, [np.log(2.), np.log(3.), np.log(300.)], method="Nelder-Mead", options={"maxiter": 4000})
pg, qg, vg = np.exp(rg.x); print(f"Gamma-Gamma p={pg:.3f} q={qg:.3f} v={vg:.1f} converged={rg.success} 相関(回数,金額)={np.corrcoef(fit['size'], fit['mean'])[0,1]:.3f}")
xg, mg = gm["size"].values.astype(float), gm["mean"].values
exp_m = (pg * vg + pg * xg * mg) / (pg * xg + qg - 1)
te["gg_aov"] = np.where(np.isnan(exp_m), np.nanmean(exp_m), exp_m)          # 購入が1回のみの顧客は平均値で補う
te["ev_bgnbd"] = te.p_bgnbd * te.gg_aov
te[["p_bgnbd", "gg_aov", "ev_bgnbd", "actual", "future_purchase", "recency"]].to_csv(TABLES / "21_bgnbd_scores.csv", encoding="utf-8-sig")

# ---- AUC(ブートストラップ1,000回)
y = te.actual.values; n = len(te)
arr = {"rule": -te.recency.values, "bg": te.p_bgnbd.values, "lr": te.prob_lr.values}
rng = np.random.default_rng(0); B = 1000; bs = {k: [] for k in arr}
for _ in range(B):
    i = rng.integers(0, n, n)
    for k, v in arr.items(): bs[k].append(roc_auc_score(y[i], v[i]))
bs = {k: np.array(v) for k, v in bs.items()}; ci = lambda a: tuple(np.percentile(a, [2.5, 97.5]))
nm = {"rule": "単純ルール(最終購入が新しい順)", "bg": "BG/NBD", "lr": "ロジスティック回帰(3時点を合わせて学習)"}
res = pd.DataFrame([{"model": nm[k], "AUC": roc_auc_score(y, v), "CI95_low": ci(bs[k])[0], "CI95_high": ci(bs[k])[1]} for k, v in arr.items()])
res.round(4).to_csv(TABLES / "18_bgnbd_auc.csv", index=False, encoding="utf-8-sig"); print(res.round(4).to_string())
d_bg_rule, d_bg_lr = bs["bg"] - bs["rule"], bs["bg"] - bs["lr"]
print("AUC差 BG/NBD−ルール", d_bg_rule.mean().round(4), np.round(ci(d_bg_rule), 4), "| BG/NBD−LR", d_bg_lr.mean().round(4), np.round(ci(d_bg_lr), 4))

# ---- デシル別(予測確率の高い順)
te["decile"] = pd.qcut(te.p_bgnbd.rank(method="first", ascending=False), 10, labels=range(1, 11)).astype(int)
dec = te.groupby("decile").agg(customers=("p_bgnbd", "size"), pred=("p_bgnbd", "mean"), actual=("actual", "mean"), future_purchase=("future_purchase", "sum"))
dec["purchase_share"] = dec.future_purchase / dec.future_purchase.sum()
dec.round(4).to_csv(TABLES / "22_bgnbd_decile.csv", encoding="utf-8-sig"); print(dec.round(3).T.to_string())
k20 = int(dec.customers.iloc[:2].sum()); tot = te.future_purchase.sum()
top20_rule = te.sort_values("recency").head(k20).future_purchase.sum() / tot

# ---- 基準日を変えた安定性(各基準日で、その時点以前のデータだけでパラメータを推定)
rows = []
for c in ["2010-06-01", "2010-09-01", "2010-12-01", "2011-03-01", "2011-06-01"]:
    cc = pd.Timestamp(c); dd = cbt(p, cc); pr = predict_bgnbd(fit_bgnbd(dd)[0], dd)
    fut = p[(p.InvoiceDate >= cc) & (p.InvoiceDate < cc + pd.Timedelta(days=180))].CustomerID.unique()
    yy = dd.index.isin(fut).astype(int); rec = (cc - p[p.InvoiceDate < cc].groupby("CustomerID").InvoiceDate.max().loc[dd.index]).dt.days.values
    r_ = np.random.default_rng(1); dif = []
    for _ in range(500):
        i = r_.integers(0, len(yy), len(yy)); dif.append(roc_auc_score(yy[i], pr[i]) - roc_auc_score(yy[i], -rec[i]))
    rows.append({"cutoff": c, "customers": len(yy), "repurchase_rate": yy.mean(), "auc_bgnbd": roc_auc_score(yy, pr), "auc_rule": roc_auc_score(yy, -rec),
                 "diff": roc_auc_score(yy, pr) - roc_auc_score(yy, -rec), "diff_ci_low": np.percentile(dif, 2.5), "diff_ci_high": np.percentile(dif, 97.5)})
stab = pd.DataFrame(rows); stab.round(4).to_csv(TABLES / "23_bgnbd_stability.csv", index=False, encoding="utf-8-sig"); print(stab.round(4).to_string())

save_key({"bgnbd_r": par["r"], "bgnbd_alpha": par["alpha"], "bgnbd_a": par["a"], "bgnbd_b": par["b"],
          "auc_bgnbd": res.AUC[1], "auc_bgnbd_ci_low": res.CI95_low[1], "auc_bgnbd_ci_high": res.CI95_high[1],
          "auc_diff_bgnbd_rule": d_bg_rule.mean(), "auc_diff_bgnbd_rule_ci_low": ci(d_bg_rule)[0], "auc_diff_bgnbd_rule_ci_high": ci(d_bg_rule)[1],
          "auc_diff_bgnbd_lr": d_bg_lr.mean(), "auc_diff_bgnbd_lr_ci_low": ci(d_bg_lr)[0], "auc_diff_bgnbd_lr_ci_high": ci(d_bg_lr)[1],
          "bg_decile1_actual": dec.actual.iloc[0], "bg_decile10_actual": dec.actual.iloc[-1], "bg_decile1_purchase_share": dec.purchase_share.iloc[0],
          "bg_top20_share": dec.purchase_share.iloc[:2].sum(), "bg_top20_share_rule": top20_rule, "bg_top20_share_random": k20 / len(te)})

fig, ax = plt.subplots(figsize=(8, 4.2))
ax.bar(dec.index, dec.actual * 100, color=BLUE, width=.7, label="実際の再購入率")
ax.plot(dec.index, dec.pred * 100, color=ORANGE, marker="o", ms=5, lw=2, label="予測確率の平均")
ax.axhline(te.actual.mean() * 100, color=GRAY, ls="--", lw=1); ax.text(10.4, te.actual.mean() * 100 + 1.5, f"全体 {te.actual.mean()*100:.0f}%", ha="right", fontsize=9, color="#52514e")
ax.set_xticks(range(1, 11)); ax.set_xlabel("予測確率のデシル(1=再購入しやすい)"); ax.set_ylabel("%"); ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, .9))
ax.set_title("BG/NBDの予測と実績(評価期間 2011-06〜11)", loc="left", fontsize=12); ax.grid(axis="y", alpha=.25)
fig.savefig(FIGURES / "06_repurchase_decile.png"); plt.close()
