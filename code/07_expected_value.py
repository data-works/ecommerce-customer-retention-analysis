"""07 評価時点(2011-06-01)での施策対象の絞り込み: 「過去に多く買った」と「今後買う」を分けて評価する。
- RFMセグメント(評価時点の購入履歴だけで再計算)ごとの、その後180日の実際の再購入率と購入額
- 並べ方の比較: ①BG/NBDの再購入確率 ②期待売上(BG/NBDの再購入確率 × Gamma-Gammaの1注文あたり金額) ③過去の純売上 ④最終購入が新しい順 ⑤(参考)ロジスティック回帰
  各方法で上位N%に連絡した場合に、評価期間(2011-06〜11)の購入額のうち何%を捕捉できるか
前提: 05, 08 の実行後(tables/13_test_scores.csv と 21_bgnbd_scores.csv を使う)。"""
import pandas as pd, numpy as np
from config import *

CUTOFF = pd.Timestamp("2011-06-01")
df = pd.read_pickle(WORK_DIR / "clean.pkl")
p = df[df.is_product & df.CustomerID.notna()].copy(); p["CustomerID"] = p.CustomerID.astype(int)
te = pd.read_csv(TABLES / "13_test_scores.csv").set_index("CustomerID").rename(columns={"prob": "prob_lr"})
bg = pd.read_csv(TABLES / "21_bgnbd_scores.csv").set_index("CustomerID")
te["prob"], te["ev"] = bg.p_bgnbd.reindex(te.index), bg.ev_bgnbd.reindex(te.index)          # 主モデル: BG/NBD(+Gamma-Gamma)
assert te.prob.notna().all()

# ---- 評価時点より前の購入履歴(全期間)
hist = p[p.InvoiceDate < CUTOFF]
hb = hist[~hist.is_cancel]
g = hb.groupby("CustomerID")
h = pd.DataFrame({"orders": g.Invoice.nunique(), "gross": g.Sales.sum()})
h["net"] = hist.groupby("CustomerID").Sales.sum()                      # 取消を差し引いた過去の純売上
h["aov"] = h.gross / h.orders                                           # 過去の平均購入額(1注文あたり)
te = te.join(h, how="left")
assert te.orders.notna().all()

# ---- 評価時点のRFMセグメント(03_rfm.py と同じ規則。Rは評価時点の最終購入日数の5分位)
te["R"] = pd.qcut(te.recency, 5, labels=[5, 4, 3, 2, 1]).astype(int)
te["F"] = pd.cut(te.orders, [0, 1, 2, 4, 9, np.inf], labels=[1, 2, 3, 4, 5]).astype(int)
def seg(r):
    if r.R >= 4 and r.F >= 4: return "優良顧客"
    if r.R >= 3 and r.F >= 3: return "安定顧客"
    if r.R >= 4:              return "直近購入・低頻度"
    if r.R <= 2 and r.F >= 3: return "離脱リスク"
    if r.R <= 2:              return "休眠"
    return "一般"
te["segment"] = te.apply(seg, axis=1)
order = ["優良顧客", "安定顧客", "直近購入・低頻度", "一般", "離脱リスク", "休眠"]
tot_f = te.future_purchase.sum()
s = te.groupby("segment").agg(customers=("prob", "size"), past_net_sales=("net", "sum"), past_net_mean=("net", "mean"),
        pred_prob=("prob", "mean"), repurchase_rate=("actual", "mean"), future_purchase=("future_purchase", "sum"),
        recency_mean=("recency", "mean")).reindex(order)
s["past_sales_share"] = s.past_net_sales / s.past_net_sales.sum()
s["future_share"] = s.future_purchase / tot_f
s["future_per_customer"] = s.future_purchase / s.customers
s["customer_share"] = s.customers / s.customers.sum()
s.round(4).to_csv(TABLES / "15_segments_at_cutoff.csv", encoding="utf-8-sig"); print(s.round(3).to_string())

# ---- 並べ方の比較(上位N%に連絡した場合に、評価期間の購入額の何%を捕捉できるか)
n = len(te)
scores = {"予測確率": te.prob, "期待売上(予測確率×1注文あたり金額)": te.ev, "過去の純売上": te.net, "最終購入が新しい順": -te.recency, "ロジスティック回帰の予測確率": te.prob_lr}
rows = []
for nm, sc in scores.items():
    srt = te.loc[sc.sort_values(ascending=False).index]
    for q in (0.05, 0.1, 0.2, 0.3):
        k = int(np.ceil(n * q)); top = srt.head(k)
        rows.append({"method": nm, "top_share": q, "customers": k, "captured_purchase_share": top.future_purchase.sum() / tot_f,
                     "repurchase_rate": top.actual.mean(), "future_purchase_per_customer": top.future_purchase.mean()})
cap = pd.DataFrame(rows)
cap.round(4).to_csv(TABLES / "16_capture_by_ranking.csv", index=False, encoding="utf-8-sig")
print(cap.pivot(index="method", columns="top_share", values="captured_purchase_share").round(3).to_string())
print("無作為: 捕捉率 = 顧客比率。全体の再購入率", round(te.actual.mean(), 3))

# 捕捉率の差の不確実性: 評価顧客を再抽出(1,000回)し、上位20%の捕捉率の差の95%区間を求める
rng = np.random.default_rng(0); B = 1000; k20 = int(np.ceil(n * 0.2))
fp = te.future_purchase.values; sc_arr = {"prob": te.prob.values, "ev": te.ev.values, "past": te.net.values, "rule": -te.recency.values, "lr": te.prob_lr.values}
def cap20(idx, v):
    o = idx[np.argsort(-v[idx], kind="stable")][: int(np.ceil(len(idx) * 0.2))]; return fp[o].sum() / fp[idx].sum()
bs = {k: [] for k in sc_arr}
for _ in range(B):
    i = rng.integers(0, n, n)
    for k_, v in sc_arr.items(): bs[k_].append(cap20(i, v))
bs = {k_: np.array(v) for k_, v in bs.items()}
ci = lambda a: tuple(np.percentile(a, [2.5, 97.5]))
diffs = {"予測確率−過去の純売上": bs["prob"] - bs["past"], "期待売上−過去の純売上": bs["ev"] - bs["past"],
         "予測確率−最終購入が新しい順": bs["prob"] - bs["rule"], "期待売上−予測確率": bs["ev"] - bs["prob"], "予測確率−ロジスティック回帰": bs["prob"] - bs["lr"]}
dd = pd.DataFrame([{"comparison": k_, "diff_mean": v.mean(), "ci_low": ci(v)[0], "ci_high": ci(v)[1]} for k_, v in diffs.items()])
dd.round(4).to_csv(TABLES / "16b_capture20_diff_ci.csv", index=False, encoding="utf-8-sig"); print(dd.round(3).to_string())

# 離脱リスク(評価時点)と、その中での並べ方
rk = te[te.segment == "離脱リスク"]
rk_rows = []
for nm, sc in {"過去の純売上": rk.net, "予測確率": rk.prob, "期待売上": rk.ev}.items():
    srt = rk.loc[sc.sort_values(ascending=False).index]
    for k in (50, 100):
        t = srt.head(k); rk_rows.append({"method": nm, "top_n": k, "repurchase_rate": t.actual.mean(),
                                         "future_purchase": t.future_purchase.sum(), "future_purchase_per_customer": t.future_purchase.mean()})
rkd = pd.DataFrame(rk_rows); rkd.round(4).to_csv(TABLES / "17_risk_segment_ranking.csv", index=False, encoding="utf-8-sig")
print(f"離脱リスク {len(rk)}人 全体の再購入率 {rk.actual.mean():.3f}"); print(rkd.round(3).to_string())

k = {"seg_cut_risk_customers": len(rk), "seg_cut_risk_repurchase": rk.actual.mean(), "seg_cut_risk_future_per_customer": rk.future_purchase.mean()}
for q in (0.1, 0.2, 0.3):
    for nm, key in [("予測確率", "prob"), ("期待売上(予測確率×1注文あたり金額)", "ev"), ("過去の純売上", "past"), ("最終購入が新しい順", "rule"), ("ロジスティック回帰の予測確率", "lr")]:
        k[f"capture_{key}_{int(q*100)}"] = cap[(cap.method == nm) & (cap.top_share == q)].captured_purchase_share.iloc[0]
for sg in order:
    k[f"cut_{sg}_repurchase"] = s.loc[sg, "repurchase_rate"]; k[f"cut_{sg}_future_share"] = s.loc[sg, "future_share"]
    k[f"cut_{sg}_customers"] = s.loc[sg, "customers"]; k[f"cut_{sg}_per_customer"] = s.loc[sg, "future_per_customer"]
save_key(k)
