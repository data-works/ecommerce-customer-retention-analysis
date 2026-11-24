"""03 RFM分析。基準日 2011-12-10。対象: 顧客IDあり・期間通算の純売上>0の顧客。
R=最終購入からの日数(5分位), F=購入伝票数(固定しきい値), M=純売上(5分位)。"""
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from config import *
setup_plot()
df = pd.read_pickle(WORK_DIR / "clean.pkl")
p = df[df.is_product & df.CustomerID.notna()]
buy = p[~p.is_cancel]
snap = pd.Timestamp(SNAPSHOT)

rfm = pd.DataFrame({
    "recency_days": (snap - buy.groupby("CustomerID").InvoiceDate.max()).dt.days,
    "frequency": buy.groupby("CustomerID").Invoice.nunique(),
    "monetary": p.groupby("CustomerID").Sales.sum(),           # 取消を差し引いた純売上
    "first_purchase": buy.groupby("CustomerID").InvoiceDate.min().dt.date,
    "country": buy.groupby("CustomerID").Country.agg(lambda s: s.mode().iat[0]),
})
rfm = rfm[rfm.monetary > 0].copy()

# スコア(5が良い)
rfm["R"] = pd.qcut(rfm.recency_days, 5, labels=[5, 4, 3, 2, 1]).astype(int)
rfm["F"] = pd.cut(rfm.frequency, [0, 1, 2, 4, 9, np.inf], labels=[1, 2, 3, 4, 5]).astype(int)
rfm["M"] = pd.qcut(rfm.monetary.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)

def seg(r):
    if r.R >= 4 and r.F >= 4: return "優良顧客"
    if r.R >= 3 and r.F >= 3: return "安定顧客"
    if r.R >= 4:              return "直近購入・低頻度"
    if r.R <= 2 and r.F >= 3: return "離脱リスク"
    if r.R <= 2:              return "休眠"
    return "一般"
rfm["segment"] = rfm.apply(seg, axis=1)
order = ["優良顧客", "安定顧客", "直近購入・低頻度", "一般", "離脱リスク", "休眠"]

edges = pd.qcut(rfm.recency_days, 5, retbins=True)[1]      # Rスコアの境界(日数)
pd.DataFrame({"R_score": [5, 4, 3, 2, 1], "recency_days_over": edges[:-1], "recency_days_up_to": edges[1:]}
             ).round(1).to_csv(TABLES / "05b_rfm_recency_thresholds.csv", index=False, encoding="utf-8-sig")
s = rfm.groupby("segment").agg(customers=("R", "size"), net_sales=("monetary", "sum"),
        recency_mean=("recency_days", "mean"), frequency_mean=("frequency", "mean"),
        monetary_mean=("monetary", "mean"), monetary_median=("monetary", "median")).reindex(order)
s["customer_share"] = s.customers / s.customers.sum()
s["sales_share"] = s.net_sales / s.net_sales.sum()
s.round(3).to_csv(TABLES / "05_rfm_segments.csv", encoding="utf-8-sig")
rfm.reset_index().to_csv(TABLES / "06_rfm_customers.csv", index=False, encoding="utf-8-sig")
save_key({"rfm_customers": len(rfm), "repeat_customer_rate": (rfm.frequency >= 2).mean(),
          **{f"seg_{k}_customer_share": s.loc[k, "customer_share"] for k in order},
          **{f"seg_{k}_sales_share": s.loc[k, "sales_share"] for k in order},
          **{f"seg_{k}_customers": s.loc[k, "customers"] for k in order},
          **{f"seg_{k}_net_sales": s.loc[k, "net_sales"] for k in order},
          **{f"seg_{k}_recency_mean": s.loc[k, "recency_mean"] for k in order},
          **{f"seg_{k}_frequency_mean": s.loc[k, "frequency_mean"] for k in order}})
print(len(rfm), "customers"); print(s.round(3).to_string())
print("F score dist", rfm.F.value_counts().sort_index().to_dict())
print(rfm[["recency_days", "frequency", "monetary"]].describe().round(1))

# 図: 顧客構成比と純売上構成比
fig, ax = plt.subplots(figsize=(9, 4.2)); x = np.arange(len(order)); w = .38
ax.bar(x - w/2, s.customer_share * 100, w, color=GRAY, label="顧客数の構成比")
ax.bar(x + w/2, s.sales_share * 100, w, color=BLUE, label="純売上の構成比")
for i, k in enumerate(order):
    ax.text(i - w/2, s.customer_share[k] * 100 + .8, f"{s.customer_share[k]*100:.0f}%", ha="center", fontsize=9)
    ax.text(i + w/2, s.sales_share[k] * 100 + .8, f"{s.sales_share[k]*100:.0f}%", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(order); ax.set_ylabel("%"); ax.legend(frameon=False)
ax.set_title("RFMセグメント別 顧客数と純売上の構成比", loc="left", fontsize=12); ax.grid(axis="y", alpha=.25)
fig.savefig(FIGURES / "04_rfm_segments.png"); plt.close()
