"""02 売上の全体像: 月次推移・前年比・国別・顧客集中度(パレート)・取消(返品)。
売上定義: 純売上 = 購入明細(商品コードのみ) − 取消明細。通貨はGBP。"""
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from config import *
setup_plot()
df = pd.read_pickle(WORK_DIR / "clean.pkl")
p = df[df.is_product]
buy, can = p[~p.is_cancel], p[p.is_cancel]

# ---- 月次(2011-12は12/9までの部分月なので図では別扱い)
m = pd.DataFrame({"gross": buy.groupby("YM").Sales.sum(), "returns": can.groupby("YM").Sales.sum().abs(),
                  "orders": buy.groupby("YM").Invoice.nunique(),
                  "customers": buy.groupby("YM").CustomerID.nunique()}).fillna(0)
m["net"] = m.gross - m.returns
m["return_rate"] = m.returns / m.gross
m["aov"] = m.net / m.orders                  # 1注文あたり純売上
m.round(3).to_csv(TABLES / "01_monthly_sales.csv", encoding="utf-8-sig")

# ---- 前年同期比(1〜11月)
# 繁忙期 = 9〜11月。年間(1〜12月)の純売上に占める割合(2010年のみ。2011年は12月が部分月のため算出しない)
peak = lambda y: m.loc[[f"{y}-09", f"{y}-10", f"{y}-11"], "net"].sum() / m.loc[[f"{y}-{i:02d}" for i in range(1, 13)], "net"].sum()
print("2010 Sep-Nov share of 2010 net", peak(2010))

# ---- 国別
c = buy.groupby("Country").agg(gross=("Sales", "sum"), customers=("CustomerID", "nunique"), orders=("Invoice", "nunique"))
c["returns"] = can.groupby("Country").Sales.sum().abs().reindex(c.index).fillna(0)
c["sales"] = c.gross - c.returns                       # 純売上(期間通算)
c["share"] = c.sales / c.sales.sum(); c["aov"] = c.sales / c.orders
c.sort_values("sales", ascending=False).round(3).to_csv(TABLES / "02_country.csv", encoding="utf-8-sig")
uk = c.loc["United Kingdom", "share"]

# ---- 顧客集中度(顧客IDあり・純売上>0)
cust = (p.dropna(subset=["CustomerID"]).groupby("CustomerID").Sales.sum())
cust = cust[cust > 0].sort_values(ascending=False)
cum = cust.cumsum() / cust.sum(); x = np.arange(1, len(cust) + 1) / len(cust)
conc = {f"上位{int(k*100)}%": cust.head(int(len(cust) * k)).sum() / cust.sum() for k in (.01, .1, .2)}
print("集中度", conc, "顧客数", len(cust))
pd.Series(conc).round(4).to_csv(TABLES / "03_concentration.csv", header=["share_of_net_sales"], encoding="utf-8-sig")

# ---- 1〜11月の前年同期比を「注文数 × 1注文あたり純売上」に分解
def jn(y):
    lo, hi = f"{y}-01-01", f"{y}-12-01"
    b = buy[(buy.InvoiceDate >= lo) & (buy.InvoiceDate < hi)]; r = can[(can.InvoiceDate >= lo) & (can.InvoiceDate < hi)]
    return pd.Series({"gross_sales": b.Sales.sum(), "net_sales": b.Sales.sum() + r.Sales.sum(),
                      "orders": b.Invoice.nunique(), "active_customers": b.CustomerID.nunique()})
yy = pd.DataFrame({"2010": jn(2010), "2011": jn(2011)}).T
yy["aov"] = yy.net_sales / yy.orders                      # 1注文あたり純売上
yy.loc["YoY"] = yy.loc["2011"] / yy.loc["2010"] - 1
yy.round(4).to_csv(TABLES / "14_yoy_jan_nov.csv", encoding="utf-8-sig"); print(yy.round(4))
# ---- 商品(上位) ※商品コード単位
top = p.groupby(["StockCode"]).agg(desc=("Description", lambda s: s.mode().iat[0] if s.notna().any() else ""),
                                    sales=("Sales", "sum"), qty=("Quantity", "sum"))      # 純売上(購入−取消)ベース
top["share"] = top.sales / top.sales.sum(); top = top.nlargest(10, "sales")
nonprod = df[~df.is_product & ~df.is_cancel].Sales.sum() / df[~df.is_cancel].Sales.sum()
save_key({"gross_total": buy.Sales.sum(), "returns_total": can.Sales.abs().sum(), "net_total": buy.Sales.sum() + can.Sales.sum(),
          "return_rate_total": can.Sales.abs().sum() / buy.Sales.sum(), "nonproduct_share_of_sales": nonprod,
          "uk_share": uk, "eire_customers": c.loc["EIRE", "customers"], "eire_share": c.loc["EIRE", "share"],
          "yoy_gross": yy.loc["YoY", "gross_sales"], "yoy_net": yy.loc["YoY", "net_sales"], "yoy_orders": yy.loc["YoY", "orders"],
          "yoy_aov": yy.loc["YoY", "aov"], "yoy_active_customers": yy.loc["YoY", "active_customers"],
          "peak_sep_nov_2010_share_of_net": peak(2010), "top10_products_share": top.share.sum(), "conc_top1": conc["上位1%"], "conc_top10": conc["上位10%"], "conc_top20": conc["上位20%"],
          "n_customers_positive_net": len(cust), "n_rows_clean": len(df)})

# ---- 商品(上位) ※商品コード単位
top.round({"sales": 1, "qty": 1, "share": 4}).to_csv(TABLES / "04_top_products.csv", encoding="utf-8-sig")

# ======== 図1: 月次純売上(部分月は薄色)
fig, ax = plt.subplots(figsize=(10, 4.2))
idx = list(m.index); cols = [BLUE] * len(idx); cols[-1] = "#9dc0ee"
ax.bar(idx, m.net / 1e3, color=cols, width=.7)
ax.set_xticks(range(0, len(idx), 3)); ax.set_xticklabels(idx[::3], rotation=0)
ax.set_ylabel("純売上(千GBP)"); ax.set_title("月次純売上の推移(2011-12は12/9までの部分月)", loc="left", fontsize=12)
for k in ("2010-11", "2011-11"):
    ax.annotate(f"{m.loc[k,'net']/1e3:,.0f}", (idx.index(k), m.loc[k, "net"] / 1e3), ha="center", va="bottom", fontsize=9)
ax.grid(axis="y", alpha=.25); fig.savefig(FIGURES / "01_monthly_net_sales.png"); plt.close()

# ======== 図2: 顧客集中度(ローレンツ/パレート)
fig, ax = plt.subplots(figsize=(5.6, 4.6))
ax.plot(np.r_[0, x] * 100, np.r_[0, cum.values] * 100, color=BLUE, lw=2)
ax.plot([0, 100], [0, 100], color=GRAY, lw=1, ls="--")
for k in (.2,):
    ax.scatter([k * 100], [conc["上位20%"] * 100], color=ORANGE, zorder=3)
    ax.annotate(f"上位20%の顧客で\n純売上の{conc['上位20%']*100:.0f}%", (k * 100, conc["上位20%"] * 100), xytext=(35, 55), fontsize=10,
                arrowprops=dict(arrowstyle="-", color=GRAY))
ax.set_xlabel("顧客の累積比率(純売上の多い順, %)"); ax.set_ylabel("純売上の累積比率(%)")
ax.set_title("顧客別純売上の集中度", loc="left", fontsize=12); ax.grid(alpha=.25)
fig.savefig(FIGURES / "02_customer_concentration.png"); plt.close()

# ======== 図3: 国別構成(英国 vs それ以外 上位)
cc = c.sort_values("sales", ascending=False).head(8)
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(cc.index[::-1], cc.sales[::-1] / 1e3, color=[BLUE if i == "United Kingdom" else GRAY for i in cc.index[::-1]])
ax.set_xlabel("純売上(千GBP, 期間通算)"); ax.set_title(f"国別売上上位(英国が全体の{uk*100:.0f}%)", loc="left", fontsize=12)
ax.grid(axis="x", alpha=.25); fig.savefig(FIGURES / "03_country_sales.png"); plt.close()

print(m[["gross", "returns", "net", "return_rate", "orders", "customers", "aov"]].round(3).to_string())
print(c.sort_values("sales", ascending=False).head(6).round(3))
print(top.round(0))
print("全体 返品率", can.Sales.abs().sum() / buy.Sales.sum())
