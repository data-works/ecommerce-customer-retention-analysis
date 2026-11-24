"""09 前処理の感度分析。「誤発注の相殺」の条件(相殺する時間幅)と、完全重複行の除去の有無を変えたときに、
主要な数値(純売上、返品率、セグメント構成比など)がどの程度動くかを確認する。
基準(24時間)は 01_load_clean.py と同じ処理で、行数・純売上が一致することを確認してから比較する。
前提: 01〜03 の実行後(tables/99_key_figures.csv で基準値を照合する)。"""
import pandas as pd, numpy as np
from config import *

try:
    y1, y2 = pd.read_pickle(WORK_DIR / "y1.pkl"), pd.read_pickle(WORK_DIR / "y2.pkl")
except FileNotFoundError:
    y1, y2 = load_raw()

def clean(offset, dedup=True):
    cut = y2.InvoiceDate.min()
    df = pd.concat([y1[y1.InvoiceDate < cut], y2], ignore_index=True).rename(columns={"Customer ID": "CustomerID"})
    if dedup: df = df.drop_duplicates()
    df["is_cancel"] = df.Invoice.str.startswith("C")
    df = df[~df.Invoice.str.startswith("A")]; df = df[df.Price > 0]; df = df[df.is_cancel | (df.Quantity > 0)]
    sc = df.StockCode.str.upper(); df["is_product"] = ~(sc.isin(NON_PRODUCT) | sc.str.startswith("GIFT_"))
    df["Sales"] = df.Quantity * df.Price
    if offset is not None:
        d = df.reset_index(drop=True)
        buy = d[~d.is_cancel & d.CustomerID.notna()].assign(q=lambda x: x.Quantity, pidx=lambda x: x.index)
        can = d[d.is_cancel & d.CustomerID.notna()].assign(q=lambda x: -x.Quantity, cidx=lambda x: x.index)
        mt = pd.merge_asof(can.sort_values("InvoiceDate"),
                           buy[["CustomerID", "StockCode", "q", "InvoiceDate", "pidx"]].rename(columns={"InvoiceDate": "buy_dt"}).sort_values("buy_dt"),
                           left_on="InvoiceDate", right_on="buy_dt", by=["CustomerID", "StockCode", "q"],
                           tolerance=pd.Timedelta(offset), direction="backward").dropna(subset=["pidx"]).drop_duplicates("pidx")
        df = d[~d.index.isin(set(mt.cidx.astype(int)) | set(mt.pidx.astype(int)))]
        npairs = len(mt)
    else: npairs = 0
    df = df.copy(); df["YM"] = df.InvoiceDate.dt.to_period("M").astype(str)
    return df, npairs

def metrics(df, npairs):
    p = df[df.is_product]; buy, can = p[~p.is_cancel], p[p.is_cancel]
    gross, ret = buy.Sales.sum(), can.Sales.abs().sum()
    pc = p[p.CustomerID.notna()]; b = pc[~pc.is_cancel]
    snap = pd.Timestamp(SNAPSHOT)
    rfm = pd.DataFrame({"rec": (snap - b.groupby("CustomerID").InvoiceDate.max()).dt.days,
                        "f": b.groupby("CustomerID").Invoice.nunique(), "m": pc.groupby("CustomerID").Sales.sum()})
    rfm = rfm[rfm.m > 0].copy()
    rfm["R"] = pd.qcut(rfm.rec, 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F"] = pd.cut(rfm.f, [0, 1, 2, 4, 9, np.inf], labels=[1, 2, 3, 4, 5]).astype(int)
    def seg(r):
        if r.R >= 4 and r.F >= 4: return "優良顧客"
        if r.R >= 3 and r.F >= 3: return "安定顧客"
        if r.R >= 4: return "直近購入・低頻度"
        if r.R <= 2 and r.F >= 3: return "離脱リスク"
        if r.R <= 2: return "休眠"
        return "一般"
    rfm["seg"] = rfm.apply(seg, axis=1)
    sh = rfm.groupby("seg").m.sum() / rfm.m.sum(); cn = rfm.seg.value_counts()
    cs = rfm.m.sort_values(ascending=False)
    ym = p.groupby("YM").Sales.sum()
    jn = lambda y: p[(p.InvoiceDate >= f"{y}-01-01") & (p.InvoiceDate < f"{y}-12-01")].Sales.sum()
    return {"相殺した組数": npairs, "行数": len(df), "総売上(£)": gross, "返品額(£)": ret, "純売上(£)": gross - ret, "返品率": ret / gross,
            "顧客別売上(£)": rfm.m.sum(), "顧客数": len(rfm),
            "優良顧客の人数": cn["優良顧客"], "優良顧客の売上構成比": sh["優良顧客"],
            "離脱リスクの人数": cn["離脱リスク"], "離脱リスクの売上構成比": sh["離脱リスク"], "休眠の人数": cn["休眠"],
            "上位20%顧客の売上構成比": cs.head(int(len(cs) * .2)).sum() / cs.sum(),
            "2011-12の純売上(£)": ym.get("2011-12", np.nan), "1〜11月の前年比": jn(2011) / jn(2010) - 1}

cases = [("相殺なし", None, True), ("1時間", "1h", True), ("24時間(基準)", "1D", True), ("7日", "7D", True), ("30日", "30D", True),
         ("24時間・完全重複を除去しない", "1D", False)]
rows = {}
for nm, off, dd in cases:
    df, n = clean(off, dd); rows[nm] = metrics(df, n); print(nm, int(rows[nm]["行数"]), round(rows[nm]["純売上(£)"]))
res = pd.DataFrame(rows)
base = res["24時間(基準)"]
kf = pd.read_csv(TABLES / "99_key_figures.csv", index_col="key")["value"]
assert int(base["行数"]) == int(kf["n_rows_clean"]) and abs(base["純売上(£)"] - kf["net_total"]) < 1, "基準が01の結果と一致しない"
assert int(base["優良顧客の人数"]) == int(kf["seg_優良顧客_customers"])
res.T.to_csv(TABLES / "20_sensitivity.csv", encoding="utf-8-sig", float_format="%.4f")
pd.set_option("display.float_format", lambda v: f"{v:,.4f}" if abs(v) < 10 else f"{v:,.0f}")
print(res.to_string())
save_key({"sens_return_rate_none": res.loc["返品率", "相殺なし"], "sens_return_rate_1h": res.loc["返品率", "1時間"],
          "sens_return_rate_7d": res.loc["返品率", "7日"], "sens_return_rate_30d": res.loc["返品率", "30日"],
          "sens_net_none": res.loc["純売上(£)", "相殺なし"], "sens_net_30d": res.loc["純売上(£)", "30日"]})
