"""01 読み込み・結合・クレンジング。処理ごとの除外件数を cleaning_log.csv に残す。"""
import pandas as pd
from config import *

# --- 読み込み(Invoice/StockCodeは文字列で読む)
y1, y2 = load_raw()   # 元ファイルの2シート(Year 2009-2010 / Year 2010-2011)
log = []
def rec(step, n_before, n_after, note=""):
    log.append({"step": step, "rows_before": n_before, "rows_after": n_after,
                "rows_removed": n_before - n_after, "note": note})

# --- 2シートの重複期間(2010-12-01〜12-09)は後ろのシート側を採用
cut = y2.InvoiceDate.min()
df = pd.concat([y1[y1.InvoiceDate < cut], y2], ignore_index=True)
rec("2シート結合(重複期間除去)", len(y1) + len(y2), len(df), "2010-12-01〜12-09が両シートに収録されているため")

df = df.rename(columns={"Customer ID": "CustomerID"})
n = len(df); df = df.drop_duplicates(); rec("完全重複行の除去", n, len(df), "全列一致の行")

df["is_cancel"] = df.Invoice.str.startswith("C")
n = len(df); df = df[~df.Invoice.str.startswith("A")]; rec("不良債権調整(A伝票)除去", n, len(df))

n = len(df); df = df[df.Price > 0]; rec("単価0以下の除去", n, len(df), "無償提供・在庫調整等")
n = len(df); df = df[df.is_cancel | (df.Quantity > 0)]; rec("取消でない数量マイナス行の除去", n, len(df), "在庫調整・破損処理等。ただし該当行は直前の単価0以下の除去で既に除外済みのため0件")

sc = df.StockCode.str.upper()
df["is_product"] = ~(sc.isin(NON_PRODUCT) | sc.str.startswith("GIFT_"))
rec("(参考)商品以外コードのフラグ化", len(df), len(df),
    f"送料・手数料・ギフト券等 {int((~df.is_product).sum()):,}行にフラグを付与(除外はしない。商品分析時のみ除外)")

df["Sales"] = df.Quantity * df.Price

# --- 誤発注の相殺: 24時間以内に「同一顧客・同一商品・同一数量」で取消された注文は
#     返品ではなく入力ミス等の可能性が高く、売上・返品率を大きく歪めるため、購入側・取消側の両方を除外する
#     (例: 2011-12-09 の 80,995個/£168,470 の注文と直後の取消)
d = df.reset_index(drop=True)
buy = d[~d.is_cancel & d.CustomerID.notna()].assign(q=lambda x: x.Quantity, pidx=lambda x: x.index)
can = d[d.is_cancel & d.CustomerID.notna()].assign(q=lambda x: -x.Quantity, cidx=lambda x: x.index)
mt = pd.merge_asof(can.sort_values("InvoiceDate"),
                   buy[["CustomerID", "StockCode", "q", "InvoiceDate", "pidx"]].rename(columns={"InvoiceDate": "buy_dt"}).sort_values("buy_dt"),
                   left_on="InvoiceDate", right_on="buy_dt", by=["CustomerID", "StockCode", "q"],
                   tolerance=pd.Timedelta("1D"), direction="backward").dropna(subset=["pidx"])
mt = mt.drop_duplicates("pidx")
drop_idx = set(mt.cidx.astype(int)) | set(mt.pidx.astype(int))
n = len(d); d = d[~d.index.isin(drop_idx)]
rec("誤発注の相殺(24時間以内の同一内容取消)", n, len(d), f"購入{len(mt):,}行+取消{len(mt):,}行。うち金額上位は2011-12(£168,470)と2011-01(£77,184)")
df = d
df["YM"] = df.InvoiceDate.dt.to_period("M").astype(str)
df["CustomerID"] = df.CustomerID.astype("Int64")
df.to_pickle(WORK_DIR / "clean.pkl")
pd.DataFrame(log).to_csv(TABLES / "00_cleaning_log.csv", index=False, encoding="utf-8-sig")
print(pd.DataFrame(log).to_string())
print(df.shape, "顧客ID欠損率", round(df.CustomerID.isna().mean() * 100, 1), "%")
