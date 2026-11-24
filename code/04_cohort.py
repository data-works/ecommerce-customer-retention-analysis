"""04 コホート分析(初回購入月別のリテンション)。
- 月次リテンション: 初回購入月を M+0 とし、M+k 月に購入した顧客の割合。2011-12は部分月のため対象外。
- 90日以内再購入率: 初回購入日の翌日以降90日以内に再購入した顧客の割合(暦月の季節性の影響を受けにくい)。
注意: データが2009-12開始のため、2009-12コホートは既存顧客を含み、2010年のコホートにも「過去に購入歴があるが
観測期間内で初めて見えた顧客」が混在する(左側打ち切り)。クリーンな新規顧客は2010-12以降の初回購入者(過去12か月に購入なし)。"""
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from config import *
setup_plot()
df = pd.read_pickle(WORK_DIR / "clean.pkl")
p = df[df.is_product & df.CustomerID.notna()]
valid = p.groupby("CustomerID").Sales.sum(); valid = valid[valid > 0].index
buy_all = p[(~p.is_cancel) & p.CustomerID.isin(valid)].copy()
buy_all["m"] = buy_all.InvoiceDate.dt.to_period("M")
buy = buy_all[buy_all.m <= "2011-11"]

# ---------- 月次リテンション
first = buy.groupby("CustomerID").m.min().rename("cohort")
act = buy[["CustomerID", "m"]].drop_duplicates().join(first, on="CustomerID")
act["k"] = (act.m - act.cohort).apply(lambda d: d.n)
size = first.value_counts().sort_index()
mat = act.groupby(["cohort", "k"]).CustomerID.nunique().unstack().reindex(columns=range(0, 24))
ret = mat.div(size, axis=0)
last = pd.Period("2011-11", "M")
for c in ret.index: ret.loc[c, [k for k in ret.columns if c + k > last]] = np.nan
ret.round(4).to_csv(TABLES / "07_cohort_retention.csv", encoding="utf-8-sig")
size.rename("cohort_size").to_csv(TABLES / "08_cohort_size.csv", encoding="utf-8-sig")

def pooled(cohorts, k):
    """コホート人数で重み付けしたリテンション(単純平均ではなく合算)"""
    cs = [c for c in cohorts if not np.isnan(ret.loc[c, k])]
    return (mat.loc[cs, k].sum() / size[cs].sum(), int(size[cs].sum())) if cs else (np.nan, 0)
sets = {"2009-12を除く全コホート": [c for c in ret.index if c > pd.Period("2009-12", "M")],
        "2010-12以降の新規顧客": [c for c in ret.index if c >= pd.Period("2010-12", "M")]}
rows = [{"cohort_set": nm, "month": f"M+{k}", "retention": pooled(cs, k)[0], "customers": pooled(cs, k)[1]}
        for nm, cs in sets.items() for k in (1, 3, 6, 12)]
summ = pd.DataFrame(rows); summ.round(4).to_csv(TABLES / "09_cohort_summary.csv", index=False, encoding="utf-8-sig")
print(summ.round(3).to_string()); print(size.to_string())

# ---------- 90日以内再購入率
buy_all["d"] = buy_all.InvoiceDate.dt.normalize()
f0 = buy_all.groupby("CustomerID").d.min().rename("first_d")
later = buy_all.join(f0, on="CustomerID"); later = later[later.d > later.first_d]
gap = (later.d - later.first_d).dt.days.groupby(later.CustomerID).min()
c = pd.DataFrame({"first_d": f0}); c["gap"] = gap
c["rep90"] = (c.gap <= 90).astype(int); c["ym"] = c.first_d.dt.to_period("M")
obs = c[c.first_d <= pd.Timestamp("2011-12-09") - pd.Timedelta(days=90)]       # 90日間を観測できる顧客のみ
byym = obs.groupby("ym").agg(customers=("rep90", "size"), repeat90=("rep90", "mean"))
byym.round(4).to_csv(TABLES / "09b_repeat90_by_first_month.csv", encoding="utf-8-sig"); print(byym.round(3).T.to_string())
clean_new = obs[obs.ym >= "2010-12"]; excl = obs[obs.ym > "2009-12"]
aut = clean_new.ym.dt.month.isin([9, 10, 11])
print("90日再購入率 新規(2010-12以降)", clean_new.rep90.mean(), len(clean_new), "| 2009-12除く全体", excl.rep90.mean(), len(excl))
save_key({"repeat90_new_2010_12_on": clean_new.rep90.mean(), "repeat90_new_n": len(clean_new),
          "repeat90_all_excl_2009_12": excl.rep90.mean(), "repeat90_all_n": len(excl),
          "cohort_ret_m1_new": rows[4]["retention"], "cohort_ret_m3_new": rows[5]["retention"], "cohort_ret_m6_new": rows[6]["retention"],
          "cohort_ret_m1_all": rows[0]["retention"], "cohort_ret_m3_all": rows[1]["retention"],
          "cohort_ret_m6_all": rows[2]["retention"], "cohort_ret_m12_all": rows[3]["retention"],
          "cohort_size_2009_12": size.iloc[0], "cohort_size_2010_h1_mean": size["2010-01":"2010-06"].mean(),
          "cohort_size_2011_h1_mean": size["2011-01":"2011-06"].mean()})

# ---------- 図: ヒートマップ(M+1〜M+12)
d = ret.iloc[:, 1:13] * 100
fig, ax = plt.subplots(figsize=(11, 7.4))
im = ax.imshow(d.values, cmap="Blues", vmin=0, vmax=60, aspect="auto")
ax.set_xticks(range(12)); ax.set_xticklabels([f"+{k}" for k in range(1, 13)])
ax.set_yticks(range(len(d))); ax.set_yticklabels([f"{cc} (n={size[cc]})" + ("*" if cc <= pd.Period("2010-11", "M") else "") for cc in d.index], fontsize=8)
for i in range(d.shape[0]):
    for j in range(12):
        v = d.values[i, j]
        if not np.isnan(v): ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white" if v > 35 else INK)
ax.set_xlabel("初回購入からの経過月数"); ax.set_ylabel("初回購入月(コホート)")
ax.set_title("コホート別 月次リテンション率(%, その月に購入した顧客の割合)", loc="left", fontsize=12)
ax.spines[:].set_visible(False); fig.colorbar(im, ax=ax, shrink=.6, label="%")
fig.text(.01, .005, "*2009-12は既存顧客を含む。2010-11以前のコホートは、過去に購入歴のある顧客が「新規」として混在している可能性がある。", fontsize=8, color="#52514e")
fig.savefig(FIGURES / "05_cohort_retention.png"); plt.close()
