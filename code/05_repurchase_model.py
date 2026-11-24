"""05 再購入予測の比較対象: ロジスティック回帰・勾配ブースティング・単純ルール(基準日から180日以内に再び購入するか)。主モデルは 08_bgnbd.py。
検証は時系列で行う: 学習 = 2010-06-01 / 2010-09-01 / 2010-12-01 時点(いずれも予測期間が2011-05-31までに終わる)、
評価 = 2011-06-01 時点(予測期間 2011-06〜11)。学習に評価期間の情報は一切含まれない。
特徴量は基準日以前の直近180日の購買行動のみ。"""
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from config import *
from model_common import make_build, tf
TRAIN_CUTOFFS = ["2010-06-01", "2010-09-01", "2010-12-01"]; TEST_CUTOFF = "2011-06-01"
df = pd.read_pickle(WORK_DIR / "clean.pkl")
p = df[df.is_product & df.CustomerID.notna()].copy()
p["CustomerID"] = p.CustomerID.astype(int)

build = make_build(p)

parts = [build(c) for c in TRAIN_CUTOFFS]
Xtr = pd.concat([a for a, _, _ in parts], ignore_index=True); ytr = pd.concat([b for _, b, _ in parts], ignore_index=True)
X1, y1, _ = parts[0]                                   # 参考: 単一時点(2010-06-01)のみで学習した場合
Xte, yte, fte = build(TEST_CUTOFF)
print("train rows", len(Xtr), "base rate by cutoff", [round(b.mean(), 3) for _, b, _ in parts], "| test", len(Xte), round(yte.mean(), 3))

mk_lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
lr = mk_lr().fit(tf(Xtr), ytr); lr1 = mk_lr().fit(tf(X1), y1)
gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=.05, max_iter=200, random_state=0).fit(Xtr, ytr)
pl, pl1, pg = lr.predict_proba(tf(Xte))[:, 1], lr1.predict_proba(tf(Xte))[:, 1], gb.predict_proba(Xte)[:, 1]
rule = -Xte.recency.values
yv = yte.values

# ブートストラップ(評価データの顧客を再抽出)でAUCの95%区間を求める
rng = np.random.default_rng(0); B = 1000; n = len(yv); boot = {k: [] for k in ["rule", "lr", "lr1", "gb"]}
for _ in range(B):
    i = rng.integers(0, n, n)
    if yv[i].min() == yv[i].max(): continue
    for k, v in zip(boot, [rule, pl, pl1, pg]): boot[k].append(roc_auc_score(yv[i], v[i]))
ci = lambda a: tuple(np.percentile(a, [2.5, 97.5]))
diff = np.array(boot["lr"]) - np.array(boot["rule"])
names = {"rule": "ルール(最終購入が新しい順)", "lr1": "ロジスティック回帰(単一時点で学習)", "lr": "ロジスティック回帰(3時点を合わせて学習)", "gb": "勾配ブースティング(3時点)"}
auc = {"rule": roc_auc_score(yv, rule), "lr1": roc_auc_score(yv, pl1), "lr": roc_auc_score(yv, pl), "gb": roc_auc_score(yv, pg)}
res = pd.DataFrame([{"model": names[k], "AUC": auc[k], "CI95_low": ci(boot[k])[0], "CI95_high": ci(boot[k])[1]} for k in ["rule", "lr1", "lr", "gb"]])
res.round(4).to_csv(TABLES / "10_model_auc.csv", index=False, encoding="utf-8-sig"); print(res.round(3).to_string())
print("AUC差(LR−ルール) mean", diff.mean().round(4), "95%CI", np.round(ci(diff), 4), " P(diff<=0)", (diff <= 0).mean())

coef = pd.Series(lr[-1].coef_[0], index=Xtr.columns).sort_values()
coef.round(3).to_csv(TABLES / "11_logit_coefficients.csv", header=["coef_standardized"], encoding="utf-8-sig"); print(coef.round(3))

te = Xte.assign(prob=pl, actual=yte, future_purchase=fte)
te["decile"] = pd.qcut(te.prob.rank(method="first", ascending=False), 10, labels=range(1, 11)).astype(int)
dec = te.groupby("decile").agg(customers=("prob", "size"), pred=("prob", "mean"), actual=("actual", "mean"), future_purchase=("future_purchase", "sum"))
dec["purchase_share"] = dec.future_purchase / dec.future_purchase.sum()
dec.round(3).to_csv(TABLES / "12_decile.csv", encoding="utf-8-sig"); print(dec.round(3))
te.reset_index().rename(columns={"index": "CustomerID"}).to_csv(TABLES / "13_test_scores.csv", index=False, encoding="utf-8-sig")
# 上位20%(デシル1〜2の顧客数と同数)が占める評価期間の購入額: モデル / 単純ルール / 無作為
k20 = int(dec.customers.iloc[:2].sum()); tot = te.future_purchase.sum()
top20_rule = te.sort_values("recency").head(k20).future_purchase.sum() / tot
save_key({"top20_share_model": dec.purchase_share.iloc[:2].sum(), "top20_share_rule": top20_rule, "top20_share_random": k20 / len(te)})
save_key({"model_train_rows": len(Xtr), "model_test_rows": len(Xte), "test_base_rate": yte.mean(),
          **{f"train_base_rate_{c}": b.mean() for c, (_, b, _) in zip(TRAIN_CUTOFFS, parts)},
          "auc_rule": auc["rule"], "auc_lr": auc["lr"], "auc_lr_single": auc["lr1"], "auc_gb": auc["gb"],
          "auc_lr_ci_low": ci(boot["lr"])[0], "auc_lr_ci_high": ci(boot["lr"])[1],
          "auc_rule_ci_low": ci(boot["rule"])[0], "auc_rule_ci_high": ci(boot["rule"])[1],
          "auc_diff_mean": diff.mean(), "auc_diff_ci_low": ci(diff)[0], "auc_diff_ci_high": ci(diff)[1],
          "decile1_purchase_share": dec.purchase_share.iloc[0], "decile12_purchase_share": dec.purchase_share.iloc[:2].sum(),
          "decile1_actual": dec.actual.iloc[0], "decile10_actual": dec.actual.iloc[-1]})
