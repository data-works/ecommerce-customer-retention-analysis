"""10 ロジスティック回帰の改善の試行。BG/NBDの情報(再購入確率、購入日数、観測期間など)を特徴量に加えたとき、BG/NBD単独を上回るかを確認する。
特徴量の作成は各基準日で、その時点以前のデータだけを使う(BG/NBDのパラメータも基準日ごとに推定)。学習は05と同じ3時点、評価は2011-06-01。
前提: 05, 08 の実行後。結果: tables/24_lr_improvement.csv"""
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from config import *
from model_common import make_build, tf, cbt, fit_bgnbd, predict_bgnbd
df = pd.read_pickle(WORK_DIR / "clean.pkl"); p = df[df.is_product & df.CustomerID.notna()].copy(); p["CustomerID"] = p.CustomerID.astype(int)
build = make_build(p); pb = p[~p.is_cancel]

def feats(c):
    X, y, _ = build(c); d = cbt(pb, c, X.index); pr = predict_bgnbd(fit_bgnbd(d)[0], d)
    Z = tf(X).copy(); q = np.clip(pr, 1e-4, 1 - 1e-4)
    Z["logit_bgnbd"] = np.log(q / (1 - q)); Z["x_total"] = np.log1p(d.x.values); Z["tx_over_T"] = d.t_x.values / np.maximum(d["T"].values, 1e-9); Z["T_weeks"] = d["T"].values
    Z["recency_log"] = np.log1p(X.recency.values)
    return Z, y, pr
tr = [feats(c) for c in ["2010-06-01", "2010-09-01", "2010-12-01"]]; Zte, yte, pbg = feats("2011-06-01")
Ztr = pd.concat([a for a, _, _ in tr]); ytr = pd.concat([b for _, b, _ in tr])
base = list(tf(build("2011-06-01")[0]).columns)
sets = {"現行のロジスティック回帰(7特徴量)": base, "現行+BG/NBDの再購入確率": base + ["logit_bgnbd"],
        "現行+購入日数・観測期間・購入間隔の規則性": base + ["x_total", "tx_over_T", "T_weeks"], "現行+上記すべて": base + ["logit_bgnbd", "x_total", "tx_over_T", "T_weeks", "recency_log"]}
yv = yte.values; rows = [{"model": nm, "AUC": roc_auc_score(yv, make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Ztr[c], ytr).predict_proba(Zte[c])[:, 1])} for nm, c in sets.items()]
rows.append({"model": "BG/NBD(単独)", "AUC": roc_auc_score(yv, pbg)})
r = pd.DataFrame(rows); r.round(4).to_csv(TABLES / "24_lr_improvement.csv", index=False, encoding="utf-8-sig"); print(r.round(4).to_string())
