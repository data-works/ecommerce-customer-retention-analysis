#!/bin/bash
# 実行順。事前に data/ に2つのxlsxを置く(環境変数 DATA_DIR / WORK_DIR / OUT_DIR で変更可)
# 必要ライブラリ: pandas, openpyxl, numpy, matplotlib, scikit-learn, scipy
set -e
for f in 01_load_clean 02_sales_overview 03_rfm 04_cohort 05_repurchase_model 08_bgnbd 10_lr_improvement 07_expected_value 09_sensitivity; do
  echo "== $f"; python3 $f.py
done
