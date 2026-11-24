"""共通設定。パスは環境変数で上書き可能。"""
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))          # 元のxlsxを置くフォルダ
WORK_DIR = Path(os.environ.get("WORK_DIR", "work"))          # 中間ファイル
OUT_DIR = Path(os.environ.get("OUT_DIR", ".."))              # tables/ figures/ の親
TABLES = OUT_DIR / "tables"
FIGURES = OUT_DIR / "figures"
for p in (WORK_DIR, TABLES, FIGURES):
    p.mkdir(parents=True, exist_ok=True)

DATA_FILE = "online_retail_II.xlsx"                          # UCI配布の元ファイル(2シート構成)
SHEETS = ["Year 2009-2010", "Year 2010-2011"]               # 2009-12-01〜2010-12-09 / 2010-12-01〜2011-12-09


def load_raw():
    """元ファイルの2シートを読み込み、(2009-2010, 2010-2011) のDataFrameを返す。"""
    import pandas as pd
    d = pd.read_excel(DATA_DIR / DATA_FILE, sheet_name=SHEETS, dtype={"Invoice": str, "StockCode": str})
    return d[SHEETS[0]], d[SHEETS[1]]

SNAPSHOT = "2011-12-10"   # データ最終日(2011-12-09)の翌日。RFMの基準日

# 商品ではない StockCode(送料・手数料・調整・テスト等)
NON_PRODUCT = {"POST", "DOT", "M", "C2", "D", "S", "B", "BANK CHARGES",
               "ADJUST", "ADJUST2", "AMAZONFEE", "CRUK", "TEST001", "TEST002"}

def setup_plot():
    import matplotlib as mpl
    from matplotlib import font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    cand = ["Noto Sans CJK JP", "Yu Gothic", "Meiryo", "Hiragino Sans", "IPAexGothic"]
    mpl.rcParams["font.family"] = [c for c in cand if c in have][:1] or ["sans-serif"]
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["figure.dpi"] = 130
    mpl.rcParams["savefig.bbox"] = "tight"

# グラフ配色(単色=青、比較=青/橙、補助=グレー)
BLUE, ORANGE, AQUA, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#a8a7a2", "#0b0b0b"


def save_key(items):
    """報告書に載せる数値を tables/99_key_figures.csv に集約する(同じキーは上書き)。"""
    import pandas as pd
    f = TABLES / "99_key_figures.csv"
    cur = pd.read_csv(f, index_col="key")["value"].to_dict() if f.exists() else {}
    cur.update({k: float(v) for k, v in items.items()})
    pd.Series(cur, name="value").rename_axis("key").to_csv(f, encoding="utf-8-sig")

