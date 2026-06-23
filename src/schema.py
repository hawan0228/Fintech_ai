# =========================
# Identifier columns
# =========================

STOCK_ID_COL = "證券代碼"
STOCK_NAME_COL = "簡稱"
YEARMONTH_COL = "年月"
YEAR_COL = "year"

ID_COLUMNS = [
    STOCK_ID_COL,
    STOCK_NAME_COL,
    YEARMONTH_COL,
    YEAR_COL,
]

# =========================
# Target columns
# =========================

TARGET_RETURN = "Return"
TARGET_CLASS = "ReturnMean_year_Label"

TARGET_COLUMNS = [
    TARGET_RETURN,
    TARGET_CLASS,
]

# =========================
# Canonical feature columns
# =========================

FEATURE_COLUMNS = [
    "市值(百萬元)",
    "收盤價(元)_年",
    "Unknown masked parameter",
    "股價淨值比",
    "股價營收比",
    "淨值報酬率─稅後",
    "資產報酬率 ROA",
    "營業利益率 OPM",
    "利潤邊際 NPM",
    "負債/淨值比",
    "流動比率",
    "速動比率",
    "存貨週轉率 (次)",
    "應收帳款週轉次",
    "營業利益成長率",
    "稅後淨利成長率",
]

REQUIRED_COLUMNS_BEFORE_YEAR = [
    STOCK_ID_COL,
    STOCK_NAME_COL,
    YEARMONTH_COL,
    *FEATURE_COLUMNS,
    TARGET_RETURN,
    TARGET_CLASS,
]

REQUIRED_COLUMNS_AFTER_CLEANING = [
    STOCK_ID_COL,
    STOCK_NAME_COL,
    YEARMONTH_COL,
    YEAR_COL,
    *FEATURE_COLUMNS,
    TARGET_RETURN,
    TARGET_CLASS,
]