# AIFT Final Project

以台股年度資料進行股票篩選建模、時間序列回測、風險衡量、外部財務資料重建，以及 CLI / Web Demo 展示的完整專題實作。

本專案目前已包含：

- Task 1：`Decision Tree (Entropy)` 股票篩選與回測
- Task 2：`Logistic Regression`、`Random Forest`、`Gradient Boosting` 比較
- Task 2 延伸：`SVR-GA` 連續報酬預測與排序選股
- Task 3：外部財務資料抓取、特徵重建、外部資料重跑與 benchmark 比較
- Demo：CLI Demo 與 Web Demo

---

## 1. 專案重點

本 repo 的核心目標不是只訓練分類器，而是把整條研究鏈路做完整：

1. 清理原始資料並建立可重現的 schema
2. 用時間序列切分避免未來資料外洩（future leakage）
3. 將模型分數轉成 Top-K 選股結果
4. 將選股結果轉成可回測投組
5. 比較報酬、風險、benchmark 與外部重跑結果
6. 提供可展示的 CLI / Web Demo

---

## 2. 目前專案狀態

以下為目前 repo 內既有正式輸出檔的狀態快照：

| 項目 | 目前狀態 |
| --- | --- |
| 主資料集 | `data/processed/cleaned_top200.csv`，`2400` 筆、`22` 欄，年份 `1997-2008` |
| 特徵數 | `16` 個 canonical features |
| Temporal splits | `next_year = 11` 組、`remaining_years = 11` 組 |
| Saved models | 主流程模型共 `55` 個：Decision Tree `11`、Task 2 classifiers `33`、SVR-GA `11` |
| 已存在 Demo 輸出 | `outputs/demo/` 下已有 predictions / selections / metrics / profile |
| Web Demo | 已完成，可直接啟動 |
| 目前內部最佳策略 | `Random Forest`，`Top-5`，`equal`，`net annualized return = 26.34%` |
| 目前外部最佳策略 | `外部 Random Forest`，`Top-20`，`equal`，`net annualized return = 25.75%` |

注意：上述數值是根據目前 repo 內既有輸出檔彙整而來；若重新訓練、重跑外部資料或更新輸出檔，結果可能會變動。

---

## 3. 需求對應

| 需求 | 對應實作 |
| --- | --- |
| 資料清理與移除 `年月 = 200912` | `prepare_data.py`, `src/preprocessing.py` |
| Temporal validation | `create_splits.py`, `src/validation.py` |
| Task 1：Decision Tree / ID3-like | `train_decision_tree.py`, `build_portfolios.py`, `src/models.py`, `src/prediction.py` |
| Task 2：第二種方法與比較 | `train_task2_models.py`, `build_task2_portfolios.py` |
| Task 2 延伸：SVR-GA | `train_svr_ga.py`, `build_svr_ga_portfolios.py`, `src/svr_ga.py` |
| 投組建構、報酬與風險評估 | `src/portfolio.py`, `src/metrics.py`, `src/benchmark.py` |
| Task 3：外部資料抓取與重跑 | `run_external_crawler.py`, `rerun_external_pipeline.py`, `external_benchmark.py` |
| 結果整併與視覺化 | `rebuild_all_models_portfolio_metrics.py`, `generate_all_model_png.py`, `generate_external_figures.py` |
| CLI Demo | `demo.py`, `src/demo_runner.py` |
| Web Demo | `web_demo.py`, `src/web_demo.py`, `src/project_dashboard.py`, `web/` |

---

## 4. 方法設計

### 4.1 預測目標

- 分類任務：
  - `ReturnMean_year_Label = 1`：該股票年報酬高於同年平均
  - `ReturnMean_year_Label = -1`：否則
- 回歸任務：
  - `Return`

### 4.2 Temporal validation

本專案保留兩種切分方式：

- `next_year`
  - 逐年擴張式訓練，下一年做 out-of-sample 測試
  - 目前主模型訓練、回測與比較結果主要使用此模式
- `remaining_years`
  - 以較早年份訓練，測試所有剩餘未來年份
  - 較貼近報告或課堂 TV 圖的規格對照

### 4.3 選股與投組

- 排名依據：
  - 分類模型：`score_label_1`
  - SVR-GA：`predicted_return`
- Top-K：
  - `5`, `10`, `20`, `30`
- 權重方式：
  - `equal`
  - `score`
- 再平衡頻率：
  - annual rebalancing

### 4.4 交易成本假設

根據 `src/config.py`：

- Buy fee = `0.0399%`
- Sell fee = `0.0399%`
- Sell tax = `0.3%`

---

## 5. 專案結構

```text
proj3/
├─ data/
│  ├─ raw/
│  │  └─ top200.xlsx
│  ├─ processed/
│  │  ├─ cleaned_top200.csv
│  │  ├─ temporal_splits_next_year.csv
│  │  └─ temporal_splits_remaining_years.csv
│  └─ external/
│     ├─ raw/
│     ├─ processed/
│     ├─ metadata/
│     └─ tickers.csv
├─ outputs/
│  ├─ predictions/
│  ├─ metrics/
│  ├─ selections/
│  ├─ portfolio/
│  ├─ benchmarks/
│  ├─ figures/
│  ├─ model_reports/
│  ├─ logs/
│  ├─ demo/
│  └─ external/
├─ saved_models/
├─ src/
│  ├─ preprocessing.py
│  ├─ validation.py
│  ├─ models.py
│  ├─ prediction.py
│  ├─ portfolio.py
│  ├─ metrics.py
│  ├─ benchmark.py
│  ├─ svr_ga.py
│  ├─ demo_runner.py
│  ├─ web_demo.py
│  ├─ project_dashboard.py
│  └─ external/
├─ web/
│  ├─ index.html
│  ├─ app.css
│  └─ app.js
├─ prepare_data.py
├─ create_splits.py
├─ train_decision_tree.py
├─ build_portfolios.py
├─ train_task2_models.py
├─ build_task2_portfolios.py
├─ train_svr_ga.py
├─ build_svr_ga_portfolios.py
├─ run_external_crawler.py
├─ rerun_external_pipeline.py
├─ external_benchmark.py
├─ compare_external_benchmarks.py
├─ generate_external_figures.py
├─ build_external_metadata.py
├─ rebuild_all_models_portfolio_metrics.py
├─ generate_all_model_png.py
├─ demo.py
├─ web_demo.py
├─ main.py
└─ validate_*.py
```

---

## 6. 安裝環境

建議環境：

- Python `3.10+`

安裝套件：

```bash
pip install -r requirements.txt
```

`requirements.txt` 目前保留的是實際使用到的必要套件：

- `pandas`
- `numpy`
- `scikit-learn`
- `joblib`
- `openpyxl`
- `xlrd`
- `matplotlib`
- `pillow`
- `requests`
- `yfinance`

若要執行 Task 3 的 `FinMind` 抓取，建議設定：

```powershell
$env:FINMIND_TOKEN="your_token_here"
```

若未設定 token，仍可用 cache / `yfinance` fallback，但資料完整度與可抓取量可能受限。

---

## 7. 快速開始

### 7.1 直接開 Web Demo

如果你只是想快速看目前專案成果與互動展示，最簡單的方式是直接啟動 Web Demo：

```bash
py -3 web_demo.py --host 127.0.0.1 --port 8771
```

或使用統一入口：

```bash
py -3 main.py web-demo --host 127.0.0.1 --port 8771
```

啟動後開啟：

```text
http://127.0.0.1:8771
```

Web Demo 會直接載入：

- 專題總覽
- 需求對照
- 各模型排行榜
- 既有圖表成果
- 重要下載入口
- 最新 Demo 快照
- 上傳資料後的互動推論流程

### 7.2 CLI Demo

```bash
py -3 demo.py --input path\to\new_testing_data.xlsx --model random_forest --top-k 10
```

或：

```bash
py -3 main.py demo --input path\to\new_testing_data.xlsx --model random_forest --top-k 10
```

支援的 Demo 模型：

- `decision_tree_entropy`
- `logistic_regression`
- `random_forest`
- `gradient_boosting`

### 7.3 統一入口 `main.py`

`main.py` 目前提供四個常用入口：

```bash
py -3 main.py external-crawl
py -3 main.py external-rerun
py -3 main.py demo --input path\to\file.xlsx
py -3 main.py web-demo --port 8771
```

注意：`main.py` 不是所有腳本的總入口；完整重現主流程仍建議依第 8 節逐步執行各腳本。

---

## 8. 從頭重現完整主流程

若你要重新生成主專題結果，建議依下列順序執行：

### Step 1. 資料清理

```bash
py -3 prepare_data.py
```

### Step 2. 建立 temporal splits

```bash
py -3 create_splits.py
```

### Step 3. Task 1：訓練 Decision Tree

```bash
py -3 train_decision_tree.py
```

### Step 4. Task 1：建立投組與 benchmark

```bash
py -3 build_portfolios.py
```

### Step 5. Task 2：訓練分類模型

```bash
py -3 train_task2_models.py
```

### Step 6. Task 2：建立投組

```bash
py -3 build_task2_portfolios.py
```

### Step 7. Task 2 延伸：訓練 SVR-GA

```bash
py -3 train_svr_ga.py
```

### Step 8. Task 2 延伸：建立 SVR-GA 投組

```bash
py -3 build_svr_ga_portfolios.py
```

### Step 9. 重建全模型比較表

```bash
py -3 rebuild_all_models_portfolio_metrics.py
```

### Step 10. 產生全模型比較圖

```bash
py -3 generate_all_model_png.py
```

---

## 9. Task 3：外部資料流程

### 9.1 抓取外部資料並重建外部資料集

```bash
py -3 run_external_crawler.py --source finmind --use-cache
```

常用選項：

- `--start-year`
- `--end-year`
- `--source {finmind,yfinance}`
- `--use-cache`
- `--force-refresh`
- `--max-tickers`

### 9.2 在外部資料上重跑模型與投組

```bash
py -3 rerun_external_pipeline.py
```

### 9.3 建立 benchmark、對齊與圖表

```bash
py -3 external_benchmark.py
py -3 compare_external_benchmarks.py
py -3 generate_external_figures.py
py -3 build_external_metadata.py
```

若你只想透過統一入口跑主要外部流程：

```bash
py -3 main.py external-crawl --source finmind --use-cache
py -3 main.py external-rerun
```

---

## 10. 驗證與品質檢查

本 repo 已提供明確的 validation scripts，用來確認輸出數量、切分邏輯、權重、圖表與模型檔是否正確。

### 10.1 建議驗證順序

```bash
py -3 validate_splits.py
py -3 validate_decision_tree.py
py -3 validate_portfolios.py
py -3 validate_task2.py
py -3 validate_svr_ga.py
py -3 validate_all_model_pngs.py
```

### 10.2 各驗證腳本檢查內容

| 腳本 | 驗證內容 |
| --- | --- |
| `validate_splits.py` | `next_year` / `remaining_years` 切分數量、年份、rows、leakage-free |
| `validate_decision_tree.py` | Task 1 predictions、metrics、feature importance、rules、saved models |
| `validate_portfolios.py` | Task 1 selected stocks、權重和、portfolio rows、benchmarks |
| `validate_task2.py` | Task 2 三個分類模型的 predictions / metrics / portfolio / saved models |
| `validate_svr_ga.py` | SVR-GA predictions、GA search log、selection leakage、portfolio recomputation |
| `validate_all_model_pngs.py` | 關鍵圖表是否存在且可開啟 |

---

## 11. Demo 輸入資料格式

### 11.1 支援格式

- `.csv`
- `.xlsx`
- `.xls`

### 11.2 必要欄位

- `stock_id`
- `yearmonth`

### 11.3 建議欄位

- `stock_name`
- `Return`
- `ReturnMean_year_Label`
- 其餘 16 個 canonical features

### 11.4 Demo 行為

- 若輸入只有 features：
  - 仍可做推論與 Top-K 選股
  - 不會計算 realized portfolio performance
- 若輸入同時含 `Return`：
  - 會額外輸出投組報酬與風險摘要
- 若輸入同時含 `ReturnMean_year_Label`：
  - 會額外輸出分類驗證指標

### 11.5 Web Demo 與 CLI Demo 共用邏輯

`src/demo_runner.py` 是 CLI / Web Demo 的共同核心，負責：

- preprocessing
- 模型載入或歷史 refit
- prediction
- Top-K ranking
- portfolio summary
- output persistence

---

## 12. Web Demo 實作架構

目前 Web Demo 並不是規劃稿，而是已實作版本。

### 12.1 前端

- `web/index.html`
- `web/app.css`
- `web/app.js`

### 12.2 後端

- `web_demo.py`
- `src/web_demo.py`
- `src/project_dashboard.py`

### 12.3 主要 API

| Method | Path | 說明 |
| --- | --- | --- |
| `GET` | `/api/health` | 檢查資料與模型是否存在 |
| `GET` | `/api/models` | 取得 Demo 可用模型 |
| `GET` | `/api/project/overview` | 取得首頁儀表板資料 |
| `POST` | `/api/demo/upload` | 上傳測試資料並做 preprocessing / 驗證 |
| `POST` | `/api/demo/run` | 執行模型推論與 Top-K 選股 |
| `GET` | `/api/demo/result/{run_id}` | 取得某次執行結果 |
| `GET` | `/artifacts/...` | 讀取專案內圖表與輸出檔 |

### 12.4 Web Demo 設計原則

- 使用既有正式輸出檔做 dashboard 展示
- 上傳流程與正式 preprocessing 對齊
- 優先載入已保存模型；沒有對應年度模型時才使用歷史資料 refit
- 不使用 realized return 做選股排序，避免 selection leakage

---

## 13. 主要輸出檔

| 類別 | 路徑 |
| --- | --- |
| 清理後資料 | `data/processed/cleaned_top200.csv` |
| Temporal splits | `data/processed/temporal_splits_next_year.csv`, `data/processed/temporal_splits_remaining_years.csv` |
| Task 1 predictions | `outputs/predictions/decision_tree_predictions.csv` |
| Task 2 predictions | `outputs/predictions/task2_classification_predictions.csv` |
| SVR-GA predictions | `outputs/predictions/svr_ga_regression_predictions.csv` |
| 主投組指標 | `outputs/portfolio/*.csv` |
| Benchmark | `outputs/benchmarks/*.csv` |
| Task 1 tree rules | `outputs/model_reports/decision_tree_rules/*.txt` |
| 主圖表 | `outputs/figures/**` |
| External outputs | `outputs/external/**` |
| Demo outputs | `outputs/demo/**` |
| Saved models | `saved_models/*.joblib` |

---

## 14. 重要模組說明

| 模組 | 功能 |
| --- | --- |
| `src/config.py` | 路徑、模型參數、Top-K、交易成本、輸出位置 |
| `src/schema.py` | 欄位 schema 與 feature list |
| `src/preprocessing.py` | 欄名標準化、數值清理、yearmonth 處理、資料整理 |
| `src/validation.py` | temporal split 建立與 leakage 檢查 |
| `src/models.py` | Decision Tree、LR、RF、GB pipelines |
| `src/prediction.py` | 訓練 / 預測 / 分數輸出 |
| `src/svr_ga.py` | GA 搜尋 SVR 參數與回歸訓練 |
| `src/portfolio.py` | 選股、權重計算、投組報酬 |
| `src/metrics.py` | 分類 / 回歸 / 投組績效與風險指標 |
| `src/benchmark.py` | all-stock 與 random Top-K benchmarks |
| `src/demo_runner.py` | CLI / Web Demo 共用核心 |
| `src/project_dashboard.py` | Dashboard 資料彙整與結果摘要 |
| `src/web_demo.py` | Web API 與靜態頁面服務 |
| `src/external/*` | 外部資料抓取、映射、特徵工程、資料集重建 |

---

## 15. 報告撰寫時應注意的規格說明

### 15.1 主回測模式

目前主回測結果以 `next_year` 為主，因為它最適合逐年投資期比較與 Demo 展示。

若你要寫正式報告，建議明確說明：

- `remaining_years`：較貼近課程 TV 規格圖
- `next_year`：較適合年度投資回測與 Top-K 策略比較

### 15.2 External dataset 的性質

Task 3 的外部資料不是原始教學資料的逐欄完全複製，而是：

- 盡量使用公開來源重建相似欄位
- 對缺失嚴重或無法完全對應的欄位使用 proxy features
- 保留 metadata 與 data quality reports 解釋限制

### 15.3 Reproducibility

此 repo 已具備：

- 明確資料路徑
- 固定輸出位置
- validation scripts
- saved models
- demo artifacts

若已執行外部資料重跑，還會額外產生 external rerun 對應輸出與外部模型 artifacts。

因此可作為：

- code submission
- report 附錄說明來源
- demo 展示環境

---

## 16. 建議使用方式

### 若你要看結果

直接開 Web Demo：

```bash
py -3 web_demo.py --port 8771
```

### 若你要重新訓練

依第 8 節從 `prepare_data.py` 開始完整重跑。

### 若你要做期末展示

建議使用：

- Web Demo 顯示 dashboard 與既有結果
- 再用一份測試資料做即時上傳展示

### 若你要做報告

建議引用：

- `outputs/portfolio/*.csv`
- `outputs/metrics/*.csv`
- `outputs/figures/**`
- `outputs/external/**`

---

## 17. License / Note

本專案為課程專題實作，資料來源與外部資料使用方式請依課堂規範、資料來源條款與實際使用情境自行確認。
