# 預先登記：用 CellViT 原作者的後處理重算 stage-3 的 K／N 判讀和 hfv 論文 Fig 2 的 PQ（2026-11-22）

只重跑 CellViT（推論，用同一個公開預訓練權重 CellViT-256），不訓練、不重新生成任何影像，不改 PixCell，不碰 share-space-paper。這份文件在任何新的 CellViT 讀數出現之前 commit 並 push。

## 0. 為什麼要做、影響哪些決定

- **起因：** sim-exp-validation `docs/experiments/cellvit_samh_20261122/`（1e42e9d3）發現，我們流程一直用的 CellViT 執行程式（hfv `stages/run_cellvit_local.py`，以下叫 **runner 路徑**）和 CellViT 原作者 repo 的推論程序（以下叫 **official 路徑**）有三處不同：
  1. **正規化：** runner 用 ImageNet 的平均值／標準差；權重檔自己的設定是平均 0.5、標準差 0.5。
  2. **「是不是核」的機率：** runner 對第 1 個通道取 sigmoid；原作者是兩個通道一起做 softmax。
  3. **切開相碰的核：** CellViT 除了「是不是核」以外，還會輸出每個像素到所屬核中心的水平／垂直距離（HV 圖）。原作者用 HV 圖做分水嶺（HoVer-Net 的做法），把相碰的核切成各自的物件；runner 不用 HV 圖，相碰的核就變成一個物件。
- 在 CRC16 的 192 張圖塊上，official 路徑讓 CellViT-256 找到 Lin 真實核（Hoechst 螢光分割）的比例從 0.41 升到 0.61，「一個物件含 ≥ 2 個真實核」的比例從 0.36 降到 0.14。
- 以前所有 CellViT 的數字都是 runner 路徑。這次重算兩組：
  - **A. stage-3 的 K／N 判讀**（hfv `docs/experiments/hfv_retrain2_20261121/RESULTS_stage3.md`，判讀表 PREREG §5、操作定義 §9.5）。原判讀是第 2 列「留用 K」，主要依據是 CellViT 在生成圖上的類型判讀（癌細胞核被判成「腫瘤」的比例，NP 0.349 對 KP 0.436；Δ 的 N − K：own −0.086 [−0.121, −0.056]、other −0.101 [−0.142, −0.063]）。**影響的決定：** sim2real 繼續用 K 還是改用 N（決定 D2）。規則照原樣套用，不重新詮釋。
  - **B. hfv 論文 Fig 2 的 PQ。** 不影響任何決定，是為了論文數字正確。新舊並列，附區間。

## 1. 名詞

- **模型 K：** 重訓 1 的 Concat（`tr_concat/step_0006520`），訓練遮罩 R0、舊的血管／氧／葡萄糖通道。
- **模型 N：** 重訓 2 的 Concat（第 20 epoch），訓練遮罩 L1（和 H&E 對齊的 Lin 核遮罩）、ch7–9 用方法 C。
- **runner／official 路徑：** 見 §0。official 路徑 = CellViT repo（TIO-IKIM/CellViT 05097e1）`cell_segmentation/inference/cell_detection.py` 的步驟：權重檔設定的正規化 → binary 與 type 兩張圖各做 softmax → `model.calculate_instance_map(magnification=40)`（HV 分水嶺；兩個權重檔都是 x40 版）。
- **每個核的類型（official）：** CellViT repo `DetectionCellPostProcessor` 的規則：在這個核的像素上數「類型圖 argmax」各類型的像素數，取最多的那一類；最多的若是背景（0），改取第二多的。`type_prob` = 該類型像素佔這個核的比例。runner 路徑也是像素多數決，但它的「核」是沒切開的大塊。
- **類型區分度 Δ** = 癌→腫瘤 − 健康→腫瘤：CyCIF 判為癌的真實細胞，配到的 CellViT 核被判成「腫瘤（Neoplastic）」的比例，減去健康細胞的同一比例。0 代表 CellViT 的「腫瘤」判斷和真實類型無關。
- **忠實度** = 模型的值 − 真實 H&E 參考線（每張圖塊）。值 = 生成圖 CellViT 核對「模型自己的訓練樣式遮罩」的 PQ（次要 PQ），或對 Lin 細胞中心點的一對一配對 F1；參考線 = 同一張圖塊的真實 H&E CellViT 的同一個量。
- **PQ（panoptic quality）：** 兩組物件用 IoU ≥ 0.5 一對一配對，PQ = 配上物件的平均 IoU × 偵測 F1（PixCell cb62fa7 `_compute_pq`）。1 = 兩組一模一樣。

## 2. 程式

- **runner 加一個選用旗標，預設不變：** hfv `stages/run_cellvit_local.py` 在 467a54f（分支 `cellvit-official-20261122`）加 `--postprocess {runner,official}`（預設 runner）、`--magnification`（預設 40）、`--save-instances`（只在 official 時存整數實例圖）。輸出 JSON 的欄位和 runner 一樣（`centroid`、`contour`、`bbox`、`type_cellvit`、`type_name`、`type_prob`），所以後面所有打分程式不用改。測試：`tests/test_run_cellvit_local.py`（假模型，8 項通過）。
- **兩個已經做完的程式檢查（只看是否相同，沒有產生任何讀數）：**
  - 預設路徑：新程式在 RAW 前 32 張上的 JSON 和應用 2 `cv/RAW` 的舊 JSON **逐位元組相同**（32／32）→ 預設確實沒變，舊數字可重現。作業 43465716。
  - official 路徑：新程式在 RAW 前 32 張上的實例圖和 cellvit_samh `full/official_cv256/RAW.npz` **逐位元相同**（32／32），JSON 的核數 = 實例數（1,472）。作業 43465715。
- **其他程式一律沿用、不修改：**
  - stage-3 讀數 1：hfv_retrain2 `scripts/r2score.py`（`metrics` 經 `pxscore.py`、`stats`、`decide`）和 `s3metrics.slurm`。
  - stage-3 讀數 3：hfv_retrain2 `scripts/r2_app8.py`；它的 K 端閘門參照用 sim2real `app8/code/app8_stats.py stats` 在新的 JSON 上重算（見 §4.2）。
  - Fig 2：見 §5。

## 3. 推論設定（和以前所有 CellViT-256 的跑法相同，只換後處理）

- CellViT-256.pth（`crc16-20261024/cellvit/ckpt`）、同一個 CellViT repo 和 Python 環境、原解析度 256 px、batch 32、`NVIDIA_TF32_OVERRIDE=0`、cheme CPU 8 核（實測：official 路徑 32 張 86 s，含載入模型；cellvit_samh 的 CellViT-256 official 384 張約 9 分鐘，所以不需要 GPU）。
- `scripts/cvo_infer.slurm RUN official NAMES`（Hyak 新目錄 `/gscratch/cheme/chiu/cellvit_official_20261122/`，其他目錄唯讀；hfv 執行程式是 467a54f 的凍結複本 `$D/hfv`）。
- 輸入的 zip 全部是以前 runner 路徑用過的同一個檔案（唯讀）：
  - stage-3 讀數 1：`hfv_retrain2_20261121/stage3/zips/{N,K,R}_{crc36ho,crc33sec}.zip`（300＋100 張 × 3 組）。
  - stage-3 讀數 3：`sim2real_apps_20261120/app2/zips/RAW.zip`（192）、`app8/zips/{KP,KD}.zip`、`hfv_retrain2_20261121/stage3/zips/{NP,ND}.zip`（各 384）；K 端閘門參照另需 `app8/zips/{CP,CD,KE}.zip`（各 384，§4.2）。
  - Fig 2：見 §5。
- **閘門：** 每個輸出目錄的 JSON 數 = zip 內圖數；RAW、KP、NP 的 official 實例圖要和 cellvit_samh `full/official_cv256/{RAW,KP,NP}.npz` 逐位元相同。不同就停下查原因。

## 4. A：stage-3 K／N 判讀

### 4.1 讀數 1（忠實度；外觀）

- 建一個新的 stage-3 根目錄 G′ = `$D/full/g3`：`inputs`、`packs`、`gen`、`kfig2` 是舊 stage-3 目錄的符號連結（唯讀），`cv/<src>_<set>` 是 official 的 JSON。在 G′ 上原樣執行 `s3metrics.slurm`（`r2score.py metrics` + `stats`）。
- 得到的量（和舊的同一個定義、同一支程式）：次要 PQ（N 對 L1、K 對 R0）、真實 H&E 參考線、忠實度 PQ；對 Lin 中心點的 F1 與忠實度 F1；主要 PQ（生成圖對真實 H&E 的 CellViT，描述）；每張偵測數、物件數比。
- 圖塊 bootstrap 10,000 次、seed 42（`r2score.py stats` 原本就是這個設定），N − K 配對差，95 % 百分位區間。
- **閘門：** 不經過 CellViT 的欄（`lpips_*`、`hed_*`、`n_gt_*`）和 FUD 必須和舊的 `score/tiles_<set>.csv`、`fud.json` 相同（同一支程式、同一批圖）。不同就停下。

### 4.2 讀數 3（類型區分度，CRC16 的 12 個視窗）

- `r2_app8.py` 原樣執行，CV 目錄全部指到 official 的 JSON（RAW、KP、KD、NP、ND）。
- `r2_app8.py` 有一個閘門：K 端（R、KP、KD）每個視窗的配對計數必須等於 `results_app8.json`。舊的 `results_app8.json` 是 runner 路徑算的，所以這次的參照改成：用 `app8_stats.py stats`（應用 8 自己的程式，未修改）在 official JSON 上重算一份 `results_app8_official.json`。兩支獨立的程式對 K 端算出同樣的計數，閘門才通過。`app8_stats.py` 需要全部 5 臂，所以 CP、CD、KE 也跑 official。
- 讀數：每臂的 Δ、癌→腫瘤、健康→腫瘤、圖塊內 Δ（Mantel–Haenszel 風險差）、核面積中位數、偵測判為腫瘤的比例、每張偵測數；N − K = NP − KP、ND − KD 的配對差。
- **區間：** 判讀用 `r2_app8.py` 原本的設定：12 個視窗 bootstrap 2,000 次、`default_rng(20261128)`（應用 8 的慣例；§9.3 寫定的規則，照原樣用）。
- **敏感度（不進判讀）：** 同一支 `r2_app8.py`，只把視窗 bootstrap 改成 10,000 次、seed 42（`scripts/cvo_r2app8_b10k.py` 在執行前設定 `app8_stats.NB`、`SEED`），official 和舊 runner 的 JSON 各算一次，讓新舊在同一組抽樣上並列。

### 4.3 判讀（PREREG §5 的表、§9.5 的操作定義，原樣套用）

- `r2score.py decide G′ r2_app8_official.json`：
  - 先看第 2 列：NP − KP 或 ND − KD 的 Δ 區間上界 < 0 → 留用 K，回報原因。
  - 再看第 1 列：忠實度四個 N − K 區間（次要 PQ、F1 × crc36ho、crc33sec）全部 > 0，且兩種風格的 Δ 區間都不在 0 以下 → sim2real 改用 N。
  - 其餘 → 第 3 列：回報，遮罩選擇交給使用者。
- 報告：新舊兩個判讀列並列；列有沒有改變；每個判讀區間新舊並列。

### 4.4 順帶的報告（不改這份文件的判讀）

- `results_app8_official.json` 會帶出應用 8 自己的判讀列（K 對 C，prereg_app8_retrained_concat §7 的規則，`app8_stats.py` 原樣計算）。K 之所以是 sim2real 的生成器，正是應用 8 的判讀。所以照實並列新舊列；如果改變，交給使用者決定，這裡不重新詮釋。

## 5. B：hfv 論文 Fig 2 的 PQ

（2026-11-22 補上；stage-3 的推論已在跑，但這一節在任何 Fig 2 影像的新 CellViT 結果出現之前 commit 並 push。）

**不影響決定**，只為論文數字正確。舊數字的來源見 hfv `docs/experiments/hfv_retrain_20261118/README.md`（參考帶表、變體表、「PQ 參考遮罩」一節）。

### 5.1 這些 PQ 在算什麼

- 所有 PQ 的「預測端」都是 CellViT 在生成圖上找到的核。「參考端」有兩種：
  - **主要 PQ（論文的定義）：** 參考 = 同一張圖塊的**真實 H&E** 上的 CellViT 核。論文的作法是先把這些核畫成一張二值的 `cell_masks` 圖（核內 = 1），PixCell 讀回來時再取 8 連通區塊當成一個一個的核（`_load_gt_instance_mask`）。所以論文的 PQ 問的是：「生成圖的核，是不是長在真實組織有核的位置、形狀是否相近」。
  - **次要 PQ（Lin 遮罩）：** 參考 = 重訓模型的輸入遮罩（Lin 的螢光核分割，R0 畫法），也就是模型被要求照著畫的那張遮罩。參考線 = 真實 H&E 的 CellViT 對同一張 Lin 遮罩的 PQ（真實組織本身能達到多少）。
- **「換成 official 後處理」在兩端各代表什麼：**
  - 預測端：生成圖的核改用 official 路徑切開。
  - 參考端（主要 PQ）：真實 H&E 的核也改用 official 路徑，再照論文的方式畫成二值遮罩、讀回連通區塊。注意：二值化會把 official 已切開、但彼此相碰的核重新連成一塊。這就是論文流程本身會發生的事，所以主要讀數照做；另報一個**不二值化**的版本（參考 = official 的實例圖本身，`pq_inst`），看這一步的影響。
  - 次要 PQ 的參考端是 Lin 遮罩，和 CellViT 無關，只有預測端改變；參考線的真實 H&E 端改變。

### 5.2 影像（全部仍在，不重新生成）

- **論文的舊模型影像（CRC33）：** S3 `share_space/04_inference/si_a1_a2/tiles/`：
  - Vanilla ControlNet（`a2_off_shelf`）1,000 張，圖塊 = `metric_tile_ids.txt`（論文的 0.7035）；其中 Fig 2 的 100 張（`cache.json.bak_n100` 的 `metric_tile_ids`）= 論文的 0.6987。
  - Concat（`a1_concat`）、Grouped + bypass（`a2_bypass_full_tme`）、Per-channel（`a1_per_channel`）各 100 張（同一組 100 張）。
  - 真實 H&E：`tiles/gt/`（1,000 張）。
  - **Grouped（production）不重算：** `tiles/production/` 的圖在 2026-07-23 被 `metrics_io select-tiles` 換成別組圖（重算得 0.7788，cache 寫 0.5694），論文那組原圖已經不在這個位置。照規則不重新生成，報告裡寫明。
- **重訓模型影像（hfv_retrain_20261118）：** S3 `hfv_retrain_20261118/outputs/score_fig2_{crc36ho,crc33sec}.tar`，五個變體（Vanilla、Grouped、Concat、Grouped + bypass、Per-channel）各 300／100 張；真實 H&E = 同一個 tar 的 `tiles/gt`，閘門：和 stage-3 `zips/R_<set>.zip` 逐像素相同（是的話，真實 H&E 的 CellViT 直接用 §4 已在跑的 `R_<set>`）。
- 本機下載後打包成 `F2_<組>_<變體>.zip`（P = 論文影像，36 = crc36ho，33 = crc33sec；van、grp、cat、gbp、pch），上傳到 `$D/zips/`，每個 zip 的 sha256 記在 `zips/manifest.json`。

### 5.3 CellViT

- 每個 zip 兩條路徑都在 Hyak cheme CPU 上跑（`cvo_infer.slurm`；official → `$D/full/cv`，runner → `$D/full_runner/cv`）。runner 也重跑，是為了讓新舊在**同一台機器、同一支程式**上逐張配對；論文和 README 的舊數字是 Colab GPU 上跑的 runner。
- 重訓組真實 H&E 的 runner 結果直接用 stage-3 的 `cv/R_<set>`（同一個 runner、同一台機器類型）。

### 5.4 讀數（`scripts/cvo_fig2.py`，PixCell 函式，未修改）

每張圖塊、每個變體、每條路徑：`pq_main`（主要）、`pq_inst`（不二值化）、重訓組另有 `pq_lin`（次要）和 `ref_lin`（參考線）、生成圖與真實圖的物件數。

### 5.5 報告

- 每個變體、每組：舊（論文或 README 公布的數）｜runner 重跑｜official｜official − runner（配對差，區間），主要 PQ 和次要 PQ 各一張表；參考線同樣並列。
- 變體 − Vanilla 的配對差（兩條路徑各一次）：看變體的排名和「比 Vanilla 高或低」在換後處理後是否一樣。只描述，不設規則。
- **重現檢查（不是門檻）：** runner 重跑的平均值和公布值的差、逐張相同的比例（公布的逐張值：論文 100 張 = `eval/diag/pqmatrix_crc33fig2.json`；重訓組 = S3 `realref_fig2_<set>.tar`、`scoreres_fig2_<set>.tar` 的 `fig2_per_tile.csv`）。另外：重訓 Concat 的 runner 次要 PQ 要和 stage-3 的 `pq2_K` 逐張相同（同一批圖、同一台機器類型、同一支程式），不同就停下查。
- 區間：圖塊 bootstrap 10,000 次、seed 42；每組一個抽樣矩陣，所有變體、路徑共用（配對）。論文 Vanilla 1,000 張和 100 張分開抽。

## 6. 統計、執行規則

- 圖塊 bootstrap 一律 10,000 次、seed 42、95 % 百分位區間、配對差（讀數 3 的判讀區間例外，見 §4.2，那是原規則本身的設定）。不設任何新的數字門檻。
- Hyak 只寫新目錄 `/gscratch/cheme/chiu/cellvit_official_20261122/`；cheme CPU，留 120 核。任何失敗：停下診斷，不原樣重送。
- 結果另寫 `RESULTS.md`，附圖和 `data/`。
