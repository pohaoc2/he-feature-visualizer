# 預先登記：輸入遮罩就是標準答案時的論文式 PQ，全程用 CellViT 官方後處理（2026-11-22）

這份文件在任何新影像、新 CellViT 讀數出現之前 commit 並 push。不訓練、不改 PixCell（只呼叫它的函式）、不碰 share-space-paper。結果另寫 `RESULTS.md`。

## 0. 為什麼要做、影響哪個決定

**要回答的問題：** 論文 Fig 2 報的 PQ（PixCell 原版 0.70、Concat 0.78、Grouped + bypass 0.79）是在「輸入遮罩就是標準答案」的設定下量的：

- 生成器拿到的細胞核遮罩 = CellViT 在**同一張真實 H&E** 上找到的核；
- 生成完再用 CellViT 找生成圖上的核，和那張輸入遮罩比 PQ。

PixCell 自己的論文和 share-space 論文 Fig 2 都是這樣設定的。問的是「生成器有沒有照著輸入遮罩畫核」，兩邊的核用同一種方法定義。

但到目前為止，兩邊用的 CellViT 都是 hfv 的舊執行程式（以下叫 **hfv 舊後處理**），不是 CellViT 原作者的流程（**官方後處理**）。兩者的差別見 §1。前一份實驗 `cellvit_official_20261122`（RESULTS §4.1）只把**評分端**換成官方後處理：生成器拿到的仍是舊後處理畫的遮罩，評分端卻用官方後處理。兩端不一致，論文式 PQ 掉了 0.10–0.13，而且 Concat 和 Grouped + bypass 的先後對調。這個數不能直接當論文數字：掉下來的部分有多少是「生成器沒照遮罩畫」，有多少只是「兩端用了不同的核定義」，分不出來。

這次把整條管線換成官方後處理：

真實 H&E → CellViT（官方）→ 細胞核遮罩 → 生成器 → 生成圖 → CellViT（官方）→ 生成圖的遮罩 → PQ（生成圖的遮罩 對 同一張輸入遮罩）。

**影響的決定：** 論文 Fig 2 要報哪一組數字、怎麼排序。論文本文由使用者決定和修改，這裡只提供數字和依規則得出的排序。另外，如果某個 Grouped 架構照遮罩畫得明顯比 Concat 好（§8.4），也標記給 sim2real 生成器的問題參考；另一份實驗（`fidelity_variants_20261122`）在 Lin 遮罩上檢查同一件事，這裡不重複。

## 1. 名詞

- **hfv 舊後處理（runner）／官方後處理（official）：** CellViT 輸出三樣東西：每個像素「是不是核」的機率、每個像素到所屬核中心的水平／垂直距離（HV 圖）、每個像素的細胞類型。
  - hfv 舊後處理（`stages/run_cellvit_local.py` 預設路徑）：用 ImageNet 的平均／標準差正規化（權重檔自己的設定是 0.5／0.5），取「是不是核」機率 ≥ 0.5 的像素，把每一塊連在一起的像素當成一個核。**不會把相碰的核切開。**
  - 官方後處理（CellViT 原作者 `cell_detection.py` 的步驟，hfv 467a54f 的 `--postprocess official`）：照權重檔設定正規化，再用 HV 圖做分水嶺（HoVer-Net 的做法），**把相碰的核切成各自的物件**。
- **論文式 PQ（paper-style PQ）：** 只在「輸入遮罩 = 標準答案，而且兩邊用同一種 CellViT 後處理」時才報。做法照 PixCell `metrics_io.py compute`：
  - 標準答案 = 生成器拿到的那張遮罩 PNG（核內 = 255），讀回時取 8 連通區塊，每塊當一個核（`_load_gt_instance_mask`）；
  - 預測 = CellViT 在生成圖上找到的核（輪廓填滿，`_cellvit_json_to_instance_mask`）；
  - 兩組核一對一配對，IoU ≥ 0.5 才算配上；PQ = 配上的核的平均 IoU × 偵測 F1（`_compute_pq`）。1 = 一模一樣。
  - **注意：** 遮罩 PNG 是二值的，官方後處理切開但彼此相碰的核，畫成 PNG 後會連成一塊，讀回時變成一個核。這是論文流程本身的作法，所以主要讀數照做；另報一個不二值化的版本（§7.1 `pq_inst`）。
- **DICE：** 不分個別的核，只看「核像素」和標準答案的像素重疊多少（`_compute_binary_segmentation_metrics` 的第一個值）。
- **hematoxylin 讀數（不經過任何分割器）：** 把影像轉成 hematoxylin 光密度（hfv `stages/lin_typing/compare.py hematoxylin`），t = 0.0224（mask_truenuc_20261122 在 CRC16 用 Youden 規則定出的門檻，原樣沿用）。對同一張輸入遮罩：
  - (a) 遮罩內的像素中，hematoxylin ≥ t 的比例：生成器有沒有在遮罩說有核的地方畫出深染的核（像 precision）；
  - (b) hematoxylin ≥ t 的像素中，落在遮罩內的比例：深染的地方是不是都在遮罩裡，沒有在別處多畫核（像 recall）；
  - (c) 遮罩內平均光密度 − 遮罩外平均光密度：核和背景的對比。
  - 公式照 `mask_truenuc_20261122/scripts/truenuc.py` 的 `pix()`。
- **真實 H&E 參考線：** 真實 H&E 對它自己的遮罩。論文式 PQ 在不二值化的版本是 1（同一張圖、同一支程式、同一台 CPU 重跑，核完全相同）；二值化版本因為相碰的核被連起來會略低於 1；hematoxylin 讀數則是真實組織本身能達到的值。
- **外觀：** FUD（生成圖整批和真實圖整批在 UNI-2h 特徵空間的距離，一批一個數）、LPIPS（每張生成圖和同一張真實 H&E 的感知距離）、style_hed（H 和 E 兩個染色通道的平均與標準差的差），都越小越像真實，函式和 Fig 2 相同（§7.3）。

### 1.1 生成器、遮罩、管線（內文和圖只用左欄的名稱；代號只出現在程式、檔名和這張表）

| 內文名稱 | 是什麼 | 代號（程式／檔名） |
|---|---|---|
| PixCell 原版（PixCell off-the-shelf） | 公開的 PixCell-256 + Cell-ControlNet，沒有 TME 通道；PixCell 作者用 CellViT-SAM-H 的遮罩訓練 | `van`（build_cache `a2_off_shelf`） |
| CellViT 遮罩生成器：Concat／Grouped＋bypass／Per-channel／Grouped | 論文那一組（share-space 論文 Fig 2 的模型）：訓練遮罩 = hfv 舊後處理的 CellViT-256 在 CRC33 真實 H&E 上的核，舊的細胞分類 | `pcat`／`pgbp`／`ppch`／`pgrp`（`a1_concat`／`a2_bypass_full_tme`／`a1_per_channel`／`production`） |
| 形狀庫遮罩生成器：Concat／Grouped＋bypass／Per-channel／Grouped | 2026-11-18 重訓那一組（hfv_retrain_20261118）：訓練遮罩 = Lin 細胞位置＋形狀庫輪廓，CELESTA＋Lin 分類 | `rcat`／`rgbp`／`rpch`／`rgrp`（`tr_concat`／`tr_bypass`／`tr_perch`／`tr_grouped`） |
| 官方 CellViT 遮罩 | 真實 H&E 上官方後處理 CellViT-256 的核，畫成二值 PNG | `off` |
| hfv 舊後處理 CellViT 遮罩 | 論文生成時實際用的 `cell_masks`（hfv 舊後處理 CellViT-256，畫法相同） | `run` |
| 官方一致管線 | 官方 CellViT 遮罩輸入 → 生成 → 官方後處理找生成圖的核 | `off` 管線 |
| hfv 舊後處理一致管線（橋接） | hfv 舊後處理遮罩輸入 → 生成 → hfv 舊後處理找生成圖的核 | `run` 管線 |
| 形狀庫遮罩 | 形狀庫遮罩生成器原本的輸入遮罩（只用來轉貼類型標籤，§3.2） | R0 |

風格來源：一律是該圖塊自己的真實 UNI（own real UNI）。

## 2. 比較對象

- **圖塊：** 論文 Fig 2 那 100 張 CRC33 圖塊（S3 `share_space/04_inference/si_a1_a2/cache.json.bak_n100` 的 `metric_tile_ids`；應和 hfv `crc33sec_fig2_ids.txt` 同一組，閘門 §4）。這些是訓練用的圖塊，和論文一樣。論文 PixCell 原版另有 1,000 張的版本，這次只做 100 張。
- **主要組（論文那一組，有 checkpoint 的全部）：** PixCell 原版；CellViT 遮罩生成器的 Concat、Grouped + bypass、Per-channel；以及 Grouped（production）。Grouped（production）的 checkpoint 還在（S3 `03_checkpoints/concat_retrain_post_fix/full_seed_42` ← `si_a1_a2/production_retrain_post_fix`，build_cache.py 的 `production`），只有論文那批生成圖被覆蓋了，所以這次重新生成、一起比。
- **次要組（另外報，不和主要組混排）：** 形狀庫遮罩生成器的 Grouped、Concat、Grouped + bypass、Per-channel（`hfv_eval_20261118/ck/infer_e20/tr_*`，epoch 20）。它們是用形狀庫遮罩訓練的，**CellViT 遮罩不在它們的訓練分佈裡**；結果只描述它們拿到陌生遮罩時的表現，不拿來和主要組排名次。
- **每個生成器都跑兩條管線**（官方一致、hfv 舊後處理一致），共 9 個生成器 × 2 條管線 × 100 張 = 1,800 張生成圖。

### 2.1 已知的不對稱（照實寫進結果）

- PixCell 原版是 PixCell 作者用 CellViT-SAM-H 的遮罩訓練的。前一份實驗 `cellvit_samh_20261122` 在我們的圖塊上發現：官方後處理下 SAM-H 和 CellViT-256 幾乎相同（找到 Lin 真實核的比例差 +0.002）。所以官方 CellViT-256 遮罩大致就是 PixCell 原版熟悉的那種遮罩。
- CellViT 遮罩生成器是用 hfv 舊後處理的 CellViT-256 遮罩訓練的。官方遮罩把相碰的核切開，但畫成二值 PNG 後相碰的地方仍然連在一起，兩者的差別主要在**哪些像素被判成核**（正規化和 softmax 不同），所以官方遮罩對它們是「略微不同於訓練時」的遮罩。
- 所以官方一致管線對 PixCell 原版稍微有利，hfv 舊後處理一致管線對 CellViT 遮罩生成器稍微有利。兩條都報，並列。
- 形狀庫遮罩生成器兩條管線都不在訓練分佈裡（見上）。

## 3. 輸入

### 3.1 細胞核遮罩通道（`cell_masks`）

- **畫法和每個模型的訓練遮罩一樣：** 這次所有生成器的訓練遮罩都是二值 L 模式 PNG（0／255），相碰的核直接連在一起，**沒有**留 1 px 的間隙：
  - CellViT 遮罩生成器：把 CellViT 輪廓用 PixCell 的 `_cellvit_json_to_instance_mask` 填滿，> 0 的像素 = 255（hfv_retrain_20261118 README：這樣畫出的遮罩和論文的 `cell_masks` 逐像素相同，100／100）；
  - 形狀庫遮罩生成器：morphology_renderer `render_crop(shape="outline", allow_contact=True)` 畫成二值遮罩，允許相碰（lintype_render_20261118 README：3.8 % 的遮罩像素被兩個以上的核畫到，所以沒有間隙）；
  - PixCell 原版：PixCell 推論時把遮罩讀成二值（`load_channel(binary=True)`）。
  - （留 1 px 間隙的是 L1 遮罩，屬於重訓 2 的模型，不在這次比較裡。）
- **官方 CellViT 遮罩：** 用前一份實驗已經算好的官方後處理 JSON：`/gscratch/cheme/chiu/cellvit_official_20261122/full/cv/F2_P_gt/<圖塊>.json`（CellViT-256，Hyak CPU，`NVIDIA_TF32_OVERRIDE=0`，hfv 467a54f）。用 `_cellvit_json_to_instance_mask` 填滿輪廓，> 0 = 255，寫成 L 模式 PNG。**9 個生成器在官方一致管線裡拿到的是同一張遮罩。**
- **hfv 舊後處理 CellViT 遮罩：** 論文生成時實際用的那張 `cell_masks`（orion-crc33.zip，未修改）。9 個生成器在這條管線裡拿到的也是同一張。

### 3.2 細胞類型、狀態通道：把標籤轉貼到新的核上

類型（healthy、cancer、immune）和狀態（prolif、nonprolif、dead）通道原本是畫在「舊的核」上的：CellViT 遮罩生成器的舊核 = 論文 `cell_masks` 的核；形狀庫遮罩生成器的舊核 = 形狀庫遮罩的核。換成新的核以後，這幾個通道也要畫在新的核上，否則類型通道會蓋在沒有核的地方，或有核的地方沒有類型。每個生成器用**它自己原本的逐圖塊通道**當標籤來源：

- **新的核：** 官方遮罩 = 官方 JSON 填滿後的每個物件；hfv 舊後處理遮罩 = 論文 `cell_masks` 的 8 連通區塊（和 PQ 讀回標準答案的方式相同）。
- **配對規則（每個新核 O 各自判斷）：**
  1. 數 O 的像素中有幾個落在舊遮罩的核內（舊 `cell_masks` > 0）。**超過一半**才算配上（「多數落在舊核上」）；否則是**沒配上**的核。
  2. 配上的核：在 O ∩ 舊核 的像素裡，數每個舊類型通道各蓋了幾個像素，取最多的那個類型；狀態通道同樣獨立取最多的那個。同票時取通道順序中較前的（healthy、cancer、immune；prolif、nonprolif、dead），同票數另外記錄。如果這些像素上沒有任何類型（或狀態）通道是開的，那一組留空並記錄。
  3. 把 O 的**全部像素**畫進選到的類型通道和狀態通道（255）。
  4. 沒配上的核：只畫進 `cell_masks`，類型、狀態通道都不畫。
- **不需要轉貼的情況：** CellViT 遮罩生成器在 hfv 舊後處理一致管線裡拿到的就是原本的論文輸入，七個細胞通道全部照原檔，不轉貼（這就是論文的生成，§9 的閘門靠它）。
- **報告：** 每個（生成器組 × 管線）：新核總數、配上、沒配上（數目和比例）、同票數、留空數。
- **檢查：** 每個類型、狀態通道都包含在新的 `cell_masks` 內；每個配上的核恰好一個類型、一個狀態（留空的除外）。

### 3.3 其他一律不變

- 血管、氧、葡萄糖：每組用它原本 Fig 2 生成時的檔案（CellViT 遮罩生成器 = orion-crc33.zip；形狀庫遮罩生成器 = `evalpack_crc33sec.tar`），不改。
- UNI：每組用它原本 Fig 2 生成時的 `features/<圖塊>_uni.npy`。PixCell 原版用 orion-crc33.zip 的（論文就是這樣跑的）。兩份檔案是否相同會在 §4 記錄；不同也不停，各用各的。
- 生成設定：照論文 `si_a1_a2`（build_cache.py `_run_full`）：30 步、guidance 1.5、seed 42。PixCell 原版：PixCell 的程式沒有設 seed，這次照 hfv_retrain_20261118 的作法，用 `pxrun.py` 在每張圖之前設 `torch.manual_seed(42)`（那份 README 已記錄的偏離）。
- TF32：生成時不設定（和 Fig 2 的生成一樣，維持 torch 預設）；所有評分（CellViT、UNI、LPIPS）用 `NVIDIA_TF32_OVERRIDE=0`。

## 4. 資料來源與閘門（任一不符就停下查原因，不往下跑）

- 論文資料：S3 `share_space/02_datasets/orion-crc33.zip`（Fig 2 的 data root）；checkpoint：S3 `share_space/03_checkpoints/` 中 build_cache.py `INFERENCE_VARIANTS` 指到的那四個（Grouped production、Concat、Per-channel 用 `tillicum_95470/per_channel_95470_3` 和它自己的 `config.py`、Grouped + bypass），下載到 Hyak 新目錄（`scripts/pp_fetch.slurm`，作業 43466493；這一步只搬檔案，沒有產生任何讀數）。
- 形狀庫遮罩生成器的輸入：`/gscratch/cheme/chiu/hfv_eval_20261118/packs/evalpack_crc33sec.tar`（Fig 2 生成時用的那個評估包）。
- 真實 H&E：前一份實驗的 `F2_P_gt.zip`（= S3 `si_a1_a2/tiles/gt`）。
- **閘門：**
  1. `cache.json.bak_n100` 的 `metric_tile_ids`（100 張）= `crc33sec_fig2_ids.txt`（集合相同）。
  2. 這 100 張的真實 H&E：`F2_P_gt.zip` = stage-3 `R_crc33sec.zip`（逐像素）。
  3. 官方 JSON 100 個都在；論文 `cell_masks`、評估包的七個細胞通道 100 張都在，而且都是 0／255。
  4. （不設閘門，只引用）論文 `cell_masks` = 用 PixCell 畫法重畫 hfv 舊後處理 JSON 的結果，逐像素相同 100／100（hfv_retrain_20261118 README 已記錄）。這次另外記錄：Hyak 上重跑的舊後處理 JSON（`full_runner/cv/F2_P_gt`）重畫後和論文 `cell_masks` 逐像素相同的張數。
  5. checkpoint 的 md5 記錄下來；Per-channel 的 `config.py` 必須能被 PixCell 的 `read_config` 讀進來。
  6. §3.2 的檢查。

## 5. 生成（`scripts/pp_gen.py`、`scripts/pp_gen.slurm`）

- 四個生成作業，各一張 GPU：CellViT 遮罩生成器組 × {官方, 舊後處理}（PixCell 原版也在這兩個作業裡跑，因為它用論文的 data root）、形狀庫遮罩生成器組 × {官方, 舊後處理}。同一個比較組的所有生成都在同一種 GPU 上。
- 程式：PixCell cb62fa7 的 `tools/ablation_a1_a2/build_cache.py` 原樣執行（經 `pxrun.py`）。它寫死的 checkpoint 路徑和 `data/orion-crc33` 用 symlink 影子目錄提供（和 hfv_retrain_20261118 `eval/evalgen.py` 的作法相同），PixCell 檔案不改。
- **放在哪裡（依現在量到的時間）：** 
  - Hyak ckpt：2026-11-22 13:35 `hyakalloc` 顯示 ckpt 有 43 張閒置 GPU；hfv_retrain 實測 A100 30 步穩定 33.0 張／分鐘（作業 43390518），每個作業第一支指令要先等約 8 分鐘。每個作業 400–500 張，約 15 分鐘＋載入。checkpoint 已在 Hyak（共約 40 GB），不用搬。
  - Colab L4：實測 25.2 張／分鐘、新 session 準備約 10 分鐘，但要先把四個 10 GB 的 checkpoint 從 S3 搬過去，而且最多兩個 session。
  - 所以先送 Hyak ckpt A100。若任一作業在 ckpt 排隊超過 1 小時，就照規則改到 Colab L4（hfv_eval_L4_a／b），Hyak 那邊留在佇列，先完成的算數；這種情況會在結果裡記錄。
- 生成後在同一個作業、同一張 GPU 上算外觀（§7.3），評分程序設 `NVIDIA_TF32_OVERRIDE=0`。

## 6. CellViT（Hyak cheme CPU）

- 每個（生成器 × 管線）的 100 張生成圖，**兩種後處理都跑**：一致的那一種是主要讀數；另一種（「交叉」）只描述，用來分開「輸入遮罩換了」和「評分端換了」各自的影響。
- 程式：`cellvit_official_20261122` 的 `cvo_infer.slurm` 同一套設定（CellViT-256、CellViT repo 05097e1、hfv 467a54f 凍結複本、batch 32、原解析度、`NVIDIA_TF32_OVERRIDE=0`、cheme 8 核），改成讀這個目錄的 zip（`scripts/pp_cv.slurm`）。
- 真實 H&E 的 CellViT 用前一份實驗的結果：官方 `full/cv/F2_P_gt`、舊後處理 `full_runner/cv/F2_P_gt`（同一支程式、同一種 CPU）。

## 7. 讀數（`scripts/pp_score.py`，Hyak CPU；外觀在 GPU 作業裡）

每張圖塊、每個生成器、每條管線：

### 7.1 照遮罩畫的程度

- `pq`：論文式 PQ（標準答案 = 該管線的輸入遮罩 PNG 的 8 連通區塊；預測 = 一致後處理的生成圖核）。**主要讀數。**
- `pq_inst`：同上，但官方管線的標準答案改用官方 JSON 填出的實例圖本身（不二值化）；舊後處理管線兩者相同。
- `dice`：同一對的 DICE。
- `a`、`b`、`c`：生成圖對輸入遮罩的 hematoxylin 讀數（t = 0.0224）。
- 描述：輸入遮罩的核數、生成圖找到的核數、交叉後處理的 `pq_x`。
- 真實 H&E 參考線：同樣的量，把生成圖換成真實 H&E（預測端用前一份實驗的真實 H&E CellViT）。

### 7.2 外觀

- LPIPS（PixCell `metrics_io._load_lpips`、`_lpips_pair`，AlexNet）、style_hed（`metrics_io.compute_and_update_metrics` 裡的 HED 寫法），每張圖塊一個值。
- FUD：`metrics_io._compute_fud_by_variant` 同樣的步驟（真實端 = 該 data root 存的 `features/<圖塊>_uni.npy`；生成端用 UNI-2h 重算，前處理照 hfv_retrain README「UNI-2h 前處理」，經 `pxscore.py`）。一批一個數，沒有區間。

## 8. 比較與排序規則

### 8.1 區間

圖塊 bootstrap，10,000 次，`numpy.random.default_rng(42)`，95 % 百分位區間。同一組重抽矩陣用在所有生成器、所有管線（配對）。不做多重比較校正；區間只用來判斷「分得開／分不開」。

### 8.2 要報的配對差（每條管線各一次）

- 每個生成器 − PixCell 原版（主要組 4 個、次要組 4 個）。
- CellViT 遮罩生成器：Concat − Grouped + bypass、Concat − Grouped（production）。
- 形狀庫遮罩生成器：Concat − Grouped + bypass、Concat − Grouped（次要）。
- 讀數：`pq`、`dice`、`a`、`b`、`c`、LPIPS、style_hed（FUD 只有點估計）。

### 8.3 排序規則（不設數字門檻）

1. **Fig 2 的數字：** 官方一致管線的論文式 PQ（`pq`），平均和 95 % 區間；hfv 舊後處理一致管線的並列。
2. **主要組的排序：** 主要組 5 個生成器兩兩比較（10 對），只用官方一致管線的 `pq`：
   - A − B 的區間整段 > 0 → 「A 高於 B」；整段 < 0 → 「A 低於 B」；跨過 0 → 「分不開」。
   - 論文只能寫「分得開」的那些先後；分不開的寫成並列。
3. **一致性標記：** 每一對「分得開」的先後，再看 `dice` 和 hematoxylin 讀數 (a)、(c)：三個都同方向、區間都不跨 0 → 標「各讀數一致」；否則標「只有 PQ」。(b) 只描述。
4. 次要組照同樣規則排它自己的 4 個生成器加 PixCell 原版，另外報，不和主要組混排。
5. 兩條管線的排序並列；如果兩條管線給出相反的「分得開」先後，照實報告，Fig 2 的建議仍以官方一致管線為準（理由：§0，兩端的核定義一致，而且是 CellViT 原作者的流程），但結果裡要寫明這個前提。

### 8.4 給 sim2real 的標記

如果在官方一致管線下，任一個 Grouped 架構（Grouped 或 Grouped + bypass，主要組或次要組）對同組的 Concat 是「高於」，而且 `pq`、(a)、(c) 三者的區間都整段 > 0 → 標記「這個 Grouped 架構照遮罩畫得明顯比 Concat 好」，交給 sim2real 生成器的問題參考（那邊的判斷用 Lin 遮罩，見 `fidelity_variants_20261122`）。否則不標記。

## 9. 閘門：hfv 舊後處理一致管線要重現論文的數

- hfv 舊後處理一致管線裡，CellViT 遮罩生成器和 PixCell 原版拿到的就是論文的輸入、用論文的程式和設定生成、用論文的 PQ 程式評分。所以它應該重現論文公布的 Fig 2 PQ（100 張：PixCell 原版 0.6987、Concat 0.7817、Grouped + bypass 0.7943、Per-channel 0.0106）。
- **規則：** 這四個公布值都必須落在重新生成後 `pq` 的 95 % 區間內。任一個不在 → 停下：官方一致管線的結果先不解讀，查原因後回報。
- 描述（不是門檻）：重新生成的圖和論文的圖（S3 `si_a1_a2/tiles/<變體>`）逐像素相同的張數；不同的話每張的 LPIPS。PixCell 原版論文那次沒有設 seed，所以不期待逐張相同。Grouped（production）論文寫的是 0.5694，但那批圖已被覆蓋（cellvit_official PREREG §5.2），只並列、不設閘門。

## 10. 執行規則

- Hyak 只寫新目錄 `/gscratch/cheme/chiu/paperpq_official_20261122/`，其他目錄唯讀。CPU 用 cheme，留 120 核；GPU 用 ckpt A100（或 §5 的 Colab L4，Tillicum 不用）。
- 任何失敗：停下診斷，不原樣重送；改了什麼寫在結果的偏離紀錄。
- Colab（如果用到）：只用 hfv_eval_L4_a／b，最多兩個 L4 session；一天有效的 presigned 連結；不放 AWS 長期金鑰或 HF token；登入由使用者點；結束後「中斷連線並刪除執行階段」並確認 0 個 session。
- 結果寫在 `RESULTS.md`，附圖（英文、Nat Commun 格式，sim-exp-validation `figstyle.py`）和 `data/`。
