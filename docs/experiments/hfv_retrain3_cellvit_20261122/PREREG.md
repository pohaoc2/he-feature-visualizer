# hfv_retrain3_cellvit_20261122 預先登記：四個生成器變體改用官方 CellViT 細胞核遮罩重訓，再重跑 Fig 2 評估

寫於 2026-11-22，**在這個實驗的任何訓練、任何新影像或新數字出現之前**。之前已經看過的數字只有：重訓 1（`hfv_retrain_20261118`）、重訓 2（`hfv_retrain2_20261121`）、`cellvit_official_20261122` 和 sim-exp-validation `mask_truenuc_20261122` 已公布的結果。

## 1. 為什麼要做，以及這個實驗改變哪個決定

- **使用者決定（2026-11-22）**：「hfv不要用Lin的mask, 都用CellViT的 然後選B重訓」。
  - hfv 這篇（生成器論文）改成和 PixCell 原論文一樣：細胞核遮罩 = CellViT 在**同一張真實 H&E** 上找到的細胞核。
  - sim2real 那篇保留自己的生成器（Lin 分割的遮罩），**不在這次範圍內**。
- **為什麼遮罩來源重要**：PixCell 的「論文式 PQ」是拿生成圖上 CellViT 找到的細胞核，去和一份「標準答案」遮罩一顆一顆配對。只有當「餵給生成器的遮罩」本身就是「標準答案」，而且兩邊用同一種 CellViT 後處理時，這個數才回答「生成器有沒有照著遮罩畫細胞核」。重訓 1、2 的輸入遮罩是 Lin 的螢光分割，標準答案卻是真實 H&E 的 CellViT，兩者本來就對不上（連真實 H&E 自己都只有 0.12–0.17），所以那個數沒有解讀。這次讓輸入遮罩 = 標準答案，論文式 PQ 才有意義，也能和 PixCell 原版直接比。
- **改變的決定**：hfv 論文 Fig 2 用哪一組生成器、以及 Fig 2 的兩個結論——(1) 四種 TME 接法（Concat、Grouped、Grouped + bypass、Per-channel）誰排前面；(2) 15 條件消融裡哪幾組通道對「照遮罩畫」和「外觀」有幫助。使用者已決定 Fig 2 改用這一組；這個實驗的結果直接決定 Fig 2 的排名和消融那兩張熱圖寫什麼。

## 2. 名詞（內文和圖上只用這些名字；代號只出現在 §9 對照表、檔名和程式）

| 寫法（英文／中文） | 指的是 |
|---|---|
| official-CellViT mask／官方 CellViT 遮罩 | CellViT-256 用原作者的後處理（見 §3.1）在真實 H&E 上找到的細胞核，畫成二值遮罩，相碰處留 1 px 縫 |
| official-CellViT-mask generators／官方 CellViT 遮罩生成器 | 這次重訓的四個變體（Concat、Grouped、Grouped + bypass、Per-channel），訓練時的細胞核遮罩是官方 CellViT 遮罩，類型轉貼自 CELESTA + Lin Fig. 3d，血管／氧／葡萄糖用 sim2real 的模擬式算法 |
| PixCell off-the-shelf／PixCell 原版 | 公開的 PixCell-256 加公開的 Cell-ControlNet，沒有 TME 通道、沒有用我們的資料微調（論文和程式裡叫 Vanilla） |
| own real UNI／該圖塊自己的真實 UNI | 生成時的風格條件向量 = 同一張真實圖塊的 UNI-2h 特徵 |
| paper-style PQ／論文式 PQ | 見 §5.2；這次輸入遮罩 = 標準答案，所以才報這個數 |
| runner／official post-processing／hfv 舊後處理／官方後處理 | 所有 CellViT 數字都註明是哪一種；這次全部用官方後處理 |

Concat、Grouped、Grouped + bypass、Per-channel 是「TME 通道怎麼接進生成器」的架構名，照用。

## 3. 訓練輸入（和重訓 1 相比，只改下面列出的東西）

### 3.1 細胞核遮罩：官方 CellViT 遮罩

- **偵測**：hfv `stages/run_cellvit_local.py --postprocess official --magnification 40`（467a54f，`cellvit_official_20261122` 用的同一份），CellViT repo 05097e1，`CellViT-256.pth`（md5 b3634bd5），batch 32，原始解析度 256 px，Hyak cheme CPU，`NVIDIA_TF32_OVERRIDE=0`。
  - 官方後處理 = 正規化照 checkpoint 設定（0.5／0.5）、softmax、用水平／垂直距離圖做分水嶺，**把黏在一起的細胞核切開**（`cellvit-runner-postprocessing`）。
  - 每一張訓練圖塊和評估圖塊的真實 H&E 都跑：CRC33_01 9,418 張、CRC36 訓練 16,651 張、CRC36 保留區 1,092 張。
- **畫法（重訓 1 怎麼處理相碰的核，以及這次選的規則）**：
  - 重訓 1 的形狀庫遮罩：相碰的核直接連成一塊（`allow_contact=True`），二值化後每個 Lin 細胞平均只剩 0.48–0.73 個連通區塊（重訓 2 RESULTS_stage1 D4）。
  - 論文時期的舊遮罩：hfv 舊後處理本來就是「取連通區塊再描外輪廓」，每個物件之間天然至少隔 1 px；所以二值化後重新取 8 連通區塊，得到的正好是原本的物件（重訓 1 README：真實 H&E 的 CellViT 對舊遮罩 PQ = 1.0000）。
  - PixCell 原論文（arXiv 2506.05127 App. A.7.1）：Cell-ControlNet 的條件是 CellViT-SAM-H 細胞核的**二值**遮罩。
  - 官方後處理的分水嶺會讓相鄰的核**直接相碰**，直接二值化會把它們連回一塊。
  - **規則（選一條）**：
    1. 用 PixCell 的 `_cellvit_json_to_instance_mask` 把官方 CellViT JSON 的輪廓填成實例圖（和生成圖打分時用的是同一個函式）；
    2. 套 sim-exp-validation `separate_instances`（重訓 2 L1 用的同一個共用函式：一個像素若八鄰域碰到編號較小的非零標記，就清成背景），讓相碰的核之間留 1 px 縫；
    3. 存成 0／255 的二值 PNG。
  - 結果：二值遮罩的 8 連通區塊數 = 官方 CellViT 的實例數（建置閘門，§3.6）。這和論文舊遮罩「一個物件一塊」的性質相同，也和 PixCell 讀遮罩的方式（`_load_gt_instance_mask`：> 0 再取 8 連通區塊）一致。

### 3.2 細胞類型與狀態：CELESTA + Lin Fig. 3d，轉貼到官方 CellViT 細胞核上

- 類型和狀態都來自共用的分類函式（`stages/lin_typing/labels.py`、`celltype_celesta_fig3d.py`），沿用重訓 2 已算好的每個 Lin 細胞標籤（CRC33_01：`hfv_lintype_20261118/out/cells_labels.npz`；CRC36：`hfv_crc36_20261118` 的同名檔，重訓 2 的 L1 通道就是從這兩份畫的）。
  - 類型：healthy／cancer／immune = CELESTA 的判定，Unknown 由 Lin 112 µm 的 H&E 組織分類補（Lin 2023 Fig. 3d）。
  - 狀態：先看死細胞（`labels.py` 的規則：Lin 細胞內含有 CellViT 死細胞物件的質心），再看 Ki-67 是否高於 Lin 的閘值（增殖／不增殖）。死細胞規則和重訓 2 相同，不重新推導（記憶 `dead-channel-empty`：這個通道幾乎是空的）。
- **轉貼規則**：每個官方 CellViT 細胞核（留縫後的像素），找和它重疊像素最多的 Lin 細胞（Lin **整個細胞**遮罩，`*-mask.ome.tif`，背景不算）；平手取 CellID 較小的。這個細胞核就畫進那個 Lin 細胞的類型通道和狀態通道。
  - 和任何 Lin 細胞都不重疊的細胞核：只畫進 `cell_masks`，不畫類型和狀態。
  - **絕不用 CellViT 自己的類型輸出**：那會讓「生成圖的 CellViT 類型」和「輸入的類型」來自同一個模型，變成循環論證。
- **要報告的數字**：
  - 沒配到 Lin 細胞的細胞核比例（以個數計、以像素計），兩張切片分開；
  - 質心落在圖塊內的 Lin 細胞中，沒有被任何細胞核選中的比例，分 healthy／cancer／immune；
  - 一個 Lin 細胞被幾個細胞核選中的分布（0、1、≥ 2）。
  - 這些是描述，不是門檻。

### 3.3 血管、氧、葡萄糖：sim2real 方法 C

- 逐位元組沿用重訓 2 建好的 `envc/crc33/envc_crc33.tar`、`envc/crc36/envc_crc36.tar`（AF1 校正的 contract 中心線 8 µm、整張切片 PDE 的氧和葡萄糖）。理由：每張圖塊各自 Otsu 的 CD31 斑點不是血管圖（記憶 `hfv-concat-legacy-vessel-channel`）。

### 3.4 資料

- 切片和切分和重訓 1、重訓 2 相同：CRC33_01 9,418 張（墨水 tile 已排除，`tiles_keep.txt`）+ CRC36 16,651 張 = 26,069 張；CRC36 保留區 1,092 張和緩衝帶 129 張不參加訓練；CRC34 完全不用。重訓 2 沒有改切分。
- UNI 特徵和 H&E 的 VAE latent 逐位元組沿用（`c36train.tar`、`orion-crc33`）。
- 遮罩的 VAE latent：用新的 `cell_masks` 重算，程式和指令同重訓 2 `masklat.slurm`（PixCell `stage1_extract_features.py --vae-prefix mask_sd3_vae --skip-uni`，Hyak ckpt GPU）。

### 3.5 變體與訓練

- 四個變體：Concat、Grouped、Grouped + bypass、Per-channel；PixCell cb62fa7 的程式和設定不改；hfv `train_wrap.py`；有效批次 80（micro 80 × 累積 1）、20 個 epoch、seed 42、不設定 TF32；每個變體 6,520 步。和重訓 1 相同；重訓 2 沒有改超參數。
- **checkpoint 的選法**：和重訓 1、2 相同，用第 20 個 epoch 結束時的 checkpoint（`step_0006520`），不依任何評估結果挑。
- **Tillicum**：每個變體一張 H200，`--no-requeue`，時間上限 4 小時（重訓 1 的上限）。

### 3.6 建置閘門（任何一條不過就 STOP，不往下做）

1. 每張圖塊都有官方 CellViT JSON；JSON 數 = 圖塊數。
2. 重現：這次對 CRC36 保留區 300 張和 CRC33 Fig 2 那 100 張跑出的官方 CellViT，和 `cellvit_official_20261122` 已存的真實 H&E 官方結果（`R_crc36ho`、`R_crc33sec`）比，逐張實例圖相同的比例 ≥ 該實驗自己報的重跑一致程度（它的重訓組是 298–300／300、100／100）。不符就停下回報，不往下建。
3. 每張圖塊：二值遮罩的 8 連通區塊數 = 留縫後的實例數；類型三個通道互斥、聯集 = 有配到 Lin 的細胞核；狀態三個通道同樣互斥、聯集相同；所有值只有 0／255。
4. 每個訓練 id 都有 10 個通道和遮罩 latent，各一次；打包後每個成員重讀，sha256 相同（重訓 2 `pack.py` 的閘門）。
5. Tillicum 準備：三個 tar 的 md5 = 打包時記下的值；10 個通道各 9,418 張；索引 26,069 筆；遮罩 latent 26,069 個；舊遮罩 latent 0 個（重訓 2 `setup2.sh` 的閘門）。

## 4. Tillicum 花費：送出前的預估和停止規則

- **2026-11-22 13:40 實查**（`hyakusage -s 2026-10-09`）：pohaoc2 已用 $27.20，上限 $40，**剩 $12.80**。實驗室帳號 bagherilab 已用 $67.20／$90（剩 $22.80，是外層上限）。
- **計費 = 執行時間 × $0.90／H200 小時**。驗算：11-18 時已用 $17.95，加上重訓 1 四個作業（1:45:39 + 2:09:33 + 2:09:12 + 2:27:01）和重訓 2（1:45:06）共 10.27 小時 × $0.90 = $9.25，合計 $27.20，和 hyakusage 一致。
- **預估（照同一變體最近一次的實測時間）**：

  | 變體 | 實測（作業） | 預估花費 |
  |---|---|---|
  | Concat | 1:45:06（509534，重訓 2）；1:45:39（498232，重訓 1） | $1.59 |
  | Grouped | 2:09:33（498308） | $1.94 |
  | Grouped + bypass | 2:09:12（498309） | $1.94 |
  | Per-channel | 2:27:01（498310） | $2.21 |
  | 合計 | | **$7.68**（< $12.80） |

- **規則**：
  1. 先只送 Concat。跑完後用 `hyakusage` 讀實際花費。
  2. 用「Concat 實測時間 ÷ 重訓 1 Concat 時間」的比例放大其他三個的重訓 1 時間，算出新預估。新預估加上已用的錢若超過 $40（或實驗室剩餘額度），**STOP 回報，不刪減計畫**。
  3. 沒超過才同時送出其他三個。
  4. 任何作業失敗：STOP、診斷，不原樣重送。撞到時間上限才用 `--fetch-resume` 接續，並記在偏離紀錄。

## 5. 評估（兩邊都用官方 CellViT）

### 5.1 圖塊與生成

- **主要組**：CRC36 保留區 300 張（`crc36ho_p300_ids.txt`，重訓 1 `eval/picks.py` 抽的同一批）。沒有參加訓練。
- **次要組**：論文 Fig 2 那 100 張 CRC33 tile（`crc33sec_fig2_ids.txt`）。這些是**訓練 tile**，只用來和論文比較。
- **五個模型**：四個官方 CellViT 遮罩生成器 + PixCell 原版。全部餵同一份官方 CellViT 遮罩（PixCell 原版只吃 `cell_masks` 和 UNI）和該圖塊自己的真實 UNI。
- **生成**：重訓 1 的 `eval/evalgen.py fig2` 任務，不改：PixCell `build_cache.py`，30 步、guidance 1.5、seed 42；PixCell 原版在每張 tile 前設 `torch.manual_seed(42)`（`pxrun.py`，重訓 1 的偏離照用）；接著 `sensitivity_eval.py --mode render` 產生「拿掉一組通道」的圖（程式定義的 ΔLPIPS 用）。
- 評估包：重訓 1 的 `evalpack_crc36ho`／`evalpack_crc33sec`，只換三樣：7 個細胞通道（§3.1–3.2 的建置）、ch7–9（§3.3）、遮罩 latent（§3.4）。UNI、H&E、H&E latent 不變。
- 一個比較組（同一個評估集的五個模型）從頭到尾在同一種 GPU 上生成；GPU 型號寫進輸出。生成在 Hyak ckpt GPU 或 Colab L4，**不在 Tillicum**，依當下實測時間和佇列決定（§7）。

### 5.2 讀數

所有 CellViT 都是官方後處理，設定同 §3.1，在 Hyak cheme CPU 上跑（生成圖和真實 H&E 用同一台類型的機器、同一份程式）。

1. **論文式 PQ 和 DICE**：PixCell `_compute_pq`（IoU ≥ 0.5 才算配對；PQ = 配對比例 × 配對核的平均 IoU）和 `_compute_binary_segmentation_metrics`（DICE = 細胞核像素的重疊，不分個別核）。
   - 標準答案 = 輸入遮罩，用 PixCell `_load_gt_instance_mask` 讀（二值 → 8 連通區塊），也就是**該 tile 真實 H&E 的官方 CellViT**。
   - 預測 = 生成圖的官方 CellViT JSON，用 `_cellvit_json_to_instance_mask` 填成實例圖。
   - 問的是：「生成圖上找得到的細胞核，是否落在真實組織有細胞核的位置、形狀是否相近」。
   - 參考線（健全性檢查）：真實 H&E 自己的官方 CellViT 對同一份輸入遮罩的 PQ／DICE。只因為 1 px 縫而略低於 1。
2. **不靠分割器的 hematoxylin 讀數（對輸入遮罩）**：sim-exp-validation `mask_truenuc_20261122/scripts/truenuc.py` 的 `pix()`，蘇木素光密度用 `stages.lin_typing.compare.hematoxylin`，門檻 t = 0.0224（該實驗在 CRC16 用 Youden 規則訂的值）。
   - (a) 遮罩像素中 hematoxylin ≥ t 的比例（像 precision：畫在遮罩上的地方是不是真的深染）；
   - (b) hematoxylin ≥ t 的像素中被遮罩蓋到的比例（像 recall：深染的地方是不是都在遮罩上）；
   - (c) 遮罩內平均光密度 − 遮罩外平均光密度（不需要門檻）；
   - 另外報像素 F1。
   - 參考 = 真實 H&E 對同一份遮罩的同一組讀數；報「生成 − 真實」的逐 tile 配對差。
3. **外觀**：
   - LPIPS（AlexNet）：生成圖和同一張真實 H&E 的感知距離，越小越像；
   - style_hed：H、E 兩個染色通道的平均和標準差差多少（`fig2tiles.py` 的寫法），越小越像；
   - FUD：整批生成圖和整批真實圖在 UNI-2h 特徵空間的距離（PixCell `compute_fid_from_stats`；UNI-2h 前處理用 `uni-2h/config.json`，同重訓 1）。一批只有一個數，**沒有逐 tile 的區間，只描述**。
4. **ΔLPIPS 兩種定義，分開報告，四個變體都用同一批 tile**：
   - 文字定義：每張 tile 算 LPIPS(PixCell 原版, 真實) − LPIPS(變體, 真實)；正值 = 變體比原版更像真實 H&E。
   - 程式定義：PixCell `sensitivity_eval.py`，完整條件生成的圖和「把某一組 TME 通道歸零」後生成的圖之間的 LPIPS，衡量模型有多依賴那組通道。

### 5.3 統計

- 每個評估集內，對 tile 做配對 bootstrap：10,000 次、seed 42、95 % 百分位區間；同一個評估集的所有模型、所有讀數用同一組重抽索引，所以兩模型的差是配對差（`evalstats.py` 的慣例）。
- 「有差別」= 區間不含 0；含 0 = 分不出。不做多重比較校正（重訓 1、2 的慣例），照實列出所有區間。
- **事先宣告的比較**（每個評估集、§5.2 的讀數 1–3 和文字定義的 ΔLPIPS）：
  1. 每個變體 − PixCell 原版（4 個）；
  2. Concat − Grouped、Concat − Grouped + bypass。
- 排名規則需要的其他兩兩差（共 10 對）也算，但只用於 §5.4。

### 5.4 排名規則（不含任何數字門檻）

- 只用主要組（CRC36 保留區）和四個讀數：論文式 PQ、DICE（照遮罩畫）；LPIPS、style_hed（外觀）。
- **A 排在 B 前面**，若且唯若：四個讀數中至少一個的 A − B 區間整個落在「A 較好」那一側，而且沒有任何一個落在「B 較好」那一側。
- 兩者各有勝場（例如 A 的 PQ 較好、B 的 LPIPS 較好）：記為「取捨」，不排先後；全部分不出：記為「並列」。
- 照這個關係排出一個（可能不完整的）順序，連同所有取捨、並列照實寫出。
- FUD、hematoxylin 讀數、兩種 ΔLPIPS、次要組不進規則，只報告；另外說明次要組的順序是否相同。

### 5.5 消融（Fig 2e/g）：新的 Concat，照重訓 1 的程序

- 重訓 1 `eval/evalgen.py abl` 任務，不改：PixCell `generate_ablation_subset_cache.py`，15 個通道群組組合（cell_types、cell_state、microenv、vasculature 四組的所有非空組合；`cell_masks` 永遠開著），20 步、guidance 2.5、seed 42；paired（自己的 UNI）和 unpaired（風格來自別張 tile，重訓 1 的 derangement）。
- 圖塊：CRC36 保留區 300 張（主要；unpaired 用 `crc36ho_p300_unpaired.json`）和 CRC33 299 張（次要、訓練 tile、可和論文比；`crc33sec_p300_ids.txt`、`crc33sec_p300_unpaired.json`）。n 事先固定，不依結果加。CRC34 不在這次範圍內。
- 每個條件的基準圖都在同一次執行裡重新生成。
- 打分：官方 CellViT（§5.2）→ PixCell `compute_ablation_metrics.py`（PQ／DICE 的標準答案 = 該 tile 自己的輸入遮罩 = 官方 CellViT 遮罩；unpaired 也是自己的遮罩）、LPIPS、HED、FUD。
- 效果量 = PixCell `summarize_added_group_effect_stats`：對其他組的每一種組合，比較「加入這組」和「不加」的差再平均；正值 = 加入這組讓指標變好。區間同 §5.3（tile bootstrap 10,000 次、seed 42）。
- Fig 2d 的 PixCell 原版參考帶（FUD、PQ、DICE）用這次同一評估集的 PixCell 原版數值（重訓 1 的作法）；兩條真實 H&E 參考帶（LPIPS、HED）照用。

## 6. 報告

- 文件放在這個資料夾；先 push 這份預先登記，再做任何訓練或產生任何新數字。
- 輸入建好後 push 一份階段報告：§3.2 的配對數字、6 張 tile 的接觸表（H&E | 官方 CellViT 遮罩 | 類型通道；從細胞數高於中位數的 tile 用 seed 42 抽，每張切片 3 張）、Tillicum 花費預估。
- 最後 `RESULTS.md`：圖（英文、Nat Commun 格式、sim-exp-validation `figstyle.py`）和數據。
- 內文和圖上不用代號；「主要 PQ」這個說法不用（這次只有「論文式 PQ」，因為輸入遮罩 = 標準答案）。

## 7. 計算資源

- **訓練**：Tillicum（§4）。
- **生成**：Hyak ckpt GPU（l40s／a100）或 Colab L4（命名筆記本 `hfv_eval_L4_a`／`_b`，最多 2 個 L4 session），依當下實測速度和佇列決定，決定時把數字和來源寫在旁邊。重訓 1 的實測：Colab L4 30 步 25.16 張／分鐘、20 步消融 84.43 張／分鐘；Hyak ckpt A100 33.04 和 198.5 張／分鐘，但每個 job 第一支指令前約 8 分鐘載入。Colab 只用一天有效的 presigned 連結，不放 AWS 長期憑證和 HF token；登入一律由使用者點；用完「中斷連線並刪除執行階段」並確認 0 個 session。
- **CPU**（CellViT、建輸入、打分）：Hyak cheme，留 120 個核心給其他人。
- **Hyak 資料夾**：只用新的 `/gscratch/cheme/chiu/hfv_retrain3_cellvit_20261122/`。不寫入其他實驗的資料夾（包括平行在跑的 `paperpq_official_20261122`）。
- 不改 PixCell，不碰 share-space-paper。

## 8. 停止條件

- 任何閘門不過、任何作業失敗：STOP、診斷，不原樣重送。
- ssh 失敗：回報需要使用者執行 `! hpc-connect hyak` 或 `! hpc-connect tillicum`，不重試登入。
- Tillicum 預估超過剩餘額度：STOP 回報，不刪減計畫（§4）。

## 9. 代號對照表（只在這裡、檔名和程式裡出現）

| 代號 | 內文寫法 |
|---|---|
| OC | 官方 CellViT 遮罩（細胞核遮罩與其轉貼類型的建置模式） |
| tr3_concat、tr3_grouped、tr3_bypass、tr3_perch | 官方 CellViT 遮罩生成器：Concat、Grouped、Grouped + bypass、Per-channel |
| a2_off_shelf、Vanilla | PixCell 原版 |
| a1_concat、a1_grouped、a1_bypass、a1_perch | `build_cache.py` 裡四個變體的鍵 |
| crc36ho、crc33sec | CRC36 保留區（主要組）、CRC33 次要組 |
| p300、fig2 | 消融的 300（CRC33 為 299）張、Fig 2 的 100 張 |
| R0、L1、K、N | 重訓 1、2 的形狀庫遮罩、Lin 寬鬆核形、形狀庫遮罩生成器、真實核形 + 模擬血管生成器（只在引用舊實驗時出現） |
