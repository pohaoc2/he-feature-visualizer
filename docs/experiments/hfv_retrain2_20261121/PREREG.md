# hfv_retrain2_20261121 預登記：Concat 重訓第 2 版（細胞核遮罩對上 H&E ＋ sim2real 方法 C 環境通道）

寫於 2026-11-21，**在這個實驗的任何新數字出現之前**。之前已經看過的數字只有 `hfv_retrain_20261118` README 裡的結果（PQ 參考線、對 Lin 的檢測 F1、應用 8 的 Δ）。

## 1. 為什麼要做

使用者在 2026-11-21 決定重訓 Concat 一次，同時修兩件事：

1. **細胞核遮罩要對上真實 H&E 上看得到的核。**
   - 現在的訓練遮罩（`lintype_render_20261118`）是用 sim2real 的 renderer 畫的：
     - 每個 Lin 細胞在它的 Lin 中心點蓋一個「外形庫」裡隨機挑的輪廓（同類型、面積相近）；
     - 輪廓不是這顆核自己的形狀，方向也不同；
     - 相鄰的核可以相碰（`allow_contact=True`），二值化以後會連成一塊。
   - 這是 2026-11-18 的設計，目的是讓真實與模擬的遮罩用同一個畫法。
   - 代價是遮罩和目標 H&E 上的核只部分重合：真實 H&E 對這個遮罩的 PQ 只有 0.12–0.17。
   - 結果模型學到「遮罩只是大概的提示」：訓練過的模型對 Lin 的 F1 比 Vanilla 低 0.11–0.15。
2. **ch7–9 改用 sim2real 方法 C。**
   - 血管：扣掉 AF1 自發螢光的 contract 中心線，畫成 8 µm 寬。
   - 氧、葡萄糖：用整張切片的 PDE，以 contract 線為血管來源。
   - 取代 hfv 每張圖塊各自 Otsu 的 CD31 遮罩。那種遮罩模擬做不出來，十月已判定不是血管圖（`vessel_contract_20261006`、`vessel_af_validation_20261011`）。

**這個實驗改變的決定：** sim2real 從模擬生成時用哪一個 Concat——現在的 `tr_concat/step_0006520`（K），還是重訓的新模型（N）。

## 2. 第 1 階段（CPU，不花 GPU／Tillicum）：診斷 → 選遮罩來源

### 2.1 資料

只用已經有真實 H&E 分割的評估圖塊（兩種分割器都已跑過，直接沿用）：

| 組 | 切片 | 圖塊 | Lin 資料 |
|---|---|---|---|
| crc33sec | CRC33_01 | `res2/crc33sec/real.csv` 的圖塊 | `hfv_lintype_20261118/out` + `he_celltype_audit_20261118/lin/CRC33_01__…-mask.ome.tif` |
| crc36ho | CRC36 | `res2/crc36ho/real.csv` 的 300 張 | `hfv_crc36_20261118/out` + `hfv_crc36_20261118/lin/data__CRC36__P37_S79-CRC36-mask.ome.tif` |

- 這些不全是訓練圖塊（crc36ho 是保留區），但切片、Lin 分割、繪製流程都和訓練資料相同。
- 真實 H&E 的分割：
  - CellViT：`realcv_<set>` JSON，用 PixCell 的 `_cellvit_json_to_instance_mask` 畫成實例；
  - StarDist：`hfv_robust_20261120/res2/<set>/sd_*.npz` 的 `real/<set>/<tile>.png`。
- 訓練樣式的遮罩：評估包 `evalpack_<set>.tar` 的 `exp_channels/cell_masks`。

### 2.2 讀數（每張圖塊算，再對圖塊 bootstrap）

**D1　細胞集合是否一致（不看形狀）**
- Lin 中心點和真實 H&E 的偵測中心點一對一配對，規則和配對半徑 r 照 `robust.py`。
- 報告：配上的、沒配上的 Lin、沒配上的偵測，分 CellViT 和 StarDist。

**D2　沒配上的 Lin 核在 H&E 上看不看得到**
- 每顆 Lin 核（loose 侵蝕，真實形狀）在 H&E 的 hematoxylin OD（`compare.hematoxylin`）上取平均值，當作分數。
- 正例：兩種分割器都配上的 Lin 核。
- 負例：把同一批正例的形狀，搬到同一張圖塊裡不碰任何 Lin 細胞、也不碰任何偵測的位置（每張圖塊一個固定 seed）。
- 門檻 t：正例對負例的 Youden 點（使 TPR − FPR 最大）。
- 報告：
  - 沒配上的 Lin 核裡，分數 ≥ t 的比例 f；
  - 用 Rogan–Gladen 修正後的比例 (f − FPR)/(TPR − FPR)，截到 [0, 1]。
- 解讀：
  - 修正後接近 1：H&E 上看得到核，是分割器漏掉；
  - 接近 0：DAPI 有核，但 H&E 上看不到。

**D3　沒配上的偵測落在哪裡**
- 中心落在某個 Lin 細胞（整個細胞遮罩）裡的比例：大多是合併或位移，不是另一顆核。
- 落在所有 Lin 細胞外的比例：H&E 上有核、DAPI 沒有分割出來。

**D4　遮罩畫法的代價分解**
- 都用 PixCell 的 `_compute_pq`，參考分別是真實 H&E 的 CellViT 和 StarDist。
- 每個遮罩先二值化、再取連通區塊，和 PixCell 讀 `cell_masks` 的方式相同，也就是模型看到的樣子。

| 代號 | 遮罩 |
|---|---|
| R0 | 現在的訓練遮罩（外形庫輪廓，相碰的核連成一塊） |
| L1 | Lin loose 核的真實形狀，全部細胞，相碰處留 1 px 背景線 |
| L2 | L1 只留 D2 判定看得到的核（分數 ≥ t） |
| H1 | 真實 H&E 的 StarDist 核，相碰處留 1 px |
| H2 | 真實 H&E 的 CellViT 核（論文時期的物件），相碰處留 1 px |

- 另外報告：
  - 每個 Lin 細胞對應幾個連通區塊（R0 的合併程度）；
  - 兩種 H&E 分割器彼此的 PQ（H&E 對 H&E 的上限）。

### 2.3 選遮罩來源的規則（數字出來前固定）

**交叉 PQ：** 一律對「沒有用來做這個遮罩」的分割器算。
- L1、L2：對 CellViT 和對 StarDist 的平均；
- H1：對 CellViT；
- H2：對 StarDist。

兩組（crc33sec、crc36ho）合併後對圖塊 bootstrap（10,000 次，seed 42），取 95 % 區間。

**H 系列的類型轉貼規則：**
- H&E 核的中心落在哪個 Lin 細胞（整個細胞遮罩）裡，就用那個細胞的類型和狀態。
- 落在 Lin 細胞外：用 r 內最近的 Lin 中心點。
- 都沒有：只畫進 `cell_masks`，不畫類型和狀態。
- 報告：有類型的 H&E 核比例，以及每個 Lin 類型有多少比例的細胞被至少一顆 H&E 核帶到。

**規則：**
1. 如果 L1 或 L2 的交叉 PQ 區間，和 H1、H2 中最好的那個重疊 → 選 L 系列。理由：
   - 一個細胞對一顆核，類型不必轉貼；
   - 模擬端也是一個細胞畫一顆核。
   - 在 L 系列裡，L2 的區間整個高於 L1 才選 L2，否則選 L1（保留所有細胞，模擬端不必丟細胞）。
2. 否則選 H1、H2 中交叉 PQ 較高的那個（區間重疊時選 H1，因為 StarDist 不像 CellViT 會合併相鄰核，見記憶 `cellvit-undersegmentation`）。
3. 不論選哪個，**畫法一律是「真實形狀＋相碰處 1 px 背景線」**，讓模型看到的二值遮罩保留每一顆核。

### 2.4 看圖用的接觸表

6 張圖塊（每組 3 張；用固定 seed 從細胞數高於該組中位數的圖塊抽），每張並排四欄：
1. H&E；
2. R0；
3. L1 疊在 H&E 上；
4. 選中的遮罩疊在 H&E 上。

存成 `figures/contact_sheet.png`，文字用英文。

## 3. 模擬端（只寫計畫，第 1 階段不改程式）

- sim2real renderer（`morphology_renderer.render_crop`）目前畫外形庫輪廓，相碰時不留縫。
- 新遮罩上線後：
  - 外形庫改由選中的遮罩物件建立（照 `build_nucleus_shape_library.py` 的 48 px 規則）；
  - 加一個預設關閉的選項，在相碰的核之間留 1 px 背景線。
- 預設不變，舊結果可以重現。

## 4. 第 2 階段：環境通道（方法 C）

- 照 `scripts/analysis/heldout/real_env_c.py`，分 tissue → coarse → contract → solve 四步，在 Hyak cheme CPU 上跑。
- AF1 擬合：兩張切片都用十月凍結的常數（`contract_vessels.FIT_VALUES`，在 CRC33 上擬合）。
  - **修訂（2026-11-21，在任何通道數字出現之前）**：原本寫「CRC36 自己擬合」。查證後發現 sim2real 的方法 C 在 CRC16、CRC34、CRC10 上都直接沿用 CRC33 的凍結常數（`real_env_c.py` → `contract_vessels.contract`），沒有逐張重擬。
  - 要「和 sim2real 相同」，CRC36 也用凍結常數。
  - 另外在 CRC36 上照十月的規則重擬一次，只做描述：報告重擬常數，以及兩版 contract 線的 Dice。
- 之後在每張訓練圖塊上取出：
  - ch7 = level 0 的 8 µm contract 線；
  - ch8–9 = 氧、葡萄糖。
- 閘門：
  - CRC33_01 的 contract 線要等於十月的那一版（重疊圖塊逐像素相同）；
  - 每個訓練圖塊都要有三個通道。

## 5. 第 3 階段：重訓與比較（使用者已核准重訓）

**訓練：**
- PixCell cb62fa7，Concat 設定不變；有效批次 80、20 epoch、seed 42。
- 資料：CRC33_01 9,418 張 + CRC36 16,651 張，只換 7 個細胞通道（新遮罩）和 ch7–9（方法 C）。
- Tillicum 一張 H200，時間上限 4 小時（預算上限 $40，剩約 $22）。

**比較的兩個模型：** N（新）對 K（`tr_concat/step_0006520`）。

**讀數：**
1. **遮罩忠實度：**
   - 次要 PQ 和對 Lin 中心點的 F1，參考是各自的訓練樣式遮罩（N 用新遮罩，K 用 R0）；
   - 都附上真實 H&E 對同一遮罩的參考線，報告「模型減參考線」；
   - 用 crc36ho 和 crc33sec。
2. **外觀：** FUD、style_hed、LPIPS（hfv Fig 2 的讀數），描述性。
3. **sim2real：**
   - 應用 8 的讀數（Δ、canc_neo），CRC16 的 12 個視窗；
   - 輸入用各自模型的訓練樣式（N 用新畫法＋方法 C，K 照應用 8）；
   - 風格 own／other；
   - 配對差 N − K，12 視窗 bootstrap。

**判讀表**（區間 = 95 % bootstrap）：

| 列 | 條件 | 決定 |
|---|---|---|
| 1 | 忠實度（模型減參考線）N 高於 K，而且 Δ 的 N − K 區間不在 0 以下 | sim2real 改用 N |
| 2 | Δ 的 N − K 區間整個在 0 以下 | 留 K，回報原因 |
| 3 | 忠實度沒有提高（區間含 0 或更低），Δ 也不在 0 以下 | 回報；遮罩選擇交給使用者 |

## 6. 不做的事

- 不改 PixCell、share-space-paper。
- 不動正在跑的 hfv 全部列探針，也不動應用 9。
- 第 1 階段不送 GPU 作業。

## 7. 修訂：遮罩改用 L1（使用者決定，2026-11-21）

**時間點：** 第 1 階段的診斷結果出來之後、任何訓練開始之前。依 §2.3 的規則選出的是 H1（`RESULTS_stage1.md` §2），使用者改選 **L1**。這是**偏離 §2.3 規則、由使用者做的決定**。

**使用者的理由（原話）：**

> 「stardist不夠準…會有一堆undersegmentation (False negative)…那感覺是segmentation model的問題？為什麼反而是我們的模擬要去配合？」

**對應的診斷數字（D2，`RESULTS_stage1.md`）：**
- 兩種分割器都沒配上的 Lin 核，校正後有 70 %（crc33sec）／84 %（crc36ho）在 H&E 上有蘇木素（hematoxylin）訊號。
- 也就是說，H1 和 L1 之間交叉 PQ 的差距，很大一部分來自分割器漏抓（偽陰性），不是 Lin 的核在 H&E 上不存在。交叉 PQ 是拿分割器當參考，分割器漏抓時，畫出所有細胞的 L1 會被扣分。
- 限制（照實寫）：在密集區，Lin loose 核大約和整個細胞一樣大，所以 D2 的「看得到」會偏高一些。

**L1 的定義：**
- 每個 Lin 細胞都畫：用它自己的 loose 核形狀（`stages/lin_typing/nuclei.erode_labels` 的 loose）。
- 相碰處留 1 px 背景線（`diag.separate` 的規則：一個像素若八鄰域碰到編號較小的非零標記，就清掉）。
- 類型、狀態照現在的 CELESTA + Lin Fig. 3d 標記。

**隨之取消的東西：**
- 模擬端不丟細胞（不做 H1 需要的 thinning）。`RESULTS_stage1.md` §3 的選項 (a)(b) 都不做。
- 不用 StarDist 外形庫，也不用對訓練圖塊跑 StarDist。

**模擬端（§3 的具體化）：**
- 外形庫不換：ARCADE 用的 `nucleus_shapes_lin_*.npz` 本來就是 Lin loose 核。
- ARCADE 模擬本來就八鄰域分開（`paired_positions`），不用改。
- 「真實細胞經 renderer」的路徑（應用 8 的 CRC16 輸入）在 `render_crop` 加一個預設關閉的選項 `separate_touching`，打開時套同一條留線規則。預設下輸出逐位元組不變。

**§5 判讀表不變。** 模型 N 的訓練樣式遮罩改為 L1。

## 8. 加一項便宜的 CPU 檢查：L1b（2026-11-21，在 L1b 任何數字出來之前寫下；協調者指示）

**L1b 是什麼：** 真實訓練圖塊也用「模擬端的畫法」畫。
- 和 R0 完全相同：Lin 中心點、外形庫輪廓 `NucleusShapes.pick(type, area, CellID)`、同一個外形庫、同一張細胞表（`cells_nuclei.npz` 的 `target_px`）。
- 唯一不同：相碰的核用 `separate_instances` 分開（`render_crop(..., allow_contact=True, separate_touching=True)`，sim-exp-validation bc14b086），不再讓相碰的核連成一塊。

**閘門（先過才算 L1b）：** 同一程式關掉 `separate_touching` 重畫，要和評估包裡的 R0 `cell_masks` 逐位元組相同；不同就停下回報，不算 L1b。

**讀數：** 交叉 PQ，算法和評估圖塊與 §2.3 的 L1 完全相同（對 CellViT 和對 StarDist 的平均；crc33sec + crc36ho 合併，圖塊 bootstrap 10,000 次，seed 42）。另外報：
- R0 → L1b：只靠「分開相碰的核」得到的差距；
- L1b → L1：再加上「每個細胞用自己的形狀」得到的差距。
兩段差距都用同一組 bootstrap 抽樣算配對差的 95 % 區間。

**規則：**
- L1b 的 95 % 區間和 L1 的重疊 → **建 L1b**。理由：真實訓練圖塊和模擬用同一條畫法，訓練輸入和 sim2real 輸入來自同一套規則。
- 不重疊 → **建 L1**（自己的形狀），並註明模擬端用的是同一族群的外形庫輪廓。

## 9. 修訂：第 3 階段的執行細節與判讀表的操作定義（2026-11-21 23:40，訓練中、任何 N 的影像或數字出現之前；協調者交辦的 agent 寫入）

這一節只把 §5 寫得不夠細的地方定下來，不改 §5 的讀數和判讀表。

### 9.1 讀數 1、2 的生成（Fig 2 設定）

- N：PixCell cb62fa7 `tools/ablation_a1_a2/build_cache.py --variants a1_concat`（hfv Fig 2 的 Concat 路徑；30 步、導引 1.5、種子 42、該圖塊自己的 UNI），透過 `hfv_retrain_20261118/eval/evalgen.py` 的影子目錄執行，只產生 Concat。
  - 圖塊：crc36ho 的 300 張（`crc36ho_p300_ids.txt`）、crc33sec 的 Fig 2 100 張（`crc33sec_fig2_ids.txt`）。
  - 輸入：評估包 `evalpack_<set>.tar` 只換三樣——7 個細胞通道（`cells2/eval_<set>_L1.tar`）、ch7–9（`envc/<slide>/envc_<slide>.tar` 的方法 C）、遮罩 latent（`masklat/masklat_eval_<set>_L1.tar`）。UNI、H&E、H&E latent 不變。
- K：沿用 2026-11-19 已生成的 Fig 2 Concat 影像（S3 `hfv_retrain_20261118/outputs/score_fig2_<set>.tar` 的 `tiles/a1_concat/`），不重生成。同一支程式、同一組設定、同一種 GPU（Colab L4）、同一套環境（torch 2.10.0+cu128）。
- 硬體：N 也在 Colab L4 生成（理由見 9.4）。

### 9.2 讀數 1、2 的打分（同一支程式、同一個裝置打兩個模型）

- 新程式 `scripts/r2score.py`，在 Hyak cheme CPU 上對 N、K 和真實 H&E 用**同一個**設定重跑 CellViT（未修改的 hfv `run_cellvit_local.py`、CellViT-256、batch 32、原始解析度，與 sim2real 應用 8 相同）。K 在 11-19 的 L4 CellViT 結果不拿來配對，只拿來比對 K 的次要 PQ 平均（描述）。
- **次要 PQ**：PixCell `_compute_pq`；參考 = 模型自己的訓練樣式遮罩（N：L1；K：R0，即評估包的 `cell_masks`），二值化後取連通區塊（PixCell 讀 `cell_masks` 的方式）；預測 = 生成圖 CellViT 的實例（`_cellvit_json_to_instance_mask`）。
  - **參考線** = 同一張圖塊的真實 H&E CellViT 對同一個遮罩的 PQ。
- **對 Lin 中心點的 F1**：`hfv_retrain_20261118/eval/robust.py` 讀數 2 的定義（Hungarian 一對一配對、配對半徑 r 沿用 `info_<set>.json`：crc36ho 9.06 px、crc33sec 8.83 px、邊緣規則相同）。參考線 = 真實 H&E 的同一個 F1（兩個模型共用）。
- **忠實度**（每張圖塊）= 模型的值 − 參考線。比較 = N − K 的配對差，對圖塊 bootstrap 10,000 次、seed 42（`evalstats.py` 的慣例），95 % 百分位區間。
- **外觀（讀數 2，描述）**：LPIPS（alex）、style_hed（`fig2tiles.py` 的寫法）逐張算，N − K 給同樣的配對區間；FUD 每組一個數（PixCell `compute_fid_from_stats`；真實側用評估包存的 UNI，生成側用 `pxscore.py` 的 UNI-2h 轉換在 CPU 上抽），沒有區間。
- 主要 PQ（參考 = 真實 H&E 的 CellViT）也一併列出，描述。

### 9.3 讀數 3 的臂與配對

| 代號 | 模型 | ch0–6 | ch7–9 | 風格 | 來源 |
|---|---|---|---|---|---|
| NP | N | CRC16 L1（`eval/crc16_L1`） | 方法 C（`orion-crc16e`，同應用 8 KE） | own（`P_<tile>`） | **= 應用 10 的 KP 臂，同一次生成** |
| ND | N | 同上 | 同上 | other（`D_<tile>`，應用 7 的排列） | 本實驗新生成 |
| KP、KD | K | s3b（R0 畫法） | hfv mx | own／other | 應用 8，沿用 |

- 生成設定：種子 42、43；20 步；導引 2.5；conditioning scale 1.0；`app8_gen_k.py` 的生成路徑；Colab L4。應用 10 預登記 §3 的設定與應用 8 §4 完全相同，所以 NP 與應用 10 共用，N − K 的配對成立。ND 用 `app8_gen_k.py` 的複本，只把「1,152 列」改成讀設計檔的列數、參考張量的鍵改成設計檔指定。
- **真值單位**：每個模型用自己看到的那份遮罩。N：CRC16 L1 實例圖（`eval/crc16_L1/labels/<tile>.npy`，類型取自同一份 L1 通道）；K：應用 8 的 s3b 單位（`app8_stats.py` 不改）。真實 H&E（應用 2 `cv/RAW`）在兩種單位上各配對一次，兩個值都報告。
- 配對規則、Δ、canc_neo、核面積：`app8_stats.py`。區間：同一個 (2,000 × 12) 視窗索引矩陣（`default_rng(20261128)`），所以 N − K 是配對差。
- **描述讀數**（應用 9 結果 §4 預告）：圖塊內 Δ（應用 10 預登記 §6.2 的 Mantel–Haenszel 風險差）、核面積中位數，四個臂都算；以及 N − K 的配對差。

### 9.4 放置（依實測）

- Colab L4：應用 9 實測 1,152 張生成 2,739 s（每張 2.38 s），全程 2,998 s；hfv Fig 2 30 步 25.16 張／分。
- Hyak GPU（2026-11-21 23:10 查）：ckpt 分區閒置的 GPU 只有 rtx6k／2080ti，A100、L40S、A40、L40 全部滿載；rtx6k 上沒有任何生成的實測速度。
- K 的 Fig 2 與應用 8 的 KP／KD 都是 L4 生成；應用 10 預登記 §8 也指定 L4。N 用同一種 GPU，N − K 才不混入 GPU 型號的差別。
- 所以：兩個 L4 session（`hfv_eval_L4_a`／`_b`），應用 10 的 4,608 張 + ND 384 張 + Fig 2 400 張（30 步）。

### 9.5 判讀表的操作定義（§5 的表不變）

- 「忠實度 N 高於 K」= 四個 N − K 區間（次要 PQ、F1 × crc36ho、crc33sec）**全部**整個 > 0。
- 「忠實度沒有提高」= 不滿足上一條（任一個區間含 0 或 < 0）。
- 「Δ 的 N − K 區間整個在 0 以下」= own 或 other 任一種風格的區間上界 < 0。
- 套用順序：先看第 2 列（任一風格 Δ 區間 < 0 → 留 K）；再看第 1 列（忠實度四個區間都 > 0，且兩種風格的 Δ 區間都不在 0 以下 → 改用 N）；其餘 → 第 3 列（回報，遮罩選擇交給使用者）。
- canc_neo、外觀、圖塊內 Δ、核面積不參與判讀，照實報告。
