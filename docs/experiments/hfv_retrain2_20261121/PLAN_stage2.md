# hfv_retrain2：模擬端畫法與方法 C 通道建置計畫（第 1 階段只寫計畫）

## 1. 模擬端：選中的遮罩，renderer 畫不畫得出來

**現況（查程式得知）：**

| 路徑 | 位置 | 外形 | 相碰 |
|---|---|---|---|
| 真實訓練遮罩（`stages/lin_typing/render.py`） | Lin 中心點 | 外形庫 `nucleus_shapes_lin_<slide>.npz`（Lin loose 核，48 px） | `allow_contact=True`：相碰的核二值化後連成一塊 |
| ARCADE 模擬（`morphology_renderer.agent_stamps` → `paired_positions`） | 六角格內抽位置 | 同一種外形庫 | 八鄰域分開：**本來就不相碰** |
| 應用 8 的真實 CRC16 輸入（`msi_s3b_render.py`） | 同真實訓練遮罩 | 同上 | `allow_contact=True` |

**各候選需要的改動：**

- **L1（Lin 真實形狀，相碰處留線）**
  - 外形庫不用換：`nucleus_shapes_lin_*.npz` 本來就是 Lin loose 核。
  - 模擬端不用改：ARCADE 本來就畫分開的核。
  - 真實端換畫法：直接畫每顆 Lin loose 核的真實形狀，再套 `diag.separate`。這是 hfv 端的新函式，不動 sim-exp-validation。
  - 應用 8 那種「真實 CRC16 細胞經 renderer」的輸入要改成同樣的畫法：在 `msi_s3b_render.render_tile` 加一個預設關閉的參數 `true_shapes=False`（stub 見 §1.1）。
- **L2**：同 L1，另外丟掉 H&E 上看不到的核。
  - 模擬端沒有「看不到」的概念，只能照 L2 在各類型丟掉的比例隨機丟細胞。這是新的畫法假設，需要另外驗證。
- **H1／H2（H&E 核＋轉貼類型）**
  - 外形庫要從 StarDist／CellViT 物件重建（照 `build_nucleus_shape_library.py` 的 48 px 規則）。
  - 一個 H&E 核可能對到 0 或 2 個 Lin 細胞。模擬端一個細胞畫一顆核，密度會和訓練資料不同，要另加規則。
  - 沒有類型的 H&E 核，模擬端畫不出來。

### 1.1 stub（預設關閉，第 1 階段不接上任何流程）

```python
# stages/lin_typing/render.py（hfv，新函式；現有函式不改）
def render_tile_true_shapes(loose_labels_tile, table, x0, y0):
    """L1: paint every Lin loose nucleus with its own shape, a separation line where nuclei touch
    (diag.separate), type / state channels from the Lin labels. Not wired into any pipeline yet."""
    raise NotImplementedError("hfv_retrain2 stage 2")
```

sim-exp-validation 的 `msi_s3b_render.render_tile` 加 `true_shapes=False` 參數，預設路徑的位元組不變（閘門：預設參數下重畫應用 8 的 192 張，逐位元組相同）。

## 2. 方法 C 通道建置（CRC33_01 9,418 張、CRC36 16,651 張訓練圖塊）

**規則：**
- 和 sim2real 相同：`scripts/analysis/heldout/real_env_c.py` 的 tissue → coarse → contract → solve。
- AF1 擬合用凍結常數 `contract_vessels.FIT_VALUES`，兩張切片都一樣（PREREG §4 修訂）。

**要加的程式（新檔，sim-exp-validation `scripts/analysis/hfv_retrain2/`）：**
1. `train_tiles_c.py contract-l0`：每張訓練圖塊在 level 0 算 `contract(CD31, AF1)`，輸出 8 µm 線（bool）。
   - 現在的 contract 步驟只存 ds-4 的區塊比例，沒存 level 0 的線。
   - 規則和 `contract_vessels.contract` 相同。
2. `train_tiles_c.py solve`：沿用 `cmd_solve` 的場，取出訓練圖塊的氧、葡萄糖（float32）。
3. 寫成 PixCell 讀得到的格式，和應用 8 KE 相同：
   - `vasculature/<t>.npy` bool；
   - `oxygen`、`glucose` `<t>.npy` float32，值在 [0, 1]。

**閘門：**
- CRC33_01 重疊圖塊的 contract 線，要等於十月 `sev-vessel-contract-20261006` 的那一版（逐像素）。
- 26,069 張圖塊都有三個通道。
- 氧、葡萄糖都在 [0, 1]。
- CRC36 另外重擬 AF 常數做描述：報告常數，以及兩版 contract 線的 Dice。

**多工影像來源：**
- CRC36：`hfv_crc36_20261118/lin/data__CRC36__…-zlib.ome.tiff`，已在 Hyak。
- CRC33_01：Lin 公開 bucket。讀取不需要詢問，只回報下載量（記憶 `data-reads-no-ask`）。用完刪除。

**實測時間**（CRC16 方法 C，Hyak cheme，8 CPU、96 GB）：

| 作業 | 內容 | 時間 |
|---|---|---|
| 42320045 | tissue + coarse | 3 分 08 秒 |
| 42320570 | contract + solve，3,615 個 1024 px 區塊 | 17 分 37 秒 |

- CRC16 的 ds-4 大小是 16,764 × 21,406。CRC36（level 0 57,967 × 72,345）和 CRC33_01（52,740 × 36,354）都比它小，所以每張切片預計 ≤ 21 分鐘。
- 訓練圖塊的 level 0 線要再讀一次 CD31／AF1，按 contract 步驟的比例估，每張切片 ≤ 15 分鐘。
- 合計 2 張 × ≤ 36 分鐘，8 CPU，在 Hyak condo 上不另外計費。

## 3. 開始訓練前要備齊的東西

1. 第 1 階段的診斷選出遮罩（`data/diag_stats.json` 的 `choice`）。
2. 新的 7 個細胞通道：
   - CRC33_01 9,418 張 + CRC36 16,651 張，用選中的畫法；
   - 閘門：每張通道的類型、狀態和現有通道分配到相同的細胞（同一組 Lin 標籤），只有形狀和相碰處不同。
3. 方法 C 的 ch7–9（§2）。
4. 打包成和 `c36train.tar`、`new_exp_channels.tar` 相同的結構；index、H&E VAE latent、UNI 不變。
   - Concat 的 TME 模組不讀 mask latent（`RawConditioningPassthrough`），所以不必重算。
5. Tillicum：
   - 重用 `tillicum/tr_variant.slurm`，新 run name `tr_concat2`，S3 新位置；
   - 時間上限 4 小時，H200 每小時 $0.90，最多 $3.60；
   - 需要使用者先登入 Tillicum（目前 ssh 被拒，要 `! hpc-connect tillicum` 或對應的登入）。
6. 評估：沿用 `hfv_retrain_20261118/eval/`（evalgen／evalscore／robust），參考遮罩換成新畫法；應用 8 的讀數要另寫一個 app 預登記。
