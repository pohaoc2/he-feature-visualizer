# CRC36 multiplex 衍生條件通道（vasculature / oxygen / glucose）

狀態（2026-11-18 21:10）：步驟 1a、1b、2 都完成。CRC33_01 重現成功（見下表）。協調者 2026-11-18 決定接受 vasculature 有 6 張 tile 因 Otsu 門檻差 1 個灰階而不同（IoU 0.99998）。

## 步驟 2：CRC36（job 43369533，COMPLETED，16 核心，耗時 4 分 47 秒，`results/mx_channels_crc36.json`）

- 17,872 張 tile：18,103 張扣掉 231 張墨水 tile；沒有任何 tile 超出影像範圍。
- 參數：ds 4、O2 L 200 µm、glucose L 120 µm、不做 SMA。
- 每張 tile 的平均值：

  | 通道 | 平均 | p5 | 中位數 | p95 |
  |---|---|---|---|---|
  | vasculature 面積比例 | 0.186 | 0.022 | 0.094 | 0.450 |
  | oxygen | 0.923 | 0.811 | 0.936 | 0.987 |
  | glucose | 0.878 | 0.709 | 0.898 | 0.979 |

- 輸出：`/gscratch/cheme/chiu/hfv_crc36_20261118/data/crc36_mx_channels.tar`（10,596,239,360 bytes），只留在 Hyak。md5 記在 c36train 工作的 log。

## 步驟 1b：CRC33_01 全部 10,379 張 tile 的重現結果（`results/compare_crc33.json`）

設定：`--wsi-pde-ds 4`、O2 L = 200 µm、glucose L = 120 µm、不做 SMA refine（全部是 hfv 預設值），來源是 Lin 的 `CRC33_01/P37_S76_01_A24_C59qX_E15_20220302_235136_561323-zlib.ome.tiff`。

| 通道 | npy bytes 完全相等的 tile 比例 | max abs diff | pooled Pearson r | 其他 |
|---|---|---|---|---|
| oxygen | 10,379 / 10,379 = 1.0 | 0 | 1.0 | — |
| glucose | 10,379 / 10,379 = 1.0 | 0 | 1.0 | — |
| vasculature | 10,373 / 10,379 = 0.99942 | 1（bool） | 0.99999 | 像素一致率 0.999996，IoU 0.99998 |

vasculature 有 6 張 tile 不同（`results/diag_vasc_crc33.json`）：
- 不同的像素數在 64–1,502 之間（佔該 tile 0.1–2.3%）。
- 每一張的差異都只落在 Otsu 門檻上的那一個 uint8 灰階（其中兩張是門檻旁相鄰的兩個灰階），也就是門檻差了 1 個灰階。
- 推測原因：浮點運算或函式庫版本的差異（numpy 的 percentile、`*255` 後轉 uint8 的捨入、OpenCV 的 Otsu）。原始執行時的套件版本沒有留下紀錄，所以無法確認。這不是方法上的差異。
- 本次使用的版本：numpy 2.0.2、opencv 4.13.0、scipy 1.17.1、tifffile 2026.4.11、zarr 3.1.5。

步驟 1a 的 sweep（`results/sweep_crc33.json`，500 張 tile）：
- 只有 ds = 4、不做 SMA、O2 L = 200、glucose L = 120 這組是 500/500 完全相等。
- 文件上記載的 160/450 µm：max abs diff 分別為 0.08 / 0.11。
- ds = 8：max abs diff 0.27–0.64，Pearson r 約 0.93。

CRC33 的耗時（16 核心）：WSI 欄位 98 s、讀 level-0 CD31 36 s、切 tile 25 s，總共 221 s。

## 方法（hfv Stage 4，`stages/multiplex_layers.py`，hfv 程式碼原封不動 import，commit d55ee2b）

- vasculature：每個 256 px tile 取 level-0 CD31 crop（等於 `stages/patchify.py` 寫進 `multiplex/<tile>.npy` 的內容；H&E 與 multiplex 都是 0.325 µm/px，所以 crop 就是 `[y0:y0+256, x0:x0+256]`），然後 `percentile_norm`（1–99 百分位）→ `binarize_otsu` → `cleanup_vasculature_mask`（預設 0, 0, 0，不做任何處理）→ `apply_vessel_mask_quality_fallback`（0.98）。
- oxygen / glucose：`read_wsi_channel_stack`（CD31、Ki67、CD68，下採樣倍數 `ds`）→ `solve_wsi_pde_map`（WKB 近似：`u = exp(-d/L(x))`，其中 `L(x) = L·sqrt(k0/k(x))`、`k = 0.1 + 0.3·Ki67 + 0.1·CD68`）→ 每個 tile 用 `extract_patch_from_coarse` 切出來（bilinear 放大回 256 px）。
- 訓練資料的格式（看 `orion-crc33.zip` 的 central directory 得知）：**是 `.npy`，不是 PNG**。
  - `exp_channels/vasculature/<x>_<y>.npy`：bool (256, 256)，等於 hfv 的 `vasculature_mask/`。
  - `exp_channels/oxygen|glucose/<x>_<y>.npy`：float32 (256, 256)，值域 [0, 1]，等於 hfv 的 `oxygen_npy/`、`glucose_npy/`。
  - `processed_crc33.zip` 裡另有 `vasculature/ oxygen/ glucose/` 的 PNG，那些是 RGBA 上色後的疊圖，不是訓練用的通道。

## 參數：從訓練資料本身推出來（不採用文件上記載的數值）

| 證據 | 結果 |
|---|---|
| 300 個 tile 上 `log(oxygen)/log(glucose)` 的中位數。同一個 tile 的距離 d 與消耗率 k 相同，所以這個比值等於 L_glc/L_O2 | 0.6001（IQR 0.6001–0.6003）。也就是 120/200，正好是 `--oxygen-pde-krogh-um 200`、`--glucose-pde-krogh-um 120` 的預設值；不是 450/160 = 2.81，也不是 100/60 = 1.67 |
| oxygen 一列像素的二階差分轉折點間距 | 每 4 px 一個轉折，對應 coarse grid 64 → 256 的 bilinear 放大，所以 `--wsi-pde-ds 4`（hfv 預設值），不是 8 |
| sweep（500 tile，`default_rng(20261118)`）的 vasculature 比對 | 只用 CD31：500/500 tile 的 npy bytes 完全相等。加上 SMA refine：只有 1.8% 相等，Pearson r 0.84。所以**沒有** `--vasc-sma-refine` |
| sweep 的 ds = 2 列（錯誤候選） | 每個 L 都只有 0.2% 的 tile 相等，max abs diff 0.41–0.58，符合「ds 不對」的預期 |

由此推得、並已在全部 tile 上驗證的指令：

```
python -m stages.multiplex_layers --multiplex-dir <multiplex/> --index <index.json> --metadata-csv <markers> \
  --out <out> --multiplex-tiff <mx tiff> --oxygen-model wsi-pde --glucose-model wsi-pde
# ds 4、O2 L 200 µm、glucose L 120 µm、不做 SMA refine：全部都是 argparse 預設值
```

`PixCell/docs/analysis/manuscript_outline.md` 根據 `handy_cli.txt` 寫的「Krogh 160/450 µm、`--wsi-pde-ds 8`」**與訓練資料不符**。可能的原因：handy_cli 記的是 commit ce3bb3e（2026-03-24，把 wsi-pde 的長度參數拆成 `--*-pde-krogh-um`）之前的指令。share-space-paper `_methods.tex` 第 25 行寫的 L = 160/450 µm 也是同樣的問題，需要使用者決定要不要更正。

## 檔案

- `mx_lib.py`：共用函式（hfv 函式包一層，加上讀 level-0 plane）。
- `sweep_crc33.py`：步驟 1a 的候選比對（vasc ± SMA；ds ∈ {2, 4, 8} × WSI SMA ± × L ∈ {60, 100, 120, 160, 200, 450}）。
- `mx_channels.py`：計算一組 tile 的三個通道，寫成 tar 與 summary JSON。
- `compare_crc33.py`：全部 tile 比對，輸出 byte-equal、array-equal、max/mean abs diff、Pearson r、vasc IoU。
- `repro33.slurm`、`crc36.slurm`：步驟 1b 與步驟 2 的 Slurm 腳本（`DS=4 O2L=200 GLCL=120`）。

## Hyak 環境

- 工作目錄：`/gscratch/cheme/chiu/hfv_mx_20261118`。
- `venv`：zarr 3.1.5，另透過 `.pth` 接上 cellvit env 與 imagecodecs pylib。zarr 3.4.2 跟 tifffile 2026.4.11 不相容（第一次 sweep job 43365399 因此失敗）。

## 偏差

- 輸出格式改成 `.npy`（與訓練資料相同），不是任務描述裡寫的 PNG。
- CRC33_01 的 vasculature 有 6 / 10,379 張 tile 差 1 個 Otsu 灰階（見上）。
- PixCell `manuscript_outline.md` 與論文 `_methods.tex` 寫的參數（Krogh 160/450 µm、ds 8）和訓練資料不符。這裡只記錄，沒有修改那兩份文件。
- `diag_vasc_crc33.py`：列出 vasculature 不相等的 tile，以及它們在 Otsu 門檻附近的灰階分布。
