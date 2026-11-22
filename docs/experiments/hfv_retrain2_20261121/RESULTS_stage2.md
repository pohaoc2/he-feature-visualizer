# 重訓 2 第 2 階段：建置、打包、送出訓練（2026-11-21）

依 PREREG §7（L1，使用者決定）與 §8（L1b 檢查 → 建 L1）。所有 Hyak 路徑都在 `/gscratch/cheme/chiu/hfv_retrain2_20261121/` 之下。

## 1. 閘門結果

| 步驟 | 輸出 | 閘門 |
|---|---|---|
| L1 細胞通道（`cells.py`，作業 43457193／43457194） | `cells2/cells_crc{33,36}_L1.tar` | 標記全部是 CellID；類型／狀態互斥且合起來等於遮罩；評估圖塊連通區塊數 = diag n_L1（824 + 300 張，0 張不同） |
| 方法 C ch7–9（`envc.py`） | `envc/crc{33,36}/envc_*.tar` | G1 solver == 舊值；G2 舊 patch 逐位元組相同（CRC33 20,758、CRC36 35,744，0 不同）；G3、G4 通過 |
| 遮罩 latent（`masklat.slurm`，作業 43457537） | `masklat/masklat_*_L1.tar` | 裝置比對 64 張：和舊 latent 最大差 0.00195（fp16 捨入範圍）；四組 latent 都齊（9,418／16,651／824／300） |
| 打包＋上傳（`pack.py`、`pack.slurm`，作業 43457604） | `pack/*.tar` → `s3://bagherilab-working/pohao/hfv_retrain2_20261121/inputs/` | 檔數 94,181 = 10 × 9,418 + 1；9,418；216,464 = 13 × 16,651 + 1；每個檔寫入後重讀，sha256 相同；S3 大小 = 本地大小（共 18.9 GB） |
| Tillicum 準備（`setup2.sh`） | `/gpfs/scrubbed/pohaoc2/hfv_retrain2_20261121/data` | 三個 tar 的 md5 = `pack.json`；10 個通道各 9,418 張；索引 26,069 筆；遮罩 latent 26,069 個；舊遮罩 latent 0 個；資料夾結構同重訓 1 |
| 訓練 | Tillicum 作業 **509534**（H200，≤ 4 h，≤ $3.60） | resume=None；total_steps 6,520（同 K） |

md5：newch33_v2 `2f4ecd0c…a790`、masklat33_v2 `68725e4e…e3b3`、c36train_v2 `6a0f9478…b967`（`data/pack.json`）。

## 2. 訓練組成和 R0 幾乎相同（`data/coverage_L1_vs_R0.json`）

- 細胞核遮罩覆蓋率：CRC33 0.172（R0 0.168）、CRC36 0.321（R0 0.313）。
- 每個通道的逐張覆蓋率，L1 與 R0 的相關都在 0.996–0.999。
- 換的只是核的形狀和相碰處，細胞組成沒變。

## 3. CRC16 L1 評估輸入（應用 8 的 192 張；重訓 2 評估與應用 10 共用）

- 路徑：`eval/crc16_L1/`
  - `exp_channels/<10 ch>/<tile>.(png|npy)`；
  - `labels/<tile>.npy`：L1 實例圖，int64，每像素一個 CellID，作為真值單位；
  - `render/<w>.npz`：12 個 6 × 6 視窗；
  - `tiles.txt`、`manifest.json`。
- content_sha256（所有檔 sha256 清單的 sha256）：`eae8e8821da2a59406da5efe6150d65b7f22fb62492b78aab2d06f44ceba3895`
- `manifest.json` 的 sha256：`a219333f3ad9a88a0f93a80a640a0c307550d958e7a68194088f9711d30c4a84`
- 程式：
  - sim-exp-validation 90622bc7：`msi_s3b_render.py` 加 `render_tile(true_shapes=False)` 與 `render_l1` 指令；
  - 執行用的副本是 `scripts/msi_s3b_render.py`，作業 43457634（`crc16l1.slurm`）。
- 輸入來源：
  - 類型取自 `msi_perturb_20261119/s3b_crc16b/cells_labels.npz`；
  - 遮罩是 Lin CRC16 mask；
  - ch7–9 從 `sim2real_apps_20261120/app8/kin/roots/orion-crc16e` 逐位元組複製。
- 閘門：
  - `render_tile` 預設路徑：12 個視窗共 432 張，和 `s3b_crc16b/render` 逐位元組相同；
  - 訓練用的建置程式 `cells.py` L1 在這 192 張上畫出的 1,344 個 PNG，和本輸出逐位元組相同；
  - 每顆核的類型／狀態等於 `cells_labels`。
- 描述：
  - 遮罩覆蓋率 0.433（腫瘤 0.314、健康 0.055、免疫 0.064）；
  - 每張連通區塊中位數 92；每個 Lin 細胞 1.01 個連通區塊。
- `cells.py` 的最小改動：缺 `cells_nuclei.npz` 時，L1 模式直接用遮罩畫，`target_px` 只有 L1b 會用到；L1b 模式仍然會 STOP。

## 4. 偏離與更正

1. **遮罩 latent 改用 GPU（Hyak ckpt）。** 指示原本是「第 2 階段用 CPU」。
   - CPU 試跑時，64 張的裝置比對 13 分鐘還沒跑完第一個 batch，照這速度全部要好幾天。
   - VAE 抽取的前例（gpu34.slurm）本來就是 GPU 步驟，所以改用 GPU，指令與環境不變。
   - 用的是 Hyak 免費的 ckpt GPU，不是 Tillicum，也不是 Colab。
2. **`separate_instances` 的邊緣 bug。**
   - 填充值用 int64 最大值，scipy 轉成 double 時會變成 2^63，在 x86 上溢位成負數，切塊邊緣的像素因此被誤清。
   - 是建置閘門抓到的：評估圖塊的連通區塊數對不上 diag 的 n_L1（CRC33 有 173 張、CRC36 有 202 張不同）。
   - 修正在 sim-exp-validation 517462bd，並加了邊緣測試 32494f0e。
   - L1b 已重跑，結論不變（RESULTS_stage1 §4）；修正前建的 `cells/` 不使用。
3. **`masklat.slurm` 的相對路徑問題。**
   - 第一次的 A100 作業（43457221）排隊沒有開始就取消了。
   - 改用 CPU 版（`masklat_cpu.slurm`，作業 43457378）跑裝置比對時，發現比對用了相對路徑，找不到圖。
   - 已改成 `$G/...` 的絕對路徑，加上「失敗就 STOP」的檢查。
   - CPU 版重送（43457415）後太慢而取消；最後改成任意型號的 ckpt GPU（43457537）。
