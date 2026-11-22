# 預先登記：Grouped 類變體是否比 Concat 更照輸入遮罩畫（重訓 1 的 Fig 2 影像，2026-11-22）

這份文件在任何新數字（配對差、hematoxylin 讀數）出現之前 commit 並 push。只讀現有影像和現有 CellViT 結果：**不生成、不訓練、不重跑 CellViT、不改 PixCell、不碰 share-space-paper**。

## 0. 為什麼要做、影響哪個決定

- **決定 D2（sim2real 用哪個生成器）目前是形狀庫遮罩生成器**，也就是重訓 1 的 Concat（`tr_concat/step_0006520`）。
- 上一份實驗（hfv `docs/experiments/cellvit_official_20261122/RESULTS.md` §4.2）用 CellViT 官方後處理重算了重訓 1 Fig 2 的「照遮罩畫的 PQ」（生成圖的 CellViT 核對它自己的輸入遮罩）。各變體的平均值已經公布：
  - CRC36（300 張）：Grouped 0.194、Concat 0.187、Grouped + bypass 0.194；
  - CRC33（100 張）：Grouped 0.283、Concat 0.247、Grouped + bypass 0.280。
- 平均值上 Concat 最低，但**同一張圖塊上的配對差（Concat − Grouped 等）從來沒算過**，所以不知道差別是不是落在抽樣誤差內。
- **這份實驗可能改變的決定：** 如果 Grouped 類變體在兩張切片上都明確比 Concat 更照遮罩畫，sim2real 的生成器選擇（D2 = 形狀庫遮罩生成器 = Concat）就有理由重議。是否真的換生成器、是否重訓，都是**使用者的決定**；這份實驗只回報規則的結果和換的代價。
- 如果結果是「沒有證據」，D2 不需要重議（就這一項而言）。

## 1. 名詞

**代號對照**（2026-11-22 補，push 後、任何數字出現前；只改寫法、不改內容。內文照 sim-exp-validation `docs/TERMINOLOGY.md`「生成器與遮罩」的名稱；代號只留在這張表、檔名和程式）：

| 內文寫法 | 代號／程式欄名 |
|---|---|
| 形狀庫遮罩生成器（重訓 1 Concat，sim2real 目前用的） | K |
| 真實核形 + 模擬血管生成器（重訓 2 Concat） | N |
| CellViT 遮罩生成器（最早的 ControlNet） | C |
| 形狀庫遮罩（Lin 細胞位置 + 形狀庫輪廓） | R0（路徑 `packs/R0/`） |
| Lin 寬鬆核形 | L1 |
| 照遮罩畫的 PQ（參考 = 輸入遮罩） | 舊稱「次要 PQ」；欄 `pq_lin_*`，真實 H&E 參考線 `ref_lin_*` |
| 論文式 PQ（參考 = 真實 H&E 的 CellViT 核） | 舊稱「主要 PQ」；欄 `pq_main_*`、`pq_inst_*`（本實驗不讀） |
| 官方後處理／hfv 舊後處理 | official／runner |
| Vanilla、Grouped、Grouped + bypass、Concat、Per-channel（架構名） | `van`、`grp`、`gbp`、`cat`、`pch` |

**五個生成器變體**（都是 PixCell 的 ControlNet，差別只在「條件影像」怎麼送進 ControlNet；條件影像 = `cell_masks` 細胞核遮罩 + 9 個 TME 通道：細胞類型 3、細胞狀態 3、血管 1、氧和葡萄糖 2）：
- **Vanilla：** 只用細胞核遮罩（VAE 編碼後的 latent）和 UNI 向量，不輸入任何 TME 通道；沒有重訓。
- **Concat：** 10 個通道（遮罩 + 9 個 TME）以原解析度疊在一起，直接送進 ControlNet（PixCell `RawConditioningPassthrough` + `PixCell_ControlNet_XL_2_UNI_Concat`）。**sim2real 目前用的形狀庫遮罩生成器就是這個 Concat**，所以「Concat」和 D2 選中的生成器是同一個模型。
- **Grouped（production）：** TME 通道分成四組（細胞類型、細胞狀態、血管、微環境），每組有自己的小 CNN 編碼器，再用 cross-attention 讀「遮罩 latent」當 query，產生一個修正量 Δ；四組的 Δ 相加後送進 ControlNet。設定 `zero_mask_latent = True`：遮罩 latent 本身**不**直接送進 ControlNet，只透過 cross-attention 的 query 影響輸出。
- **Grouped + bypass：** 和 Grouped 一樣，但 `zero_mask_latent = False`：遮罩 latent 本身也直接加進送給 ControlNet 的條件（「bypass」= 遮罩有一條不經過 TME 編碼器的直通路徑）。
- **Per-channel：** 每個通道一個編碼器、不分組。

**影像與遮罩：**
- **重訓 1 的 Fig 2 影像：** hfv `docs/experiments/hfv_retrain_20261118` 用四個重訓變體和 Vanilla 生成的 Fig 2 圖，CRC36 保留區 300 張（`crc36ho`）、CRC33 次要組 100 張（`crc33sec`），每張圖塊每個變體一張圖。已經打包在 Hyak `/gscratch/cheme/chiu/cellvit_official_20261122/zips/F2_{36,33}_{van,grp,cat,gbp,pch}.zip`（唯讀；sha256 記在上一份實驗的 `data/fig2/zips_manifest.json`）。
- **輸入遮罩：** 這批圖生成時用的是重訓 1 的 `cell_masks`，也就是**形狀庫遮罩**（Lin 細胞位置 + 形狀庫輪廓），**不是** Lin 寬鬆核形（重訓 2 用的遮罩）。檔案：`hfv_retrain2_20261121/stage3/packs/R0/orion-<set>/exp_channels/cell_masks/<tile>.png`。閘門會檢查它和重訓 1 評估包（`hfv_eval_20261118/packs/evalpack_<set>.tar`）裡的同名檔案逐像素相同。所以這裡的結論只適用於「形狀庫遮罩條件下的這四個重訓 1 變體」。
- **真實 H&E：** 同一張圖塊的真實 H&E（`zips/retgt_<set>/`，上一份實驗已驗證和 stage-3 的 R 圖逐像素相同）。

**照遮罩畫的程度（忠實度）的兩類量：**
- **照遮罩畫的 PQ（參考 = 輸入遮罩）：** 生成圖的 CellViT 核和輸入遮罩的核，用 IoU ≥ 0.5 一對一配對；PQ = 配對核的平均 IoU × 偵測 F1（PixCell `_compute_pq`）。1 = 完全一樣。CellViT 有兩種後處理：
  - **官方後處理：** CellViT 原作者的流程（權重設定的正規化、softmax、用 HV 距離圖做分水嶺，把相碰的核切開）。**本實驗規則用這個。**
  - **hfv 舊後處理：** hfv 管線原本的作法（不切開相碰的核）。只描述。
  - 每張圖塊的值已經在上一份實驗的 `data/fig2/tiles_{36,33}.csv`（欄 `pq_lin_official`、`pq_lin_runner`；真實 H&E 對同一遮罩的參考線是 `ref_lin_official`、`ref_lin_runner`）。本實驗不重算 PQ，只算配對差。
  - 「官方／hfv 舊後處理」指的是 **CellViT 的後處理方式**；「照遮罩畫的 PQ／論文式 PQ」指的是**參考是什麼**，兩者是不同的兩件事：照遮罩畫的 PQ 的參考 = 輸入遮罩；**論文式 PQ** 的參考 = 真實 H&E 上 CellViT 找到的核（論文的定義）。
  - **本實驗不計算、也不回報論文式 PQ**（協調者 2026-11-22，在任何數字出現之前決定）。理由：這些生成器收到的是形狀庫遮罩，論文式 PQ 是拿生成圖和「它從來沒被要求去畫的東西」（真實 H&E 上 CellViT 的核）比較，在「是否照輸入遮罩畫」這個問題上沒有解釋。`tiles_*.csv` 裡的 `pq_main_*`、`pq_inst_*` 欄一律不讀。
- **不經過分割器的 hematoxylin 讀數**（定義照 sim-exp-validation `docs/experiments/mask_truenuc_20261122/PREREG.md` §5、§9；程式照 `scripts/truenuc.py` 的 `pix()`）。先把影像做顏色反卷積（Ruifrok & Johnston，`skimage.color.rgb2hed`，hfv `stages/lin_typing/compare.py:hematoxylin`），得到每個像素的 hematoxylin 光學密度（OD，越大 = 越像被蘇木素染成藍紫色的核）。對每張影像、它的輸入遮罩（二值，`cell_masks > 0`）：
  - **(a) 像 precision：** 遮罩像素中 hematoxylin ≥ t 的比例。問的是「遮罩說有核的地方，圖上真的畫了核嗎」。
  - **(b) 像 recall：** hematoxylin ≥ t 的像素中落在遮罩內的比例。問的是「圖上畫出來的核，是不是都在遮罩說的位置」。
  - **(c) 對比：** 遮罩內平均 hematoxylin OD − 遮罩外平均。不需要門檻。
  - **門檻 t = 0.0224**（精確值 0.02243768021416608）：stage-1 D2 的 Youden 規則在 CRC16 真實 H&E 上重算的值（mask_truenuc `data/info.json` 的 `t16`）。規則：正例 = 被 CellViT 和 StarDist 都找到的 Lin 細胞核範圍內的平均 hematoxylin，負例 = 同一個外形搬到沒有細胞的位置；t = 讓「正例判對率 − 負例判錯率」最大的分數。
  - **為什麼 (a)、(c) 當共同主要量、(b) 只描述：** (a) 直接問生成器有沒有在遮罩位置畫核；(c) 不需要門檻，不受「t 是在真實 H&E 上定的、生成圖的顏色可能偏移」影響。(b) 的分母包含遮罩外所有被染色的像素（基質、背景的染色深淺），混進了「整體染色風格」，比較像外觀量，所以只描述。這和 mask_truenuc 的讀法相同（它的規則也只用 (a)、(c)）。
- **只描述的檢查量（不進規則）：**
  - **AUC：** 遮罩內任取一個像素的 hematoxylin 大於遮罩外任取一個像素的機率（hfv `compare.py:nucleus_auc`）。只看排序，對「整張圖統一變深或變淺」不敏感，用來檢查 (a)、(c) 的差是不是只因為某個變體整體染得比較深。
  - **整張圖的平均 hematoxylin OD**、遮罩前景比例（同一張遮罩，各變體相同）。
  - **敏感度門檻：** 用 stage-1 各切片自己的 Youden t（crc36ho 0.021965、crc33sec 0.020252）再算一次 (a)、(b)。
- **參考線：** 真實 H&E 對同一張遮罩的同一個量。真實組織本身也不會完全貼合遮罩（遮罩來自螢光影像的分割，物件定義和 H&E 上看到的核不同），所以參考線不是 1；「變體 − 真實」> 0 代表生成圖比真實組織更貼遮罩。

## 2. 要算的比較

- **配對差：** 每張切片分開算，**不合併兩張切片**。
  - Concat − Grouped、Concat − Grouped + bypass、Grouped − Grouped + bypass；
  - 每個變體（五個都算）− 真實 H&E 參考線。
- 因為所有變體用同一批圖塊、同一張遮罩，「Concat − Grouped」的原始值差 = 兩者「離真實參考線的差距」之差，所以變體之間只比原始值差。
- **量：** 照遮罩畫的 PQ，官方後處理（規則用）；照遮罩畫的 PQ，hfv 舊後處理（描述）；(a)、(c)（共同主要）、(b)、AUC、平均 hematoxylin（描述）、敏感度 t 的 (a)、(b)（描述）。
- **區間：** 圖塊 bootstrap，10,000 次、seed 42：`numpy.random.default_rng(42).integers(0, n, (10000, n))`，圖塊按名稱排序；每張切片一個抽樣矩陣，該切片的所有比較、所有量共用（配對）。95 % 百分位區間（2.5、97.5 百分位）。這和上一份實驗 `cvo_stats.py` 的設定相同。
- **缺值：** 遮罩是空的圖塊，(a)、(c)、AUC 記為缺值；圖上沒有任何 hematoxylin ≥ t 的像素時 (b) 記為缺值。某一量有缺值的圖塊，在該量的所有臂一起排除（保持配對），並回報排除幾張。（目前已知兩組的 Lin 遮罩都沒有空的圖塊；這是從遮罩物件數 `n_lin` 看的，不是讀數。）

## 3. 預先說好的判讀規則（不含任何數字門檻）

對每一個 Grouped 類變體 G（Grouped、Grouped + bypass）分別檢查：

- **「Concat 比 G 更不照遮罩」成立**，當且僅當在 **CRC36 和 CRC33 兩張切片上**，以下三個量的「Concat − G」95 % 區間**都整段在 0 以下**：
  1. 照遮罩畫的 PQ（參考 = 輸入遮罩），官方後處理；
  2. (a)；
  3. (c)。
- 規則只用「參考 = 輸入遮罩」的量（上面三個），每個量另外附「變體 − 真實 H&E 參考線」當背景。論文式 PQ 不進規則、也不計算（§1）。
- **結果 A：** 至少一個 G 成立 → 回報「**a Grouped variant follows masks better**」（寫明是哪一個 G），並列出換的代價：
  - 外觀量：FUD、LPIPS、style_hed（重訓 1 README 的 Fig 2 表；Concat 在這些量上較好或較差都照實列出）；
  - sim2real 應用（apps 1–8、stage 3）全部是在 Concat（形狀庫遮罩生成器，或更早的 CellViT 遮罩生成器）上跑的，換生成器要重跑哪些、哪些結論可能受影響。
- **結果 B：** 兩個 G 都不成立 → 回報「**no evidence Grouped follows masks better**」。其他組合（例如只有 PQ 成立、只有一張切片成立、或方向相反 Concat 較好）都歸在 B，並照實列出各區間，不另下結論。
- **多重比較：** 兩個 G 各要六個區間（三個量 × 兩張切片）同時整段 < 0，是很嚴的「全部成立」條件，所以不另做校正。
- **描述量的用法：** hfv 舊後處理的 PQ、(b)、AUC、平均 hematoxylin、敏感度 t 和規則結果並列。若它們和規則結果方向相反（例如 (a)、(c) 偏向 G，但 AUC 顯示差別只來自 G 整體染得較深），寫成註記；**不改變結果的標籤**。
- **換生成器、重訓，都由使用者決定。** 本實驗只回報。
- **限制（先寫好）：** 只有形狀庫遮罩下的重訓 1 變體；每張圖塊只有一張生成圖（一個 seed），所以區間只反映圖塊間的變異，不包含同一張圖塊換 seed 的變異；Grouped（production）論文原圖已被覆蓋，這裡用的是重訓 1 的新 Grouped 圖。

## 4. 執行

- **Hyak（cheme compute，CPU 4 核，新目錄 `/gscratch/cheme/chiu/fidelity_variants_20261122/`，其他目錄唯讀）：** `scripts/fv_hema.slurm` → `scripts/fv_hema.py`，用和 mask_truenuc 相同的 Python 環境（`crc16-20261024/cellvit/env`，skimage 0.26.0）和同一份 `compare.py`（md5 932ebd72…，Hyak `hfv_lintype_20261118/hfv` 的複本和本 repo 相同）。輸出 `hema_{36,33}.csv`（每張圖塊、每臂一列）。資料量小（約 2,400 張 256 px 影像），算 hematoxylin 只需幾分鐘；放在 Hyak 是為了和 t 的來源用同一個環境，並且免去下載。
- **閘門（任何一項失敗就 STOP，不重送同樣的工作）：**
  1. 10 個 zip 的 sha256 = 上一份實驗 `zips_manifest.json` 的值。
  2. 每組五個 zip 的圖塊名稱相同，且等於 `retgt_<set>/` 的檔名、也等於 `tiles_<group>.csv` 的圖塊（本機檢查）。
  3. 形狀庫遮罩 = 重訓 1 評估包 `evalpack_<set>.tar` 的 `orion-<set>/exp_channels/cell_masks/<tile>.png`，逐像素相同（全部 400 張）。
  4. hematoxylin 程式路徑：用本腳本在 mask_truenuc 的前 8 張 CRC16 圖塊重算「真實 H&E 對 Lin 寬鬆核形」的 (a)、(b)、(c)（t16），和 `mask_truenuc_20261122/full/tiles.csv` 的 `L1_{a,b,c}_t16` 相差 ≤ 1e-9。
- **本機：** `scripts/fv_stats.py`（讀 `tiles_{36,33}.csv` 和 `hema_{36,33}.csv`，算配對差與區間，套用 §3 的規則）→ `data/stats.json`；`scripts/fv_fig.py`（共用 figstyle，Nat Commun 規格）→ `figures/fig_fv.{png,pdf}`。
- **圖（預定）：** 各量的配對差（Concat − Grouped、Concat − Grouped + bypass、Grouped − Grouped + bypass）在兩張切片上的點和區間；另一個 panel 是各變體 − 真實參考線。英文標示。
- **結果：** 另寫一份 `RESULTS.md`（繁體中文、附圖和 `data/stats.json`），不改這份預先登記；任何偏離記在 RESULTS 的「偏離」一節。
