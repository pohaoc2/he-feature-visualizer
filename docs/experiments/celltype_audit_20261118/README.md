# 生成器訓練資料的細胞分型稽核（celltype_audit，2026-11-18）

**問題：**多通道 H&E 生成器（PixCell ControlNet）是用 hfv 自己做的細胞分型當作條件訓練的。這套分型和 Lin et al. 2023 發表的分型差多少？應該換成什麼？

**範圍：**
- 只做稽核和書面提案。
- 沒有重新訓練生成器，沒有改 hfv 的正式程式碼（`stages/` 等），也沒有動任何正在跑的 Hyak 工作。
- 全部是事後描述，不是預先登記的實驗。

**資料位置：**
- 計算都在 Hyak `/gscratch/cheme/chiu/he_celltype_audit_20261118/`。
- CRC33_01 的 H&E、多重螢光影像和逐細胞表格都只放在 Hyak，沒有下載到 Mac。
- 本目錄只放小的摘要 json、腳本和圖。

**名詞：**
- **hfv 分型**：`stages/assign_cells.py`（commit e88f450）產生的 cancer／healthy／immune 與 prolif／nonprolif／dead 標籤，也就是生成器訓練時的條件通道。
- **Lin 閘門**：每個標記一個強度門檻，高於門檻算陽性。用 Lin 公開的 GMM 方法求，求不出來時改用分位數，讓陽性比例等於 Lin 發表的比例（`scripts/lin_gate.py`）。
- **舊規則**：CD45+ → 免疫；否則 Pan-CK+ 或 E-cad+ → 上皮（算 cancer）；其他 → healthy。
- **組織步驟**：上皮細胞如果落在 Lin H&E 組織分類的 NORM（正常黏膜）格子裡，改成 healthy。
- **rule (a)**：sim-exp-validation `scripts/analysis/tcmvp/tcfix_rule.py`。在組織步驟之外，再規定 TUM（腫瘤）格子裡的非免疫細胞，除非 SMA+ 或 CD31+，否則都算 cancer。
- **TUM／NORM／STR／LYM**：Lin H&E CNN 每 112 µm 一格的組織類別，分別是腫瘤、正常黏膜、間質、淋巴。

## 1. 盤點

### 1.1 生成器與訓練資料

| 項目 | 內容 | 來源 |
|---|---|---|
| 訓練切片 | **CRC33_01**（Lin 的 C33 切片第 1 片）。10,379 個 256 px tile，0.325 µm/px | `data/orion-crc33/metadata/exp_index.hdf5`；§1.3 逐像素確認 |
| 模型 | `PixCell_ControlNet_XL_2_UNI_Concat` + `RawConditioningPassthrough` | `PixCell/configs/config_controlnet_exp_a1_concat.py`（底層 `config_controlnet_exp.py`） |
| 條件通道（10 個） | cell_masks；cell_type_healthy／cancer／immune；cell_state_prolif／nonprolif／dead；vasculature；oxygen；glucose | 同上；`PairedExpControlNetData` |
| 訓練程式 | `PixCell/train_scripts/train_controlnet_exp.py`。bs 80，20 epochs，AdamW lr 1e-5，warmup 200 | config |
| 訓練紀錄 | 2026-04-28 13:52 → 14:29：**38 分鐘，1 張 GPU（Tillicum g013），2,600 步**，約 90 樣本／秒 | `s3://bagherilab-working/pohao/share_space/03_checkpoints/tillicum_95470/concat_95470_0/train_log.log` |
| 分型輸出 | `processed_crc33/cell_assignments.csv`（263,446 個 CellViT 物件）、`cell_types/`、`cell_states/` | S3 `processed_crc33` zip |
| 通道和分型是否一致 | 抽 500 個 tile，在物件中心讀訓練用 `exp_channels`：類型和狀態都和 `cell_assignments.csv` 一致（98.95 %；其餘 1.05 % 落在沒有填色的像素） | `data/chancheck.json` |

**哪些通道會因為重新分型而改變：**
- cell_type_* 和 cell_state_* 會變。
- cell_masks 只在改用 Lin 的分割時才會變。
- vasculature、oxygen、glucose 由 CD31、Ki67、PCNA 的影像強度直接算，不依賴細胞分型（`stages/multiplex_layers.py`），所以不變。

### 1.2 hfv 分型的做法（`stages/assign_cells.py`）

| 步驟 | 做法 | 有沒有發表依據 |
|---|---|---|
| 物件 | CellViT 從 H&E 切出的細胞核 | CellViT 有發表；但已知約 30 % 腫瘤／上皮物件是黏在一起的多個核（記憶 cellvit-undersegmentation） |
| 多重螢光特徵 | 在 CellViT 物件本身上量（`match_distance` 最大 2e-13 px） | — |
| 類型先驗 | CellViT 類別 → 先驗：Neoplastic → cancer 1；Inflammatory → immune 1；Connective → healthy 1；Epithelial → cancer／healthy 各 0.5；background、dead → 三類各 1/3 | 自訂 |
| 融合 | 先驗和多重螢光細分類的 softmax 加權相加；各 CellViT 類別的權重 {0: .5, 1: .3, 2: .7, 3: .4, 4: .5, 5: .5}；模型信心 < 0.40 時只用先驗 | 自訂，數字都是手設 |
| 平手 | 先驗平手時 `_argmax_label` 回傳第一個鍵 "cancer" | 程式行為，不是設計 |
| 組織類別 | 沒有。Pan-CK／E-cad 陽性的上皮全部算 cancer | — |
| 增生狀態 | Ki67 ≥ 整張切片 Ki67 的第 75 百分位 → prolif，所以一定約 25 % 是 prolif（`compute_state_thresholds`） | 自訂 |
| 死亡狀態 | CellViT 類別 4（dead） | CellViT |

### 1.3 Lin 的公開資料（CRC33_01）

| 資料 | 有無 | 用法 |
|---|---|---|
| 逐細胞表格（MCMICRO：每個細胞每個通道的平均強度、形態） | 有，443,514 個細胞 | 本稽核的 Lin 端 |
| 細胞分割 mask | 有（UNMICST2 核 mask 再膨脹 5 px，所以是「細胞」mask，不是核） | 把 hfv 物件對到 Lin 細胞 |
| 配準後的 H&E | 有 | 確認就是生成器的訓練 H&E |
| H&E 組織類別圖 | 有（`labsyspharm/orion-crc` 的 `HE_patch_prediction`，112 µm 一格） | TUM／NORM 等組織類別 |
| 逐細胞的細胞類型標籤 | **沒有公開**。只有整個 cohort 的陽性比例摘要（sumAllsample） | 用來校準閘門 |
| 47 GB 多重螢光 OME-TIFF | 沒有下載，因為逐細胞表格已經有每個通道的平均強度 | — |

**逐像素檢查：**隨機抽 50 個 hfv tile，和 Lin 配準後 H&E 在相同 (y0, x0) 的區塊逐像素比較，最大平均絕對差 = 0.0。hfv 的全域像素座標就是 Lin 的像素座標。另外隨機抽 200 個 Lin mask 標籤，其中心和 CSV 中心的距離中位數是 0 px。（`data/lookup.json` 的 G_frame、G_id）

### 1.4 Lin 2023 怎麼分細胞

Lin JR et al. 2023, *Nature Cancer* 4:1036–1052, doi 10.1038/s43018-023-00576-1。（是 Nature Cancer，不是 Cell。）

| 位置 | 內容 |
|---|---|
| Methods「Orion image processing and data analysis」 | 用 MCMICRO 做單細胞分析；UNMICST2 分割，細胞 mask 是核 mask 膨脹 5 px；每個細胞量每個通道的平均強度和形態特徵 |
| Methods「IFM computation from Orion data」 | 標記用 **Gaussian mixture model 分成高／低**，再人工檢查確認（文中寫在 pan-cytokeratin、CD3ε、CD8α 上） |
| Extended Data Fig. 1d | 大腸癌的細胞類型判定樹狀圖：免疫（CD45）→ 細分；上皮（Pan-CK／E-cad）；間質（SMA、CD31）。**只有圖，沒有寫門檻數值** |
| Results「Integrated analysis of IF and H&E images」、Fig. 3c | 40 個 CRC 檢體中有 6.5–42 %（中位數 16 %）的細胞核無法用螢光標記判定類型（「unidentifiable」） |
| Fig. 3d–e | 這些細胞用 **H&E CNN**（Kather 的九類組織模型，ref. 36）補判：超過 50 % 落在平滑肌、間質纖維母細胞或脂肪 |
| Fig. 3f | C26 的鋸齒狀腺瘤 Pan-CK 和 E-cad 染色弱，「Low staining intensity … interferes with IF-based cell-type calling」 |

**重點：**
- Lin 的分型是**螢光閘門加樹狀圖**，判不出的細胞**用 H&E 組織類別補**。
- Lin 沒有把「上皮」再分成「正常」和「腫瘤」。這一步要靠 H&E 組織類別，也就是我們的組織步驟。
- 角蛋白弱的腫瘤細胞會被漏掉，Lin 自己就指出了（Fig. 3f）。

### 1.5 本稽核的 Lin 閘門

- Lin 發表的 C33 陽性比例是 **CRC33_01 和 CRC33_02 兩片合起來**算的。對兩片合併重算的比例和發表值相差 ≈ 0（`data/identity_c33.json`），單算任一片則差 5–17 %。所以閘門是在兩片合併上求的（偏差 1，見 §5）。
- `data/lin_gates_crc33.json`：

| 標記 | 門檻 | 來源 | CRC33_01 陽性比例 |
|---|---|---|---|
| Pan-CK | 280.2 | GMM（合併兩片比例 0.457，發表 0.392） | 0.406 |
| E-cadherin | 275.0 | GMM | 0.403 |
| CD45 | 565.8 | 分位數（GMM 對不上發表比例） | 0.259 |
| Ki67 | 331.5 | 分位數 | 0.038 |
| SMA | 455.7 | 分位數 | 0.080 |
| CD31 | 410.3 | 分位數 | 0.038 |

## 2. 對應：hfv 物件 ↔ Lin 細胞

**方法：**
- 主要對應：讀 hfv 物件中心像素上的 Lin mask 標籤。不需要任何半徑。
- 次要對應（檢查用）：最近的 Lin 中心，半徑 r = √(Lin 細胞面積中位數 / π) = 11.74 px（3.82 µm）。這個半徑是從資料算的。
- 只比較落在 10,379 個訓練 tile 裡的 Lin 細胞（417,694 個）。

| 指標 | 數值 |
|---|---|
| hfv 物件 | 263,446 |
| hfv 物件中心落在 Lin 背景（沒有對到細胞） | 10.4 %（CellViT background 類 29 %，neoplastic 7.7 %，epithelial 12 %） |
| 兩個以上 hfv 物件對到同一個 Lin 細胞 | 8.6 % |
| hfv 物件裡面包含 ≥ 2 個 Lin 細胞中心（黏在一起） | 16.5 %；neoplastic／epithelial 22.4 %；NORM 23 %，LYM 28 % |
| Lin 細胞沒有任何 hfv 物件中心 | 46 %（上皮 62 %） |
| 最近鄰距離分位數（5／25／50／75／95 %，px） | 1.29／2.25／4.09／7.51／12.23 |
| 半徑內對到 | 89 %，和主要對應一致 97.3 % |
| 面積中位數 | Lin 細胞 mask 45.7 µm²，hfv 核 23.8 µm² |

**判讀：**
- 兩套物件大致對得上，但 hfv 少切了很多上皮細胞：62 % 的 Lin 上皮細胞沒有自己的 hfv 物件，常和鄰居黏成一個物件。
- 這和已知的 CellViT 分割不足一致。

## 3. 結果

圖：`figures/fig1_audit.png`（類型、角蛋白弱的腫瘤細胞、正常上皮、CellViT 類別、增生、墨水），`figures/fig2_overview.png`（整張切片）。

### 3.1 這張切片大部分是正常黏膜

訓練 tile 內 Lin 細胞所在的組織類別：

| NORM | DEB | STR | MUS | TUM | LYM | 其他 |
|---|---|---|---|---|---|---|
| 59.4 % | 13.8 % | 10.6 % | 7.6 % | **4.4 %** | 2.8 % | 1.4 % |

- 整張切片 TUM 只有 300 格（112 µm），NORM 有 2,076 格；訓練 tile 涵蓋其中 265 和 1,985 格。
- 生成器學到的「cancer」通道，主要來自正常黏膜的上皮（§3.3）。

### 3.2 (a) 各組織類別的類型比例

所有訓練 tile：

| 分型 | cancer | healthy | immune |
|---|---|---|---|
| hfv | 41.9 % | 44.0 % | 14.1 % |
| Lin 舊規則 | 36.3 % | 37.3 % | 26.4 % |
| 舊規則 + 組織步驟 | 4.2 % | 69.4 % | 26.4 % |
| rule (a) | 4.8 % | 68.8 % | 26.4 % |

Lin 樹狀圖四類：上皮 36.3 %，免疫 26.4 %，間質（SMA+ 或 CD31+）8.9 %，判不出 28.4 %。

依組織類別（cancer／healthy／immune，%）：

| 組織 | hfv | 舊規則 | + 組織步驟 | rule (a) |
|---|---|---|---|---|
| NORM | 54／29／16 | 54／12／34 | 0／66／34 | 0／66／34 |
| TUM | 50／30／20 | 56／20／24 | 56／20／24 | 69／7／24 |
| STR | 15／78／7 | 3／88／9 | 3／88／9 | 3／88／9 |
| LYM | 13／23／64 | 7／8／85 | 7／8／85 | 7／8／85 |

hfv 的 cancer 物件有 64 % 在 NORM，只有 5 % 在 TUM（`data/extra.json`）。

**和 hfv 一致的比例（有對到的細胞）：**舊規則 0.64，加組織步驟 0.48，rule (a) 0.48。NORM 從 0.61 掉到 0.30，因為 hfv 把正常上皮叫 cancer。

### 3.3 (b) 角蛋白弱的腫瘤細胞、(c) 正常上皮

| 指標 | hfv（有對到的） | 舊規則 | + 組織步驟 | rule (a) |
|---|---|---|---|---|
| (b) TUM 裡 CD45−、Pan-CK−、E-cad−、SMA−、CD31− 的細胞被叫 healthy（n = 2,392，占 TUM 非免疫細胞 17 %；hfv 對到 1,400） | 0.66 | 1.00 | 1.00 | 0 |
| (c) NORM 裡的上皮（CD45−，Pan-CK+ 或 E-cad+）被叫 cancer（n = 133,770；hfv 對到 54,468） | **0.84** | 1.00 | 0 | 0 |

**判讀：**
- (c) 是 hfv 最大的問題：正常黏膜的上皮 84 % 被當成 cancer。
- (b) 在 hfv 裡由 CellViT 決定：CellViT 說 connective 的（903／1,400），100 % 變 healthy；其他的大多變 cancer。
- **(b) 在這張切片上不一定是錯的：**這些 TUM「判不出」細胞的 Pan-CK 中位數 234，和間質一樣（234）；Ki67+ 只有 0.13 %；面積 61.9 µm²，比 TUM 上皮（42.6）大。這和 CRC16 不同（CRC16 的這群細胞核和上皮一樣大，見 sim-exp-validation `celltype_check.md`）。在這裡它們比較像間質，rule (a) 把它們叫 cancer 可能是錯的。這要靠 CELESTA／Astir（§4）判斷。

### 3.4 hfv 的融合實際上怎麼決定類型

| CellViT 類別 | 占物件 | 最終 cancer／healthy／immune | 只用先驗（多重螢光被忽略） | 多重螢光模型自己說的 |
|---|---|---|---|---|
| neoplastic | 24 % | 100／0／0 | 38 % | cancer 43 %、healthy 43 % |
| connective | 40 % | 0／100／0 | 67 % | healthy 85 % |
| epithelial | 16 % | 82／11／7 | 65 %，**全部因為平手變 cancer** | healthy 76 %、cancer 17 % |
| inflammatory | 13 % | 2／7／91 | 48 % | immune 42 % |
| background | 5.7 % | 73／15／12 | 61 %，全部因為平手變 cancer | healthy 76 % |
| dead | 0.3 % | 61／17／22 | 58 %，全部因為平手變 cancer | healthy 74 % |

**判讀：**
- 對 neoplastic 和 connective（共 64 % 的物件），多重螢光完全改變不了 CellViT 的判斷。
- 約 37.6k 個 cancer（hfv cancer 的 34 %）來自先驗平手時的預設值，不是任何標記的證據。
- 所以 hfv 的類型通道主要是「CellViT 的 H&E 類別」，而生成器的工作正是從 H&E 學圖像。條件通道和輸出圖像出自同一張 H&E，有循環的風險。

### 3.5 增生狀態

| 組織 | hfv prolif（Ki67 ≥ 第 75 百分位） | Lin Ki-67 閘門 |
|---|---|---|
| 全部 | 25 % | 4.0 % |
| NORM | 43 % | 5.5 % |
| TUM | 32 % | 12 % |
| STR | 2.7 % | 0.7 % |
| LYM | 15 % | 2.9 % |

- 對到的細胞：hfv 26 % vs Lin 2.5 %，一致 0.76。
- hfv 的第 75 百分位規則保證約 25 % 是 prolif，比 Lin 閘門多約 6 倍。

### 3.6 (d) 墨水

**墨水判定：**和 sim-exp-validation `report_20261031.md` §10b 的 bridge 重擬合相同：`style_pool.PIGMENT_LEVEL = 60`，`MAX_PIGMENT_FRACTION = 0`。物件層級用中心 3×3 px 的最小灰階。

| 指標 | 數值 |
|---|---|
| 有墨水像素的 tile | 961／10,379（9.3 %）；墨水 > 0.5 % 的 405 個 |
| 中心在墨水下的 hfv 物件 | 5,482；79 % 落在 Lin 背景（Lin 在墨水下幾乎切不到細胞） |
| 這些物件的類型 | cancer 36 %、immune 52 %、healthy 12 % |
| 這些物件的狀態 | **prolif 0.02 %**（無墨水 tile：26.4 %） |
| 這些物件所在組織 | TUM 49 %、DEB 17 %、BACK 16 % |
| 中心在墨水下的 Lin 細胞 | 只有 704 個；91 % 判成上皮；Ki67+ 0；Hoechst 中位數 2,544（其他 4,122） |
| 訓練通道（500 tile 抽樣）prolif 面積比例 | 墨水 tile 0.0019，無墨水 tile 0.0465 |

**判讀：**
- 墨水主要蓋在這張切片的腫瘤區上。
- 墨水下的物件幾乎全部是 nonprolif，因為墨水擋住螢光，Ki67 讀不到（Hoechst 也偏低）。
- 生成器因此學到「墨水外觀 ↔ 成群的非增生核」，這就是 §10b 黑色團塊的來源。§10b 只重擬合 bridge，生成器本身沒有重訓。
- 墨水下 Lin 細胞 91 % 是上皮，可能是腫瘤，也可能是墨水造成的假訊號，沒有確認。

## 4. CELESTA／Astir（CRC33_01）

（待補：工作 43360813–43360817 執行中。）

## 5. 偏差與限制

1. **閘門在兩片合併上求。**原本打算只用 CRC33_01，但 Lin 發表的 C33 比例是兩片合起來的（§1.5）。只用 CRC33_01 的閘門另存為 `lin_gates_crc33_01only.json`（在 Hyak，未使用）。
2. **只抽 500 個 tile 檢查訓練通道。**整套 10,379 個 tile 的 raster 在計算節點上寫小檔太慢（GPFS 約 1 檔／秒），改成抽樣並打包成 tar。抽樣種子 20261118。
3. **慢節點。**下載工作 43359509 在 n3305 上停住，已記錄到 `slow_nodes.sh` 並取消；43359731 在 n3013 也停住，改成上述抽樣後 43360003 在 1.5 分鐘完成。
4. **imagecodecs。**Lin 的 H&E 是 JPEG 壓縮，cellvit 環境沒有 imagecodecs，裝在 `$W/pylib`（`--no-deps --target`），只給本稽核用。
5. **物件不是一對一。**hfv 物件和 Lin 細胞的分割不同（§2）。類型比較用「hfv 物件中心落在哪個 Lin 細胞」，黏在一起的物件只算到中心所在的那一個細胞。
6. **TUM 很少。**訓練 tile 內只有 4.4 % 的細胞在 TUM，(b) 的樣本只有 2,392 個細胞。
7. 組織類別是 Lin 的 H&E CNN 判的，不是病理醫師判讀。

## 6. 提案：新的分型規格（不執行，由使用者決定）

### 6.1 原則

- 生成器的科學貢獻是「多通道 → H&E」，不是分型方法。所以分型照 **Lin 2023 發表的方法**做，只在 Lin 自己指出的弱點加上修正。
- 分型只用多重螢光和 Lin 公開的 H&E 組織類別，**不用 CellViT 的 H&E 類別**，避免條件通道和輸出圖像循環（§3.4）。

### 6.2 規格

| 步驟 | 新做法 | 依據 | 對訓練資料的影響（CRC33_01） |
|---|---|---|---|
| 0. 墨水 | 有墨水像素的 tile 不進訓練：`PIGMENT_LEVEL = 60`，`MAX_PIGMENT_FRACTION = 0` | 和 §10b bridge 同一規則 | 移除 961 個 tile（9.3 %） |
| 1. 閘門 | Lin GMM 閘門，對不上發表比例時用分位數，讓陽性比例等於發表的 C33 比例；在 CRC33_01 + CRC33_02 合併上求 | Lin Methods「IFM computation」；sumAllsample | 已求好（§1.5） |
| 2. 樹狀圖 | CD45+ → 免疫；否則 Pan-CK+ 或 E-cad+ → 上皮；否則 SMA+ 或 CD31+ → 間質；其他 → 判不出 | Lin Ext Data Fig. 1d | 上皮 36 %，免疫 26 %，間質 9 %，判不出 28 % |
| 3. 組織步驟 | 上皮在 NORM → healthy；上皮在其他組織 → cancer | Lin Fig. 3d 用 H&E CNN 補判；`tcfix_rule.py` | cancer 從 42 %（hfv）降到約 4–5 % |
| 4. 判不出的細胞 | 預設 healthy（Lin Fig. 3d：超過一半是間質、平滑肌、脂肪）。**TUM 裡的判不出細胞**是否改成 cancer（rule (a)），等 CELESTA／Astir 在這張切片的結果決定（§4） | Lin Fig. 3d、3f | 影響 2,392 個 TUM 細胞（cancer 4.2 % ↔ 4.8 %） |
| 5. 增生 | Ki67 用 Lin 閘門（步驟 1）；不再用第 75 百分位 | 同步驟 1 | prolif 從 25 % 降到約 4 % |
| 6. 死亡 | 維持 CellViT dead 類別（Lin 的面板沒有死亡標記，Lin 也用 H&E 補判螢光判不出的東西） | Lin Fig. 3b、3d | 不變（0.3 %） |
| 7. 物件和標籤轉移 | **選項 A（建議）**：保留 CellViT 物件和 cell_masks 通道；每個物件取中心所在 Lin 細胞的標籤；中心在 Lin 背景的物件（10.4 %）不填類型。**選項 B**：改用 Lin 的分割 mask（細胞 mask，不是核），cell_masks 也一起換 | §2 | A：cell_masks 不變；B：要重建 cell_masks，而且模擬端輸出的遮罩要改成細胞輪廓 |

**需要使用者決定：**
- 選項 A 或 B。B 改變了模擬端要提供的遮罩定義，影響比較大。
- TUM 判不出的細胞歸哪一類（等 §4）。
- 中心在 Lin 背景的物件，是不填類型，還是保留 CellViT 判斷。

### 6.3 成本（粗估）

| 工作 | 計算 | 說明 |
|---|---|---|
| 重新分型 | Hyak CPU，數十分鐘 | 閘門和對應已經做好（本稽核 `audit` 工作 1.3 分鐘，對應用的 Lin mask 讀取在之前的工作完成） |
| 重建 cell_type_*／cell_state_* raster | Hyak CPU；10,379 個 tile（扣掉墨水後 9,418） | 要在 hfv 寫一個轉移標籤的新步驟（改正式程式碼，需使用者同意）；寫小檔要用本地 scratch 或 tar，避免 GPFS 太慢 |
| VAE／UNI 特徵 | 不用重算 | H&E 沒變；墨水 tile 只要從 `exp_index` 拿掉 |
| 重新訓練生成器 | 原始訓練：1 張 GPU 38 分鐘、2,600 步 | Tillicum 預算約剩 $2；Hyak l40s／a100 或 Colab L4 的速度沒有量過，要先跑一小段計時 |
| 驗證 | GPU 少量 | 用固定的條件 tile 比較新舊生成器 |

**下游要重做的：**
- bridge（細胞組成 → 參考條件）重擬合；
- 所有 G(SIM) 生成臂重新生成；
- 所有用到生成圖的 UNI 數字（dev／check／held）；
- 引用這些數字的報告和投影片。

**不用重做的：**真實 H&E 的 UNI 參考值（它們是從真實 H&E 算的，與分型無關）。

## 7. 資料量

抓到 Hyak 的資料（Mac 上只有小摘要和圖）：

| 來源 | 大小 |
|---|---|
| processed_crc33（S3 zip，範圍下載：表格、CellViT JSON、抽樣 raster） | 約 0.24 + 0.04（停住的工作，部分）+ 0.26 GB |
| orion-crc33（metadata、抽樣 exp_channels） | 0.08 GB |
| Lin 公開資料：CRC33_01 CSV 222.8 MB、mask 84.5 MB、H&E 390.3 MB；CRC33_02 CSV 293.4 MB | 0.99 GB |
| H&E 組織類別圖（GitHub） | 約 83 KB |
| **總計** | **約 1.6 GB** |

47 GB 的多重螢光 OME-TIFF 沒有下載。

## 8. 檔案

| 路徑 | 內容 |
|---|---|
| `scripts/fetch.slurm`、`scripts/zip_extract.py` | 下載（S3 zip 範圍讀取，只抽需要的成員） |
| `scripts/lin_gate.py`、`findcutoff.py`、`lin_sumall_col.py`、`tcfix_rule.py` | 從 sim-exp-validation 複製；`lin_gate.py` 加了多片合併 |
| `scripts/lookup.py` | 逐像素檢查、hfv ↔ Lin 對應、墨水灰階 |
| `scripts/audit.py`、`audit.slurm` | §2–§3 的所有數字 → `data/audit.json` |
| `scripts/chancheck.py` | 訓練通道 vs 分型輸出 → `data/chancheck.json` |
| `scripts/extra.py` | CellViT 類別 × 組織 → `data/extra.json` |
| `scripts/pub33.slurm` | CELESTA／Astir（§4） |
| `scripts/fig_audit.py`、`fig_overview.py`、`figstyle.py` | 圖 |
| `data/*.json` | 小摘要；逐細胞資料只在 Hyak |
