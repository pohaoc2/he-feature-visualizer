# hfv_retrain_20261118：用新的細胞通道重新訓練四個生成器變體，並重跑評估

這份報告與舊的結果分開放。論文（`share-space-paper/`）是依據，這裡只讀不改。結果寫在最後一節，要等訓練和評估完成才會填入。

## 方法：開始執行前就定下的規則

這一節的規則都在任何新結果出現之前寫好並 commit（2026-11-18）。之後若有任何改動，會寫在「偏離紀錄」，並註明是在看到什麼結果之後才改的。

### 訓練資料

- **CRC33_01**：新的 7 個細胞通道。
  - 細胞類型用 CELESTA 加上 Lin 2023 Fig. 3d 的規則判定。
  - 增殖狀態用 Ki-67 的 Lin 閘值判定。
  - 死細胞由 CellViT 判定。
  - 細胞核用 loose nuclei 版本。
  - 有墨水的 tile 排除後剩 9,419 張。
  - 血管、氧、葡萄糖三個通道沿用原本的檔案。
  - 遮罩的 VAE latent 用新的 cell_masks 重新算。
  - UNI 和 H&E 的 VAE latent 沿用已存的檔案。
- **CRC36**：
  - 用相同流程從 Lin CRC36 的資料建立通道。
  - 用 CellViT 判定死細胞：438 個死細胞物件，對到 350 個 Lin 細胞。
  - 血管、氧、葡萄糖通道用 `crc36_render_20261118/mx/` 的流程，先確認在 CRC33 上能重現原本的檔案，再用到 CRC36。
  - tile 名稱一律加上前綴 `c36_`，避免和 CRC33 的 tile 撞名。
- **CRC36 保留區（不參加訓練）**：
  - 選法：16 個 CELESTA k-d 矩形裡，TUM 類細胞比例最接近整張切片的那一個（`combine.py`）。
  - 和保留區相鄰（8 鄰域）的 tile 當作緩衝帶，也不參加訓練。
- **CRC34（patient 2）**：完全保留，不參加訓練。
- **論文的切片說明（只在這裡記錄，不改論文）**：C33 是 MMR 缺陷；C36 是 hypermutant，沒有做 MMR 檢測；兩者都是低惡性度；C34 保留不用。C36 和 C34 都位於盲腸。

### UNI-2h 前處理

- 所有重新計算的 UNI 特徵（CRC36 訓練特徵、評估時對生成圖重算的 UNI）一律用 `uni-2h/config.json` 的設定：bilinear 縮放、ImageNet 的 mean/std、crop_pct 1。
- 這和 sim-exp-validation 的 `forward.py:load_uni_extractor`（commit ea5d5bd9，2026-09-30，"Fix UNI preprocessing"）是同一個定義。
- 在 L4 上用 50 張 CRC33 tile 檢查：這套設定重現已存特徵的最大絕對差是 1.1e-4。PixCell HEAD 的 `UNI2hExtractor`（commit 4bf85d7 之後改用 timm 內建設定）重現不了，最小 cosine 只有 0.588。
- sim-exp-validation 的 D 流程本來就用正確的前處理，不受影響。

### 訓練

- 四個變體（Concat、Grouped、Grouped+bypass、Per-channel）用 PixCell cb62fa7 的程式和設定，不做修改。
- 有效批次 80、20 個 epoch、seed 42。
- micro-batch 依 GPU 而定，不足的部分用梯度累積補到 80。EMA 只在每個 optimizer step 更新一次。
- GPU 用 Colab 的 A100：probe 顯示 A100 比 L4 快 3.6–3.9 倍。
- 每個 epoch 結束時存 checkpoint，每 2 個 epoch 上傳到 S3；中斷時從最近上傳的 checkpoint 接續。接續後各 epoch 的資料順序會和不中斷的情況不同。

### 15 條件通道群組消融（Concat）

1. 開跑前先用 50 張 tile × 1 個條件實測速度，再把估計表換成實測數字。
2. 規模：15 個群組組合 × （300 張 paired + 300 張 unpaired）。**n = 300 事先固定，之後不依結果補到 1,000 張。** 依信賴區間是否跨過 0 再決定補多少，屬於 optional stopping，會讓偽陽性率膨脹。所有信賴區間都照實報告；跨過 0 只代表「無法區分」。
3. 變體用 Concat，複製論文的設計。用四方比較勝出的變體再做一次消融，屬於另一個後續問題，不在這次執行範圍內。
4. 每個條件的基準圖都在同一次執行裡重新生成，不用舊的快取。論文的 handover 記錄過：同樣設定重跑，ΔE 差了 10.9。
5. tile 分兩組，分開報告：
   - **主要結果**：CRC36 保留區和 CRC34（patient 2）的 tile。這些 tile 都沒有參加訓練。
   - **「訓練 tile，可和論文比較」**：論文原本的 tile 清單，即 `concat_ablation_1000/tile_lists/paired_1000_tile_ids.txt` 的前 300 張，和 Fig 4 相同。unpaired 的風格來源用 `unpaired_mapping.json`（seed 42）。這 300 張裡有 1 張（`10496_17664`）在新的 CRC33_01 資料中因墨水被排除、沒有新通道，所以這組實際是 299 張。unpaired 的風格只用 UNI，5 張風格來源 tile 是墨水 tile 也照用。
   - 主要結果那組的取樣方式（提議，等協調者確認後才算定案）：
     - CRC36 保留區：保留區 tile 不超過 300 張就全部用；超過就以 seed 42 隨機取 300 張。
     - CRC34：照論文跨病人分析的做法，取排序後的前 300 張。
     - unpaired 的風格來源：在同一組 tile 內以 seed 42 做一個沒有不動點的排列（derangement），讓每張 tile 的風格都來自另一張 tile。
   - 兩份檔案都從 `s3://bagherilab-working/pohao/share_space/04_inference/inference_output_20260527.zip` 用 range 讀取取出，共讀了約 51 MB。

## 偏離紀錄

（尚無）

## 結果

（訓練和評估完成後填入）
