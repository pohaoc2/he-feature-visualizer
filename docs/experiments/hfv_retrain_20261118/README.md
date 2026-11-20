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
  - 有墨水的 tile 排除後剩 9,418 張（訓練索引 `exp_256` 的實際數量；先前寫成 9,419 是筆誤）。
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
  - 結果（`combine.py`，2026-11-18）：選中第 13 號矩形（TUM 比例 0.454，整張切片 0.440）。17,872 張保留 tile 裡，保留區 1,092 張、緩衝帶 129 張、訓練 16,651 張。
- **訓練集合計**：CRC33_01 9,418 張 + CRC36 16,651 張 = 26,069 張。批次 80 時每個 epoch 326 步，20 個 epoch 共約 6,520 步。CRC36 部分是 `inputs/c36train.tar`（12,661,534,720 bytes，md5 c562541460dfc266a14f4e8c8a46af5c）。
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
- **Concat 改在 Tillicum 的一張 H200 上訓練（協調者 2026-11-18 決定）**：sim2real 下游用的是 Concat，所以先跑它；兩次申請 Colab A100 都退回 L4，L4 那次在開始訓練前就關掉，沒有寫出任何 checkpoint（S3 上只有一個 503 bytes 的 `log_tr_concat`，沒有 `ckpt_tr_concat_*`）。之後（使用者 2026-11-18 決定）其他三個變體也改到 Tillicum，各用一張 H200，三個作業同時送出（`tillicum/tr_variant.slurm`，run name 沿用 Colab 的 tr_grouped / tr_bypass / tr_perch，S3 位置相同）。原因是這個 Colab 帳號一次只能開一個付費 GPU session。Colab 上的 Grouped 在開始訓練前就停掉了，Colab 的任何產出都不沿用。三個作業第一次送出時不加 `--fetch-resume`，確保不會接到 Colab 可能留在同名位置的東西；只有撞到時間上限、需要接續時才加。四個變體都在 H200 上訓練，和原本預定的 Colab A100 不同（偏離）。
  - 和 Colab 相同的部分：PixCell cb62fa7、同一份設定、同一個 `train_wrap.py`（有效批次 80 = micro 80 × 累積 1、20 epoch、seed 42）、checkpoint 上傳到同一組 S3 位置 `ckpt_tr_concat_eNN_*`（Colab 可以用 `--fetch-resume` 接續）、不設定 TF32（Colab 和 PixCell 都沒有設定，維持 torch 預設值）。GPU 型號寫在 log 裡。
  - 預訓練權重：用 Tillicum 上原有的 `/gpfs/projects/bagherilab/pohaoc2/models/pretrained_models`，不複製。Colab 下載的四個權重 tar 中每個檔案（pixcell-256 3 個、pixcell-256-controlnet 1 個、sd-3.5-vae 2 個、uni-2h 2 個）在 Tillicum 上都有，而且 md5 相同。Tillicum 那邊另外多了 HF 快取、README 等檔案。PixCell 讀的模型資料夾只用逐檔 symlink 放進 Colab 那幾個檔案，多出來的檔案不會被讀到。
  - Python 環境：用 Tillicum 上原有的 `envs/pixcell`，不重建。torch 2.10.0+cu128、torchvision 0.25.0、xformers 0.0.35、diffusers 0.36.0、transformers 5.0.0、timm 1.0.26、accelerate 1.13.0、mmcv 1.7.0、numpy 2.0.2 都和 Colab 的版本表相同。不同的只有工具類套件的小版本（tqdm、requests 等 31 個），另外少了 scikit-learn、joblib、threadpoolctl、cloudpickle、narwhals；PixCell 訓練程式沒有用到這幾個（`train_scripts.train_controlnet_exp` 在這個環境可以正常 import）。
  - 資料：和 Colab 同一批 S3 輸入，用一天有效的 presigned 連結下載，Tillicum 上不放 AWS 長期金鑰。舊的 `datasets/orion-crc33`（舊的細胞分類）不使用。壓縮檔一邊下載一邊解開，不在硬碟上同時留壓縮檔和解開的檔案。舊的七個細胞通道、舊 index 和舊的 mask latent 在解壓時直接略過，和 Colab 先解開再移走的結果相同（`tillicum/setup.sh`、`tillicum/streamunzip.py`；每個檔案都檢查 CRC-32 和大小）。
  - 存放位置：所有新檔案（資料、checkpoint、log、快取）都放在 `/gpfs/scrubbed/pohaoc2/hfv_retrain_20261118`，不佔 projects（實驗室共用，剩約 36 GB）或 home。scrubbed 的檔案一段時間沒被存取就會被自動清掉；checkpoint 另外會上傳到 S3，所以不會遺失。
  - 下載和環境檢查都在登入節點做，GPU 作業時間內不下載、不安裝。時間上限 4 小時（每小時 $0.90，最多 $3.60；預算從 2026-10-09 起算，上限 $40，作業送出前已用 $17.95）。

### 第五個比較對象：沒有 TME 的 Vanilla ControlNet（不需要訓練）

- **論文怎麼定義**：論文的比較基準叫做「TME-free *Vanilla* ControlNet」（`_si.tex` 的 Table `si:archmetrics` 和 Fig. `si:variants`；`TERMINOLOGY.md` 規定只用這個名稱）。PixCell 程式裡是 `tools/ablation_a1_a2/build_cache.py` 的 `a2_off_shelf`：
  - 模型：公開的 PixCell-256 主幹，加上公開的 PixCell-256 Cell-ControlNet（`pretrained_models/pixcell-256-controlnet`）。兩者都是現成權重，沒有用我們的資料微調過，也沒有 TME 編碼器。
  - 條件：只有 cell_masks 通道（二值化後經 SD3.5 VAE 編碼成 latent）和該 tile 的 UNI-2h 向量。其他 9 個 TME 通道都不輸入。
  - 生成設定：30 步、guidance 1.5、seed 42（`tools/baselines/pixcell_offshelf_inference.py`）。
- **所以這次不訓練它，也沒有第五個訓練 cell。** 它沒有可訓練的部分，新資料只會改變它的輸入。評估時要用新的輸入重新生成一次：新的 cell_masks（Lin 分割 + loose nuclei 繪製）和新算的 UNI（ea5d5bd9 前處理）。它和四個變體用同一批 tile，指標也用同一套程式算。
- **ΔLPIPS 的說法和程式不一致（只在這裡記錄，論文不改）**：論文正文說 ΔLPIPS 是「相對於沒有條件的基準」的改善（`_results.tex` 第 91 行、`_methods.tex` Architecture selection）。但 PixCell 的 `sensitivity_eval.py` 實際算的是：完整條件生成的圖，和把某一組 TME 通道歸零後生成的圖，兩張之間的 LPIPS。也就是各通道群組被拿掉時圖會改變多少，不是相對於 Vanilla ControlNet 的差距。這次重跑 Fig 2 時兩種都會算，分開報告：
  1. 照程式的定義（群組歸零），和論文的數字可以直接比；
  2. 照正文的說法：每張 tile 先算 LPIPS(Vanilla 生成圖, 真實 H&E)，再減去 LPIPS(該變體生成圖, 真實 H&E)。

### 15 條件通道群組消融（Concat）

1. 開跑前先用 50 張 tile × 1 個條件實測速度，再把估計表換成實測數字。
2. 規模：15 個群組組合 × （300 張 paired + 300 張 unpaired）。**n = 300 事先固定，之後不依結果補到 1,000 張。** 依信賴區間是否跨過 0 再決定補多少，屬於 optional stopping，會讓偽陽性率膨脹。所有信賴區間都照實報告；跨過 0 只代表「無法區分」。
3. 變體用 Concat，複製論文的設計。用四方比較勝出的變體再做一次消融，屬於另一個後續問題，不在這次執行範圍內。
4. 每個條件的基準圖都在同一次執行裡重新生成，不用舊的快取。論文的 handover 記錄過：同樣設定重跑，ΔE 差了 10.9。
5. tile 分兩組，分開報告：
   - **主要結果**：CRC36 保留區和 CRC34（patient 2）的 tile。這些 tile 都沒有參加訓練。
   - **「訓練 tile，可和論文比較」**：論文原本的 tile 清單，即 `concat_ablation_1000/tile_lists/paired_1000_tile_ids.txt` 的前 300 張，和 Fig 4 相同。unpaired 的風格來源用 `unpaired_mapping.json`（seed 42）。這 300 張裡有 1 張（`10496_17664`）在新的 CRC33_01 資料中因墨水被排除、沒有新通道，所以這組實際是 299 張。unpaired 的風格只用 UNI，5 張風格來源 tile 是墨水 tile 也照用。
   - 主要結果那組的取樣方式（協調者 2026-11-18 決定，在任何評估結果出現之前）：
     - 取樣前，兩張切片都先套用和訓練 tile 相同的篩選：組織篩選、排除墨水 tile，門檻和程式都相同。CRC36 保留區另外排除緩衝帶。
     - CRC36 保留區：不超過 300 張就全部用；超過就以 seed 42 隨機取 300 張。
     - CRC34：以 seed 42 隨機取 300 張。不取「排序後的前 300 張」，因為 tile 名稱依 y_x 排序，前 300 張會集中在切片上方，取樣在空間上有偏差。
     - unpaired 的風格來源：在同一組 tile 內以 seed 42 做一個沒有不動點的排列（derangement），讓每張 tile 的風格都來自另一張 tile。
   - 次要組「與論文相同的 tile 選法」：CRC34 排序後的前 300 張，和論文跨病人分析相同。單獨報告，不和主要結果合併。
   - 兩份檔案都從 `s3://bagherilab-working/pohao/share_space/04_inference/inference_output_20260527.zip` 用 range 讀取取出，共讀了約 51 MB。

### CRC34 的 CellViT：沿用舊 JSON 前的閘門（協調者 2026-11-18 決定，在開跑之前）

- 先確認新 CRC34 的 tile 格子（原點、大小）和舊 CellViT JSON 完全一致。不一致就直接全部重跑。
- 從 CRC34 評估 tile（已套用訓練篩選）以 seed 42 隨機取 50 張，在 Colab 上以 TF32 關閉重跑 CellViT-256，和舊 JSON 比對（`colab/cvgate34.py`）。
- 每張 tile 要同時符合以下四點才算通過：
  1. 細胞數相同；
  2. 每顆細胞配對後質心距離都是 0；
  3. 類別 100% 一致；
  4. 輪廓的每個點座標都相同。這是協調者加的條件。（更正：`cellvit_dead.py` 對 Lin 細胞時，看的是 CellViT 死細胞物件的質心落在哪個 Lin 細胞上，不看輪廓重疊。所以輪廓這一條比實際需要的更嚴格，條件本身不改。）
- 抽樣方法：把 CRC34 評估 tile 的名稱排序後，用 `numpy.random.default_rng(42).choice(..., 50, replace=False)` 抽 50 張（`crc34_render_20261118/scripts/cvprep34.py`）。消融主要組的 300 張也用同一種寫法另外抽一次（`default_rng(42).choice(排序後的名稱, 300, replace=False)`），CRC36 保留區也一樣。這兩次抽樣互相獨立，閘門的 50 張不一定落在那 300 張裡面。
- 50/50 張都通過才沿用舊 JSON，並在這裡引用結果；只要有任何一張不符，就把所有 CRC34 tile 重跑一次（cv34）。
- 逐位元組相同的檔案數另外回報，不算在通過條件裡。
- 執行位置（協調者 2026-11-18）：這個 Colab 帳號一次只能開一個付費 GPU session，Colab 要依序跑三個變體的訓練，所以 CRC34 的 GPU 步驟（cvgate34、必要時 cv34、uni34hub、feat34、masklat34）改在 Hyak 的 checkpoint 分區 A100 上跑（`crc34_render_20261118/scripts/gpu34.slurm`）。指令和 Colab 的 `boot.sh` 相同；環境用 CRC16 G-ckpt 閘門那次的 Hyak CellViT 環境（它的套件版本就是 Colab 版本表加上 scikit-image / numba），以前的 G-ckpt 閘門也是在 Hyak A100 上跑的。預訓練權重用 Colab tar 當初打包時的 Hyak 原始檔案。第一次送出（43373407）在 GPU 節點上往 gscratch 寫小檔案時，每個檔案要花約 1 秒，70 分鐘只解開了 4.5 萬張 H&E 中的 3.2 萬張，所以取消了。改成在節點本機的 /tmp 工作，標記檔、cvgate34.json 和輸出 tar 留在 gscratch（43376448）。這只改了存放位置，指令和判定條件都沒變。
- 背景：以前的 G-ckpt 閘門（sim-exp-validation prereg_crc16.md 修訂 1）用的就是固定的 50 張 CRC34 tile。TF32 關閉時，50/50 張細胞數相同，1,228/1,228 顆質心距離為 0，48/50 個 JSON 逐位元組相同。

### 評估的 tile、生成和評分規則（協調者 2026-11-18 決定，在任何評估結果出現之前）

這些規則在第一張評估圖生成之前就已 commit。各項評估的程式都是 PixCell cb62fa7 的原始腳本；唯一的改動是在執行時把 checkpoint 路徑和資料根目錄換成這次的（例如 `build_cache.py` 的 `INFERENCE_VARIANTS` 原本寫死了舊 checkpoint）。

1. **Fig 2 四個變體的指標（FUD、LPIPS、PQ、DICE、HED）**
   - 比較組：四個變體加上 Vanilla ControlNet，五者算一組。五者在同一種 GPU 上生成，盡量在同一個 session 或作業內完成，步數、guidance、seed 都相同（照論文 `si_a1_a2`：30 步、guidance 1.5、seed 42）。
   - **主要結果**：CRC36 保留區 300 張和 CRC34 300 張，與消融主要組是同一批 tile（取樣方法見上面的消融段落）。每一組都重新生成 Vanilla。
   - **次要組**：論文 Fig 2 那 100 張 CRC33 tile（`metric_tile_ids`）。**可和論文比較；這些是訓練 tile，不是保留區。**
2. **ΔLPIPS**：和第 1 點用同一批 tile，兩種定義都算、分開報告（定義見上面 Vanilla 段落）。
3. **UNI×TME 分解（Fig 3）和每個子通道的影響（Fig 4，ΔE / ΔPQ）**
   - 都用 Concat、20 步、guidance 2.5、seed 42（照論文）。
   - 主要結果：CRC36 保留區那 300 張。
   - 次要組：照論文的 tile 選法，在新的 CRC33_01 上各跑一次（分解取排序後前 500 張；Fig 4 取 `paired_1000_tile_ids.txt` 前 300 張，扣掉墨水 tile 後實際 299 張）。這組同樣是訓練 tile，只用來和論文比較。
   - 每個子通道拿掉時用的基準圖，都在同一次執行裡重新生成（`--regenerate-baseline`）。
   - Fig 3D–F 的 a4 掃描（30 張 tile × 掃描值）**這次未執行**。這不是取消，之後需要時再做。
4. **CRC34 凍結推論**
   - 照論文 `run_evaluation.py`：n = 1000，`random.seed(42)` 加 `random.sample`；paired 用該 tile 自己的 UNI，unpaired 用「下一張」tile 的 UNI（`tile_ids[(i+1) % N]`）；Concat、20 步、guidance 2.5、seed 42。
   - **注意**：CRC34 的輸入已經重建（見偏離紀錄），送進抽樣的 tile 清單和論文的不同。抽樣程序相同，但抽到的 tile 不一定相同，和論文比較時要附上這個前提。
5. **生成和評分的硬體**
   - 每個比較組（一個條件和它的基準圖、Fig 2 的五個模型）從頭到尾在同一種 GPU 上生成。如果有組被拆到不同種 GPU，要在這裡記錄。生成可以在 Colab L4 或 Hyak checkpoint 分區的 A100 上做，Tillicum 只用來訓練。
   - 所有評分（CellViT、FUD 的 UNI 特徵、LPIPS 等）都集中在 Hyak 的 A100 上做，TF32 關閉（`NVIDIA_TF32_OVERRIDE=0`）。
   - 速度：kit 完成後，先在 Colab L4 和 Hyak A100 上各跑一批，分別實測 20 步和 30 步每分鐘生成幾張，再依實測數字分配 session。只寫實測數字，不寫推估值。
6. **可解碼性**：只用真實 H&E，不需要新的 checkpoint。照論文的程序做，目標換成新的細胞通道：CRC33_01 的 tile，空間分塊 5-fold（2048 px 一塊）。程式預設是 3 折，所以要加 `--n-splits 5`。

### 評估 kit（`eval/`）

- **評估包**（Hyak CPU 作業打包後放到 S3 `inputs/eval/`）：每張切片一個 tar，格式就是 PixCell 讀的 ORION 目錄（10 個通道、UNI、H&E 的 VAE latent、mask latent、H&E、index）。來源檔案就是組訓練資料用的那些檔案（`eval/evalpack.py`）。缺任何一項就停止。
  - `evalpack_crc36ho`：CRC36 保留區全部 1,092 張（緩衝帶和訓練 tile 不放）。主要組的 300 張和 derangement 由 `eval/picks.py` 產生（`crc36ho_p300_ids.txt`、`crc36ho_p300_unpaired.json`）。
  - `evalpack_crc33sec`：CRC33_01 次要組，共 824 張；另有 209 張只當 unpaired 風格來源（只需要 UNI 和 H&E）。`eval/sets33.py` 從論文的 tile 清單切出：Fig 2 100 張（全部都有新通道）、消融和 Fig 4 共 299 張（`10496_17664` 是墨水 tile，扣掉）、分解 500 張（PixCell 的數字排序，取前 500 張）。論文的三份清單從 `inference_output_20260527.zip` 用 range 讀取取出（另讀約 52 MB），放在 `inputs/eval/paper_lists/`。
  - `evalpack_crc34eval`：要等 CRC34 的 CellViT 閘門和重繪完成才能打包。
- **checkpoint**：生成用的是 epoch 20 的推論副本 `outputs/infer_<run>_e20.tar`（`eval/slimck.py`）。PixCell 推論只讀 `state_dict`（`_extract_state_dict`）和 `tme_module.pth`，所以副本只保留這兩部分。副本重新載入後逐個 tensor 和原檔比對，必須完全相同。完整的訓練 checkpoint 仍留在 S3。
- **生成**：`eval/evalgen.py`，Colab（`eval/boot_eval.sh` + `eval/presign_eval.py`，只用 presigned 連結）和 Hyak A100（`eval/gen_hyak.slurm`）跑的是同一支程式。
  - PixCell 程式不改。有兩支腳本把 checkpoint 和資料路徑寫死在程式裡（`build_cache.py` 的 `INFERENCE_VARIANTS`、`sensitivity_eval.py` 的 `data/orion-crc33`），改用一個由 symlink 組成的影子目錄提供這些路徑。
  - 每個比較組有自己的輸出資料夾，裡面記錄生成時用的 GPU 型號；如果換到別種 GPU 上接續，程式就停止。
  - 速度實測用 `speed` 任務：生成的圖只用來算速度，不進入任何結果。
    - 30 步：前 16 張 tile，每張 1 張圖（`build_cache.py`）。20 步：前 8 張 tile × 15 個條件（消融腳本）。消融腳本每張 tile 另外寫一個 `cell_mask.png`，它不是生成的圖，不算進張數。
    - 每支指令記四個數：總耗時（含模型載入）、到第一張 tile 完成的秒數、穩定速度、載入開銷。
    - 穩定速度只用 tile 完成的時間點算：(tile 數 − 1) × 每張 tile 的圖數 ÷ (最後一張 tile 完成時間 − 第一張完成時間)。這樣模型載入和第一張 tile 都不算在內。
    - 載入開銷 = 到第一張 tile 完成的秒數 − 一個穩定的 tile 間隔。
    - 2026-11-19 實測（新計法，Concat，CRC36 保留區的前 16 / 8 張 tile）：

      | GPU | 30 步穩定速度（張/分鐘） | 30 步到第一張 tile 的秒數 | 20 步穩定速度（15 條件消融） | 20 步到第一張 tile 的秒數 |
      |---|---|---|---|---|
      | Colab L4（`l4speed2`） | 25.16 | 32.6 | 84.43 | 36.4 |
      | Hyak ckpt A100 80GB PCIe（43390518，g3085） | 33.04 | 489.0 | 198.5 | 60.7 |

      Hyak 上每個 job 的第一支指令要先等約 8 分鐘才出第一張圖，第二支指令只要約 1 分鐘。新 Colab session 的準備時間另計（setup 加輸入下載：一次 13 分鐘，一次 2,102 s）。
    - 2026-11-19 第一版的速度數字（L4：30 步 25.38 張/分鐘，只有 8 張；20 步 113.24 張/分鐘，64 個檔案裡含 4 個 `cell_mask.png`）樣本太少，而且計法不同，不用來規劃。
  - 上傳失敗也算這個任務失敗。Colab 的 task 行會寫出 `upload ok/FAILED`，最後一行寫失敗的任務數。Hyak 上傳失敗時不寫 `sent_` 記號，重排時會重做。
  - multipart 的上傳槽只能用一次：第一次完成上傳後，upload id 就失效，再傳同一個槽會得到 404。所以每個 `gen_` 槽另外附一個可以重複使用的單一 PUT 連結，5 GB 以下的檔案走這條。每次重跑都產生新的 boot 和新的 TAG。
  - Colab 下載（2026-11-19 起）：c36fig2 session 的準備花了 2,102 s、輸入下載又花了 2,578 s，原因是每個檔案依序用單一連線下載。改成：
    - boot 一開始就同時下載所有檔案，pip 安裝和下載同時進行。
    - 大於 200 MB 的檔案分成多段（每個檔案 8 段），用 Range 同時下載，每段可以接續，下載完依序接起來。
    - 每個檔案下載完都要和 S3 物件比對大小和 ETag。ETag 是 presign 時用 head_object 讀的；分段上傳的物件，用可能的分段大小重算 ETag 來比對。
    - 仍然只用 presigned 連結，不放任何帳號憑證或 HF token。
- **Vanilla ControlNet 的 seed（偏離）**：`pixcell_offshelf_inference.py` 的起始雜訊用 `torch.randn_like` 產生，沒有設 seed，所以論文的 Vanilla 圖會受前面生成過的 tile 影響（上面寫的「seed 42」只對四個變體成立）。這次用 `eval/pxrun.py` 在每張 Vanilla tile 生成前設 `torch.manual_seed(42)`，讓結果可以重現；腳本的其他部分不動。
- **論文 ΔLPIPS 的 tile 數不一致（只記錄）**：論文 `si_a1_a2/cache.json` 裡，Grouped 和 Concat 的敏感度各用了 500 張 tile，Per-channel 和 Grouped+bypass 只用 100 張。這次四個變體都用同一批 tile。

### 論文 figure 用新模型重繪：逐 panel 的輸入（協調者 2026-11-19 交辦；這張表在重繪之前就 commit）

用新模型重新產生一份論文 figure，和原圖並排比較。原本的 figure 一張都不覆蓋；要不要換進論文由使用者決定。

**做法**
- 在 Hyak 上另開一份 PixCell cb62fa7 checkout（`/gscratch/cheme/chiu/hfv_figs_20261118/PixCell`），腳本照原樣執行，不改 PixCell。
- 腳本寫死的 `inference_output/`、`data/`、`results/` 用和 evalgen 一樣的 symlink 影子目錄，指到這次的評估輸出。
- 輸出（PNG/PDF）只寫在這份 checkout 底下，完成後放到 S3 `hfv_retrain_20261118/figures_rerender/`。
- 因為論文 figure 都畫 CRC33，重繪一律用評估規則第 1、3 點裡的 **CRC33 次要組**，也就是可和論文比較的訓練 tile：
  - 消融和 Fig 4：299 張。論文是 1,000 張（消融）或 300 張（Fig 4）。
  - Fig 2 的四個變體：100 張。論文 Grouped/Concat 的 ΔLPIPS 用 500 張。
  - 分解：500 張。
- **重繪的論文 figure 用的是 CRC33 次要組（訓練 tile），目的是和論文比較；保留區的結論來自 CRC36 主要組，寫在結果報告裡。**

**分類**
- **新模型重做**：輸入由這次的新 checkpoint 生成或評分。
- **不變**：輸入和生成器無關，例如示意圖、真實 H&E 上的細胞測量、真實 UNI 特徵的探針。照用原輸入。
- **這次未重做**：輸入依賴生成器或舊通道，但這次沒有重跑。圖中照用原輸入，並排頁標註「舊模型輸入」。

**逐 panel 輸入**

| Figure | Panel | 輸入 | 分類 |
|---|---|---|---|
| Fig 1 | a 示意圖 | `stage_1_svg.svg` | 不變 |
| | b、c 各細胞類型的面積、圓度、marker 熱圖 | hfv `processed_crc33/cell_assignments.csv`、`cell_shape_features.csv`（真實 H&E 和 CODEX） | 不變 |
| | d 真實資料拼圖 | `stage_1_results.png` | 不變 |
| | | Fig 1 全部不變，不重繪，並排頁直接放原圖。 | |
| Fig 2 | a 示意圖 | `stage_2_svg.svg` | 不變 |
| | b 訓練 loss 曲線 | `si_a1_a2/cache.json["training_curves"]`，來源是各 run 的 `train_log.jsonl`（`log_utils.py` 寫死路徑，用影子目錄指到這次 Tillicum 四個 run 的 log） | 新模型重做 |
| | c ΔLPIPS 敏感度 | `cache.json["sensitivity"]`（評估規則第 2 點，次要組 100 張） | 新模型重做 |
| | d 15 條件的指標取捨（paired + unpaired） | `concat_ablation_1000/{paired,unpaired}_ablation/ablation_results/*/metrics.json`、`fud_scores.json`（評分都在 Hyak A100）；Vanilla 參考帶是 `tools/ablation_report/shared.py` 的寫死常數 `TRADEOFF_REFERENCE_BANDS` | 消融 = 新模型重做；三條 Vanilla 參考帶（FUD、PQ、DICE）換成這次次要組 100 張的 Vanilla 數值（協調者 2026-11-19 核准；由 `eval/` 的 wrapper 在執行前替換這個常數，PixCell 不改；新舊數值並列記在偏離紀錄）。兩條「ORION H&E」參考帶（LPIPS、HED）是真實 H&E 的參考值，不變 |
| | e、g paired / unpaired 效果量熱圖 | 同 d | 新模型重做 |
| | f、h paired / unpaired 示例條 | 消融生成圖、`*_cellvit_instances.json`；tile 由腳本從現有 tile 中抽（seed 1），可能和原圖不同；參考 H&E 和 cell_masks 是真實資料 | 新模型重做 |
| Fig 3 | a 分解 2×2 示例 | `results/a2_decomposition/generated/<tile>/*.png`、`representative_tile.json` | 新模型重做 |
| | b、c 各模式指標和 UNI/TME/交互熱圖 | `decomposition_summary.csv` | 新模型重做 |
| | d 探針 ΔR² | `results/a4_uni_probe/probe_results.csv`（真實 UNI 特徵上的 ridge） | 不變 |
| | e 編輯專一性熱圖 | `specificity_full.csv`，來自 a4 掃描的生成圖 | 這次未重做（舊模型輸入） |
| | f α 掃描示例 | `results/a4_uni_probe/sweep/*/targeted/alpha_*.png` | 這次未重做（舊模型輸入） |
| Fig 4 | a 示意圖 | `stage_3_simple_svg.svg` | 不變 |
| | b ΔE / ΔPQ 對 R² 的象限圖 | ΔE、ΔPQ 來自 `subchannel_loo_n300/per_subchannel_{,layout_}summary.csv`；R² 來自 `a1_probe_mlp_spatial/uni_16`（評估規則第 6 點，新通道） | ΔE/ΔPQ = 新模型重做；R² = 新通道重做 |
| | c、d、e、f 跨世代比較（C33 / C34 / C10） | `results/a6_cohort_invariance/cohort_invariance_metrics.json`，由 a6 從三個世代的 LOO CSV 和 R² 算出 | LOO（ΔE、ΔPQ）：C33、C34 = 新模型重做（C34 見下面「CRC34 子通道 LOO」），C10 這次未重做（沒有重建的資料，舊模型輸入）。R²：C33 = 新通道重做，C34、C10 這次未重做（舊通道）。a6 用這些輸入重算 |
| si_channel_decodability_impact | a 四個編碼器的 R² | `a1_probe_mlp_spatial/{uni_16,virchow2_16,ctranspath_07,resnet50_07}` | UNI = 新通道重做；其他三個編碼器這次未重做（舊通道） |
| | b raw MX 的 R² | `a1_probe_mlp_spatial/t2_spatial` | 這次未重做（舊通道） |
| | c、d ΔE / ΔPQ 排名 | 同 Fig 4b 的 LOO CSV | 新模型重做 |
| si_performance_ranking | 單 panel | 同 Fig 2d 的消融指標 | 新模型重做 |
| ablation_grids_combined | a paired 網格，tile `10752_13824` | 消融生成圖和 metrics（這張 tile 在 299 張之內） | 新模型重做 |
| | b unpaired 網格，tile `13056_27392`（風格 `12032_15104`） | 這張是論文 unpaired 清單的第 1,001 張，不在 299 張裡 | 協調者 2026-11-19 核准：把這張 tile 補進消融生成（1 張 × 15 條件，unpaired，風格 `12032_15104`），和次要組消融同一比較組、同一種 GPU。如果它在新的 CRC33_01 裡沒有新通道，就這次未重做（舊模型輸入）。**結果（packgrid33 43393902，2026-11-19）：這張 tile 在新的 CRC33_01 裡沒有新的細胞通道（NO_CHANNELS），所以 panel b 這次未重做，保留舊模型輸入。** |
| si_qualitative_variants | 單 panel：10 張 tile × 五個模型 | `si_a1_a2/tiles/<variant>/<tile>.png`（`metric_tile_ids[:10]`） | 新模型重做 |
| si_threshold_sensitivity | a、b | 同 Fig 4c 的 a6 JSON | 同 Fig 4c–f：ΔPQ 的 C33、C34 = 新模型重做，C10 這次未重做；R² 的 C33 = 新通道重做，C34、C10 這次未重做 |
| si_patient_generalization | a CRC33 對 CRC34 指標 | `share-space-paper/figures/data/patient_generalization_metrics.json`。PixCell 裡沒有產生這個檔的程式，原本的 JSON 由哪支程式產生**不明** | 新模型重做。這次的 JSON 是**重建的**：由 `eval/` 的 wrapper 用同樣的鍵，從 CRC34 評估（規則第 4 點）和 CRC33 消融的全通道條件算出 |
| | b 每位病人 2 張示例 | CRC33 消融生成圖，CRC34 的 `CRC34_paired/<tid>/{paired,unpaired}/generated_he.png` | 新模型重做 |

**寫入位置的保護**
- 有幾支腳本會寫入或讀取 `ROOT.parent/share-space-paper/figures`：`publish.save_mirrored`，Fig 4 直接寫 png/pdf，patient 腳本會刪掉 `assets/gen/*.png` 再重新複製。
- 所以 checkout 的上一層（`hfv_figs_20261118/`）只放一個新建的空資料夾 `share-space-paper/`，當作替身，只裝 patient JSON。
- 建立前先確認那裡原本沒有這個資料夾（協調者 2026-11-19 核准替身）。每次執行前由腳本檢查兩件事，任一不通過就停止：
  - 替身是一般資料夾，不是 symlink；
  - 從 Hyak checkout 沿 symlink 解析不到真正的 `share-space-paper`。真正的論文 repo 只在 Mac 上，不碰。
- patient 腳本刪除 `assets/gen/*.png` 只會發生在替身裡。
- `figures/pngs_updated/` 和 `results/paper_figures/` 都在這份新 checkout 裡。

**CRC34 子通道 LOO（協調者 2026-11-19 決定，在執行之前寫入）**
- 目的：讓 Fig 4c–f 和 si_threshold_sensitivity 的 C34 也用新模型。
- 做法：CRC34 評估包完成後，在 CRC34 的 300 張上跑 `run_subchannel_loo --regenerate-baseline`。這 300 張是 `eval/picks.py` 抽的，和評估規則第 1 點同一批。
- 設定和 CRC33 相同：Concat、20 步、guidance 2.5、seed 42；基準圖在同一次執行、同一種 GPU 上生成。
- 優先順序最低：排在主要生成、評分和 CRC33 次要組之後。
- C10 沒有重建的資料，仍是舊模型輸入。

## 偏離紀錄

- **CRC34 的輸入改用新流程重建（偏離論文的 CRC34 輸入）**：CRC34 的評估輸入照 CRC36 的流程重新建立，規則、門檻和程式路徑都相同（`crc34_render_20261118/`）。流程包含 Lin 分割、CELESTA 加 Lin Fig. 3d 判定細胞類型、NucleusShapes 繪製、CellViT 判定死細胞，以及血管、氧、葡萄糖通道；這三個通道的參數直接讀 CRC36 那次執行的設定。論文用的舊 `orion-CRC34.tar` 不再使用，但仍原封不動保留在 S3（`share_space/02_datasets/`）。這是協調者在 2026-11-18 決定的，當時還沒有任何評估結果。
- **重繪 Fig 2d 時替換 Vanilla 參考帶（偏離 PixCell 寫死的常數，協調者 2026-11-19 核准）**：PixCell `tools/ablation_report/shared.py` 的 `TRADEOFF_REFERENCE_BANDS` 寫死了論文 Vanilla 的數值，重新生成也不會更新；新數字旁邊留著舊參考帶會誤導。重繪時由 `eval/` 的 wrapper 在執行前換成這次 CRC33 次要組 100 張的 Vanilla 數值，PixCell 不改。兩條「ORION H&E」參考帶是真實 H&E 的參考值，照用。

  | 參考帶 | 論文（寫死在 `shared.py`） | 這次（次要組 100 張） |
  |---|---|---|
  | Vanilla ControlNet FUD | 198.376（平均） | 評分完成後填入 |
  | Vanilla ControlNet PQ | 0.7035 ± 0.1185 | 評分完成後填入 |
  | Vanilla ControlNet DICE | 0.8588 ± 0.1061 | 評分完成後填入 |
  | ORION H&E LPIPS | 0.5463 ± 0.0792 | 不變 |
  | ORION H&E HED | 0.2930 ± 0.2129 | 不變 |

  - 寫死的 Vanilla 數值和論文 `si_a1_a2/cache.json` 裡的 Vanilla（`a2_off_shelf`，100 張）並不相同：cache 是 FUD 240.93、PQ 0.6987 ± 0.1471、DICE 0.8529 ± 0.1408。寫死的值是從哪一批 tile 算出來的，查不到。這次照協調者的決定，用和 cache 同樣定義的次要組 100 張。
  - 替換由 `eval/figrender.py`（`bands`、`run --bands`）做：`bands` 從這次次要組的 `cache.json` 取出三條 Vanilla 參考帶，`run` 在載入 figure 腳本前就地改寫這個 dict。

- **血管、氧、葡萄糖通道：重現原本的 CRC33_01 檔案時有 6 張 tile 的 vasculature 不完全相同（協調者 2026-11-18 決定接受）**：CRC36 和 CRC34 的這三個通道是用 `crc36_render_20261118/mx/` 的流程產生的，參數是 hfv 的程式預設值：ds 4、O2 Krogh 長度 200 µm、glucose 120 µm、不做 SMA 修正。這組參數是從訓練資料本身推出來的；論文 `_methods.tex` 和 PixCell 文件寫的是 160/450 µm、ds 8，但這組數值重現不了訓練資料。用這組參數重算 CRC33_01 全部 10,379 張 tile：oxygen 和 glucose 都是 10,379/10,379 張逐位元組相同；vasculature 是 10,373/10,379 張相同。不同的 6 張每張差 64–1,502 個像素（IoU 0.99998），差異都落在 Otsu 門檻上，也就是門檻差了 1 個灰階。可能是函式庫版本或浮點捨入造成，原本執行時的版本沒有紀錄，無法確認。CRC33_01 的訓練仍然用原本的通道檔案。

- **評分的執行方式（2026-11-20，在任何分數出現之前寫入）**：評分照規則第 5 點在 Hyak checkpoint 分區的 A100 上做，TF32 關閉。程式是 `eval/evalscore.py`（加上 `eval/score_hyak.slurm`），每一步都直接執行 PixCell cb62fa7 自己的評分腳本，順序照 PixCell 文件寫的流程。下面幾點是執行上需要另外處理的地方：
  - **UNI-2h 前處理（套用上面「UNI-2h 前處理」規則，不是新的規則）**：PixCell 的 `pipeline/extract_features.py` `UNI2hExtractor` 用 timm 內建的設定建立前處理，重現不了已存的真實 tile 特徵（最小 cosine 0.588）。如果照用，FUD 和 UNI cosine 會拿兩種前處理做出的特徵互相比較。`eval/pxscore.py` 在執行時只替換這個類別的 `transform`，改成 `uni-2h/config.json` 的設定（和 `colab/feat_uni.py` 的 hub 寫法相同），PixCell 檔案不改。替換是用 import hook 做的，所以 PixCell 用多個 process 平行計算時也一樣有效。
  - **CellViT**：用 he-feature-visualizer `stages/run_cellvit_local.py`（CellViT-256），和 CRC34 閘門用的是同一支程式、同一個 Hyak 環境。
  - **套件**：那個 Hyak 環境沒有 `lpips` 和 `pandas`，另外用 `pip install --no-deps --target` 裝在新資料夾 `/gscratch/cheme/chiu/hfv_score_20261119/pylib`：lpips 0.1.4、pandas 2.2.3，以及 pandas 需要的 python-dateutil、pytz、tzdata、six。LPIPS 用的 AlexNet 權重先下載到 `torch_home`，sha256 開頭 7be5be79，和 torchvision 檔名裡的雜湊一致。
  - **每張 tile 的 Fig 2 指標**：PixCell 的 `metrics_io.py compute` 只在 `cache.json` 存平均和標準差。正文定義的 ΔLPIPS 和它的信賴區間需要每張 tile 的數值，所以 `eval/fig2tiles.py` 用同樣的 PixCell 函式、同一批 tile 再算一次並保留每張 tile 的值，和 `cache.json` 的平均之間的差也一起記錄。
  - **信賴區間**：`eval/evalstats.py` 在 CPU 上讀評分結果，不重新評分。區間是 95 % 的 percentile bootstrap，以 tile 為單位重抽（10,000 次，seed 42）。同一批 tile 上兩個模型或兩個條件的差，用同一組重抽。消融的效果量照 PixCell `summarize_added_group_effect_stats` 的定義（Fig 2e/g 熱圖）。FUD 是整組圖片一個數字（兩個特徵分佈之間的距離），不是每張 tile 一個值，所以沒有逐 tile 的區間。
  - 作業一律不自動重排（`--no-requeue`）。被搶占或失敗都視為停止，記錄後回報。
  - **PQ 檢查（CPU 試跑 43416299，2026-11-20，在任何正式分數出現之前；結論：不是評分錯誤，正式作業 43415409–11 不動）**：
    - 起因：前一次 CPU 試跑（43415428，每組 3 張 tile）裡，LOO 的基準圖和分解的 `uni_plus_tme` 平均 PQ 到小數第 6 位都相同（0.109403），消融「四組全開」的 PQ 卻明顯不同。原本猜是三組挑到不同的 tile。
    - 查證結果：三組挑到的是**同一批 3 張 tile**（`0_10752`、`0_11008`、`0_6400`），所以「挑到不同 tile」的猜測不成立。
    - LOO 基準圖和分解 `uni_plus_tme` 是**逐位元組相同的圖**（3/3 張 md5 相同），所以兩者 PQ 相同是必然的：兩支腳本都走 PixCell 的 `generate_from_ctrl`，模型、條件、seed 都一樣。
    - 消融「四組全開」的圖和它們**不同**（3/3 張 md5 不同）。消融腳本走另一條路徑 `generate_group_combination_ablation_images`：15 個條件共用同一個起始雜訊，分批一起去噪（`denoise_batched`）；`generate_from_ctrl` 則在去噪前重設 seed，一次只畫一張。兩條路徑都是 PixCell 原本的寫法，論文的消融和 LOO 也是這樣分開生成的。
    - 評分本身一致：用 PixCell 的 `run_cellvit`、`_compute_pq` 和同一張輸入細胞遮罩，逐張重算三組圖的 PQ，結果和評分輸出完全相同（消融 0.4287 / 0 / 0.1706，對應 `metrics.json`；LOO 和分解都是 0.3282 / 0 / 0，平均就是 0.109403）。
    - 所以：消融「四組全開」的 PQ 和 LOO 基準、分解 `uni_plus_tme` 的 PQ 本來就不必相同，三者不能互相代替；報告時各自引用自己那一組。這 3 張 tile 都在切片邊緣，每張只有 2–17 個真實細胞，只用來檢查程式，不代表任何結果。
  - **兩條生成路徑的逐項比較（協調者 2026-11-20 要求；結果出現前的說明，只讀程式和論文既有數字，沒有新的 GPU 執行，生成和評分程式都不改）**
    - 先更正數字：3 張試跑 tile 上，消融「四組全開」的逐張 PQ 是 0.4287 / 0 / 0.1706，平均 **0.200**；LOO 基準（＝分解 `uni_plus_tme`）是 0.3282 / 0 / 0，平均 **0.109**。之前說的「0.43」只是第一張 tile 的值，不是平均，所以並沒有 4 倍的差距。
    - 「消融路徑」是 `generate_ablation_subset_cache.py` → `generate_ablation_images_with_context` → `denoise_batched`。「單張路徑」是 `run_subchannel_loo.py` 和 `src.a2_decomposition.main` → `generate_from_ctrl` → `denoise`。下表逐項比較兩者（PixCell cb62fa7）：

      | 項目 | 消融路徑 | 單張路徑 | 相同？ |
      |---|---|---|---|
      | 模型和 checkpoint | `load_all_models(config, config_path, ckpt_dir)`；evalgen 給的 config 和 `step_dir("tr_concat")` 相同 | `load_sweep_models` 內部呼叫同一個 `load_all_models`，config 和 checkpoint 也相同 | 相同 |
      | 排程器（scheduler） | `make_inference_scheduler`：DDPM，1000 個訓練步、linear beta、epsilon | 同一個函式 | 相同 |
      | 步數 | `--num-steps 20`；`denoise_batched` 內部另外用預設的 `num_inference_steps=20` 重設 | `--num-steps 20`；`denoise` 內部同樣用預設 20 重設 | 相同（20 步） |
      | guidance | 2.5（腳本預設） | 2.5（`DEFAULT_GUIDANCE_SCALE`） | 相同 |
      | CFG 寫法 | 無條件分支用全 0 的 UNI 加全 0 的 ControlNet 殘差 | 同樣寫法 | 相同 |
      | 起始雜訊 | `_make_fixed_noise`：先 `manual_seed(42)` 再 `randn`，乘上 `init_noise_sigma`；15 個條件共用 | 同一個函式，同樣的 seed | 相同 |
      | 去噪過程中每一步的隨機雜訊 | DDPM 每一步都從 torch 全域亂數抽新雜訊。消融路徑在去噪前**不重設 seed**；15 個條件分成 8 張和 7 張兩批一起去噪，「四組全開」是第 15 個條件，在第二批 | 去噪前**重設** `manual_seed(42)`，一次只畫一張 | **不同** |
      | 條件組裝 | Concat（`RawConditioningPassthrough`）：群組名展開成通道，加上永遠開著的 `cell_masks`。四組全開＝9 個子通道加 `cell_masks`；通道不歸零，也沒有 dropout | 直接給 9 個子通道名稱，加上永遠開著的 `cell_masks`；走同一個 `_fuse_conditioning_latent` | 相同 |
      | UNI 特徵 | paired：`resolve_uni_embedding` 讀該 tile 自己的 `features/<tile>_uni.npy` | 同一個函式、同一個檔案（資料夾是指向同一批檔案的 view） | 相同 |
      | 解碼和輸出解析度 | `_decode_latents_to_images`，`config.image_size` | `_decode_latents_to_image`，同樣的 VAE、scale 和 shift | 相同 |
      | 數值 | fp16，batch 8 或 7 | fp16，batch 1 | batch 大小不同，可能有微小差異 |

    - 結論：兩條路徑畫的是**同一個條件分佈**，用同一個模型、同樣的條件、同一個起始雜訊。差別只在 DDPM 去噪過程中每一步加入的隨機雜訊：兩條路徑各抽了一組不同的雜訊。所以同一張 tile 會得到兩張不同的樣本圖，逐張 PQ 也會不同。但這只是取樣上的變異，不會造成系統性偏差：平均來說，兩條路徑的 PQ 應該相同。
    - **論文自己的數字支持這個結論**（同一批 tile，舊模型）：
      - 論文消融「四組全開」（paired）在 `paired_1000_tile_ids.txt` 前 300 張上的平均 PQ 是 **0.8088**（SD 0.1446，n = 300）。這是從 `inference_output_20260527.zip` 用 range 讀取這 300 個 `metrics.json` 算出來的。
      - 論文 LOO 基準在同樣 300 張上是 **0.8055**（SEM 0.0090，`subchannel_loo_n300/per_subchannel_layout_summary.csv`）。
      - 兩者差 0.003，不到 LOO 平均值 SEM 的一半。所以在論文裡，兩條路徑對「全通道 PQ」給出的是同一個數。
      - 論文分解 `uni_plus_tme` 是 0.7769（SD 0.1808，n = 500，`fig3b_condition_means.csv`）。它用的是另一批 tile（數字排序後的前 500 張，包含切片邊緣幾乎沒有細胞的 tile），不能直接和前兩者比較。
      - 論文消融 unpaired「四組全開」在同樣 300 張上是 0.7260（SD 0.2115），是另一個條件（風格來自別張 tile）。
    - 對結果的影響：Fig 2（消融）和 Fig 3/4（分解、LOO）的全通道 PQ，各自來自自己那條路徑的樣本。比較時，兩者的差應該落在取樣變異的範圍內；正式結果出來後，用 299 張上的數字確認。每個比較組（例如 LOO 的基準和拿掉一個子通道的圖）都在同一條路徑、同一次執行裡生成，所以組內的差（ΔPQ、效果量）不受這個影響。
    - 不需要修改任何程式。

## 結果

（訓練和評估完成後填入）
