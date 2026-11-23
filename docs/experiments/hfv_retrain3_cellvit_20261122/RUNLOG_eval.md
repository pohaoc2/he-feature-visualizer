# 重訓 3 執行紀錄：訓練、生成、評分（2026-11-22，進行中）

這份是流水帳，記下每一步實際在哪裡跑、花多久、和預先登記不同的地方。最後的數字與結論寫在 `RESULTS.md`。
模型名稱照 `docs/TERMINOLOGY.md`：Concat、Grouped、Grouped + bypass、Per-channel、PixCell 原版。
評估集：CRC36 保留區（主要，300 張 tile）、CRC33 次要組（Fig 2 用 100 張；消融用 299 張）。

## 1. 訓練（Tillicum）

| 模型 | 作業 | 開始 | 實測時間 | 狀態 |
|---|---|---|---|---|
| Concat | 512440 | 15:31 | 1:46:55 | 完成 |
| Grouped | 512599 | 17:20 | 2:09:50 | 完成 |
| Grouped + bypass | 512600 | 17:20 | 2:10:49 | 完成 |
| Per-channel | 512601 | 17:20 | 2:26:33 | 完成 |

- 花費：照 `hyakusage` 實查，Tillicum 累計 $34.91（上限 $40）；重訓 3 四個模型合計約 $7.71，和預估 $7.68 幾乎一樣。
- 每個 checkpoint 都用 `slimck3.slurm` 瘦身，每個約 4–6 分鐘，比對結果都是 `"identical": true`。

## 2. 生成與評分在哪裡跑

| 工作 | 原本排的位置 | 實際位置 | 實測 |
|---|---|---|---|
| CRC36 保留區 消融生成 | Hyak ckpt A100（43469094） | Colab hfv_eval_L4_b | 共 7021 s，每種模式 3338 s，9600 張 |
| CRC33 次要組 消融生成 | Hyak ckpt A100（43469095） | Colab hfv_eval_L4_b（沿用同一個 runtime） | 共 6799 s（生成 6661 s），9568 張 |
| CRC36 保留區 Fig 2 生成 | Hyak ckpt A40（43470820） | Colab hfv_eval_L4_b（沿用同一個 runtime） | 進行中；第一步（5 個模型各產一張）4113 s |
| CRC33 次要組 Fig 2 生成 | Hyak ckpt A40（43470821） | Colab hfv_eval_L4_a | 共 5678 s：第一步 1445 s，第二步（4 個重訓模型逐一移除通道組重繪）3981 s，2200 張 |
| CRC36 保留區 消融評分 | Hyak cheme CPU | 同左（43471220，40 核） | 計算 12465 s |
| CRC33 次要組 消融評分 | Hyak cheme CPU | 同左（43472897，40 核） | 進行中 |
| CRC33 次要組 Fig 2 評分 | Hyak cheme CPU | 同左（43474560，40 核） | 排隊中 |

Colab 的啟動檔都由 `eval/presign_eval3.py` 產生（一天有效的 presigned 連結，只存在本機暫存區，不進 repo）。開機程式內容和重訓 1 的 `boot_eval.sh` 相同，沒有改動。

## 3. 偏離紀錄（接續 `STAGE_inputs.md` 的 D1–D3）

- **D4、D5**：原本的代理在交接時列出，但內容沒有寫進 repo。repo 裡唯一能查到的是 commit 3f4e715：CRC36 保留區 消融生成從 Hyak A100（43469094，17:26 送出、17:45 取消、從未開始）移到 Colab hfv_eval_L4_b。
- **D6 CRC33 次要組 消融生成改到 Colab**
  - Hyak A100 作業 43469095 排隊 2 小時 20 分（17:26 → 19:45 取消），從未開始。
  - CRC36 消融生成結束後，沿用 hfv_eval_L4_b 的同一個 runtime 接著跑；設定 75 s，總共 6799 s，產出 9568 張，上傳後核對過張數。
- **D7 Fig 2 生成改送 Hyak ckpt A40，不是 A100**
  - `gen3.slurm` 寫的是 A100，但 A100 一直排不到，送出時改成 A40（43470820、43470821）。結果這兩個 A40 作業也從未開始（見 D9、D10）。
- **D8 評分用 40 核，不是 32 核**
  - 第一次送 64 核被 Slurm 拒絕（單一節點核數不足），這是送出被拒，不是作業失敗；改用 40 核送出。所有評分作業都用 40 核。
- **D9 CRC36 保留區 Fig 2 生成改到 Colab hfv_eval_L4_b**
  - Hyak A40 作業 43470820 排隊 1 小時 50 分（19:54 送出、21:44 取消），從未開始。
  - CRC33 消融生成一結束，就沿用 hfv_eval_L4_b 的同一個 runtime；21:40 啟動，159 s 進到輸入階段；確認 Colab 已開始產圖後才取消 Hyak 作業。
  - 第二步（逐一移除通道組重繪）照 CRC33 的速度推算約 3.3 小時。要加快只能拆給第二個 runtime，但那要改開機程式，所以沒拆，讓它跑完。
- **D10 CRC33 次要組 Fig 2 生成改到 Colab hfv_eval_L4_a**
  - cpm_tracks 的代理關掉 hfv_eval_L4_a 並交還後才用它。Hyak A40 作業 43470821 排隊 2 小時 09 分（19:54 送出、22:03 取消），從未開始。
  - 第一次連線時頁面停在「Allocating runtime」約 10 分鐘（頁面沒有更新），重新載入頁面後再連線才成功。21:59 啟動，228 s 進到輸入階段，GPU 是 NVIDIA L4。
  - 總共 5678 s，產出 2200 張並上傳；23:37 用「Disconnect and delete runtime」刪除 runtime。
- 同時使用的 L4 runtime 從未超過 2 個。

## 4. 還沒做完

- CRC36 保留區 Fig 2 生成（hfv_eval_L4_b）跑完後：核對張數、送評分、刪除 runtime。
- 兩組 Fig 2 評分 → `eval/r3stats.py`（排名照預先登記 §5.4）；兩組消融評分 → `evalstats.py`；`scripts/fig_r3.py` 畫圖；寫 `RESULTS.md`。
