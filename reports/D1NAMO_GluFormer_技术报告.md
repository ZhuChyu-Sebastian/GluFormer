# GluFormer 在 D1NAMO 多模态（CGM + PPG + IMU + 体温 + 进食）上的训练、预测与可视化技术报告

## 1. 目标与范围
本报告给出一条可直接落地的工程路径：使用 `GluFormer` 的自回归 Transformer 思想，将 D1NAMO 的多源可穿戴数据融合后用于未来血糖预测，并输出可视化结果。

实现文件：
- `demo/d1namo_gluformer_pipeline.py`

该脚本覆盖：
1) 数据预处理（字段校验、标准化、CGM 离散化、滑窗构建）
2) 训练（多模态 Transformer，自回归 next-token）
3) 预测（给定历史窗口预测未来 `pred_len` 点）
4) 可视化（预测 vs 真实曲线）

---

## 2. 方法设计

### 2.1 输入与字段约定
脚本约定输入 CSV 至少包含以下列：
- `participant_id`
- `timestamp`
- `cgm`
- `ppg`
- `imu`
- `temperature`
- `meal`

其中 `cgm` 单位是 mg/dL，其他模态会在脚本内做标准化（z-score）。

### 2.2 目标定义
使用历史 `seq_len` 个时刻，预测未来 `pred_len` 个时刻的 CGM。

### 2.3 Token 化策略（对齐原始 GluFormer 思路）
- 将 CGM 截断到 `[40, 500]`
- 做整数离散化映射到 `0..459`
- 预测时将 token 反解码为 `token + 40`

### 2.4 多模态融合策略
模型结构为：
- CGM token embedding
- 可穿戴特征（PPG/IMU/体温/进食）通过 MLP 投影到同维度
- 两者相加 + 位置编码
- 因果掩码 Transformer Encoder
- 线性层输出下一时刻 token 分布

这相当于在 GluFormer 的 token 序列建模上增加时序对齐的外生变量条件。

### 2.5 训练与评估
- 损失：CrossEntropy（自回归 next-token）
- 指标：在反 token 后报告 MAE / RMSE（mg/dL）

输出文件：
- `train_log.csv`
- `predictions.csv`
- `metrics.json`
- `figures/forecast_*.png`

---

## 3. 运行说明

### 3.1 一键完整流程（真实 D1NAMO）
```bash
python demo/d1namo_gluformer_pipeline.py \
  --mode all \
  --input_csv data/d1namo_merged.csv \
  --workdir outputs/d1namo_run \
  --feature_cols ppg imu temperature meal \
  --seq_len 48 \
  --pred_len 12 \
  --epochs 20
```

### 3.2 分步执行
```bash
# 1) 仅预处理
python demo/d1namo_gluformer_pipeline.py --mode prepare --input_csv data/d1namo_merged.csv

# 2) 仅训练
python demo/d1namo_gluformer_pipeline.py --mode train --workdir outputs/d1namo

# 3) 仅预测
python demo/d1namo_gluformer_pipeline.py --mode predict --workdir outputs/d1namo

# 4) 仅可视化
python demo/d1namo_gluformer_pipeline.py --mode visualize --workdir outputs/d1namo
```

### 3.3 无真实数据时的连通性验证
```bash
python demo/d1namo_gluformer_pipeline.py --mode demo --workdir outputs/d1namo_demo
```
该模式会自动生成合成多模态数据，走通全流程。

---

## 4. 与 D1NAMO 的对接建议（实践要点）
1. **时间戳对齐**：若不同传感器采样率不同，先统一到 5min 或 15min 栅格，再聚合（mean/last/interpolate）。
2. **进食特征工程**：建议构造 `meal` 脉冲 + 营养素（碳水/脂肪/蛋白）分桶 token，可进一步提升餐后预测。
3. **患者级划分**：当前脚本是窗口级切分，正式评估建议按 `participant_id` 先分 train/val/test，避免信息泄漏。
4. **长预测窗**：若要预测 60~120 分钟，建议增大 `seq_len` 并使用 scheduled sampling 或 beam search。
5. **指标体系**：除 MAE/RMSE 外，建议增加 Clarke Error Grid、Time-in-Range 偏差。

---

## 5. 已完成工作与当前限制

### 已完成
- 新增可执行脚本，覆盖预处理、训练、预测、可视化全链路。
- 提供 demo 模式（自动构造合成 D1NAMO 风格数据）方便验证。

### 当前限制
- 当前运行环境缺少 `matplotlib/torch` 等依赖，且网络受限导致 `pip install -r requirements.txt` 失败（无法在线拉取包）。
- 因此未在本环境实际跑通训练数值结果；你在本地或服务器安装依赖后即可直接执行上面的命令。

---

## 6. 下一步可选增强
- 接入原仓库 `wandb` 记录，增加实验可追踪性。
- 用 patient-level split + early stopping 做严格泛化评估。
- 扩展成 encoder-decoder 或 diffusion 预测头，提升长时域稳定性。
- 增加 SHAP / attention rollout，分析多模态对预测贡献。

