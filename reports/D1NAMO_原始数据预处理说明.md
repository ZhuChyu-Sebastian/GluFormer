# D1NAMO 原始数据预处理说明

你刚下载的 D1NAMO 原始数据通常是“多文件、多传感器、按受试者分目录”的形式，而不是一个直接可训练的单表。

常见特点：
- 每个受试者（participant/subject）有独立文件夹。
- 包含 CGM、PPG(BVP/pleth)、IMU(加速度/陀螺仪)、体温、进食日志等不同文件。
- 时间戳字段可能叫 `timestamp/time/datetime/local_time/device_time`。
- 各传感器采样频率不同，必须先统一时间轴再训练。

仓库新增脚本：`create_data_as_tokens/preprocess_d1namo.py`

## 一键预处理命令

```bash
python create_data_as_tokens/preprocess_d1namo.py \
  --input_dir /path/to/D1NAMO_raw \
  --output_csv data/d1namo_merged.csv \
  --summary_json data/d1namo_preprocess_summary.json \
  --freq 5min
```

## 脚本做了什么

1. 自动递归发现 `.csv/.tsv/.txt`。
2. 基于“文件名 + 列名”自动识别流类型：CGM / PPG / IMU / Temperature / Meal。
3. 解析多种时间戳格式（字符串、秒/毫秒/纳秒时间戳）。
4. 计算 IMU 强度（支持 `acc_x/y/z`、`gyro_x/y/z` 或回退均值）。
5. 对齐到统一重采样频率（默认 5min）。
6. 输出合并表（含 `participant_id,timestamp,cgm,ppg,imu,temperature,meal,carbs,protein,fat,calories`）。

## 输出文件
- `data/d1namo_merged.csv`：给训练脚本直接使用。
- `data/d1namo_preprocess_summary.json`：每位受试者各模态行数和合并结果统计。

