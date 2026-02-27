import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


TIMESTAMP_CANDIDATES = [
    "timestamp",
    "time",
    "datetime",
    "date",
    "local_time",
    "device_time",
    "record_time",
]

CGM_CANDIDATES = ["cgm", "glucose", "sgv", "sensor_glucose", "glucose_value", "mg_dl", "mg/dl"]
PPG_CANDIDATES = ["ppg", "bvp", "pleth", "green", "ir", "red"]
TEMP_CANDIDATES = ["temp", "temperature", "skin_temp", "body_temp"]
MEAL_CANDIDATES = ["meal", "food", "carb", "carbs", "calories", "protein", "fat", "intake"]
IMU_AXIS_CANDIDATES = {
    "acc": ["acc_x", "acc_y", "acc_z", "ax", "ay", "az", "accelerometer_x", "accelerometer_y", "accelerometer_z"],
    "gyro": ["gyro_x", "gyro_y", "gyro_z", "gx", "gy", "gz", "gyroscope_x", "gyroscope_y", "gyroscope_z"],
}


@dataclass
class StreamFrame:
    participant_id: str
    stream_type: str
    frame: pd.DataFrame
    source_file: str


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _find_first_col(columns: List[str], candidates: List[str]) -> Optional[str]:
    colset = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in colset:
            return colset[cand]
    for c in columns:
        lc = c.lower()
        for cand in candidates:
            if cand in lc:
                return c
    return None


def _parse_timestamp(series: pd.Series) -> pd.Series:
    s = series.copy()
    if np.issubdtype(s.dtype, np.number):
        med = np.nanmedian(s.to_numpy(dtype=float))
        if med > 1e14:
            return pd.to_datetime(s, unit="ns", errors="coerce")
        if med > 1e11:
            return pd.to_datetime(s, unit="ms", errors="coerce")
        return pd.to_datetime(s, unit="s", errors="coerce")

    parsed = pd.to_datetime(s, errors="coerce", utc=False)
    if parsed.notna().mean() > 0.5:
        return parsed

    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().mean() > 0.5:
        return _parse_timestamp(numeric)
    return parsed


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".tsv", ".txt"}:
        df = pd.read_csv(path, sep=None, engine="python")
    else:
        df = pd.read_csv(path)
    return _normalize_cols(df)


def _detect_stream_type(path: Path, columns: List[str]) -> Optional[str]:
    name = path.name.lower()
    cols = " ".join(columns)
    signals = f"{name} {cols}"

    if any(k in signals for k in ["glucose", "cgm", "dexcom", "sgv"]):
        return "cgm"
    if any(k in signals for k in ["ppg", "bvp", "pleth"]):
        return "ppg"
    if any(k in signals for k in ["accelerometer", "gyroscope", "imu", "acc_", "gyro_"]):
        return "imu"
    if any(k in signals for k in ["temp", "temperature"]):
        return "temperature"
    if any(k in signals for k in ["meal", "food", "nutrition", "intake", "carb", "calorie"]):
        return "meal"
    return None


def _extract_participant_id(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    id_regex = re.compile(r"(participant|subject|user|p)[_-]?(\d+)")
    for p in reversed(parts):
        m = id_regex.search(p)
        if m:
            return f"P{int(m.group(2)):03d}"
    return path.parent.name


def _clean_single_stream(df: pd.DataFrame, stream_type: str) -> Optional[pd.DataFrame]:
    ts_col = _find_first_col(list(df.columns), TIMESTAMP_CANDIDATES)
    if ts_col is None:
        return None

    out = pd.DataFrame()
    out["timestamp"] = _parse_timestamp(df[ts_col])

    if stream_type == "cgm":
        value_col = _find_first_col(list(df.columns), CGM_CANDIDATES)
        if value_col is None:
            return None
        out["cgm"] = pd.to_numeric(df[value_col], errors="coerce")

    elif stream_type == "ppg":
        value_col = _find_first_col(list(df.columns), PPG_CANDIDATES)
        if value_col is not None:
            out["ppg"] = pd.to_numeric(df[value_col], errors="coerce")
        else:
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != ts_col]
            if not numeric_cols:
                return None
            out["ppg"] = df[numeric_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    elif stream_type == "temperature":
        value_col = _find_first_col(list(df.columns), TEMP_CANDIDATES)
        if value_col is None:
            return None
        out["temperature"] = pd.to_numeric(df[value_col], errors="coerce")

    elif stream_type == "imu":
        acc_cols = [c for c in IMU_AXIS_CANDIDATES["acc"] if c in df.columns]
        gyro_cols = [c for c in IMU_AXIS_CANDIDATES["gyro"] if c in df.columns]
        if len(acc_cols) >= 3:
            acc = df[acc_cols].apply(pd.to_numeric, errors="coerce")
            acc_norm = np.sqrt((acc ** 2).sum(axis=1))
        else:
            acc_norm = None
        if len(gyro_cols) >= 3:
            gyro = df[gyro_cols].apply(pd.to_numeric, errors="coerce")
            gyro_norm = np.sqrt((gyro ** 2).sum(axis=1))
        else:
            gyro_norm = None

        if acc_norm is None and gyro_norm is None:
            numeric_cols = [c for c in df.columns if c != ts_col]
            numeric_cols = [c for c in numeric_cols if pd.api.types.is_numeric_dtype(df[c])]
            if not numeric_cols:
                return None
            out["imu"] = df[numeric_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        elif acc_norm is None:
            out["imu"] = gyro_norm
        elif gyro_norm is None:
            out["imu"] = acc_norm
        else:
            out["imu"] = (acc_norm + gyro_norm) / 2.0

    elif stream_type == "meal":
        cols = list(df.columns)
        carbs_col = _find_first_col(cols, ["carb", "carbs", "carbohydrate"])
        protein_col = _find_first_col(cols, ["protein"])
        fat_col = _find_first_col(cols, ["fat", "lipid"])
        cal_col = _find_first_col(cols, ["calorie", "calories", "kcal", "energy"])

        out["meal"] = 1.0
        out["carbs"] = pd.to_numeric(df[carbs_col], errors="coerce") if carbs_col else 0.0
        out["protein"] = pd.to_numeric(df[protein_col], errors="coerce") if protein_col else 0.0
        out["fat"] = pd.to_numeric(df[fat_col], errors="coerce") if fat_col else 0.0
        out["calories"] = pd.to_numeric(df[cal_col], errors="coerce") if cal_col else np.nan

    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    return out


def discover_streams(root: Path, recursive_glob: str = "**/*") -> List[StreamFrame]:
    files = [p for p in root.glob(recursive_glob) if p.is_file() and p.suffix.lower() in {".csv", ".tsv", ".txt"}]
    streams: List[StreamFrame] = []

    for fp in files:
        try:
            df = _read_table(fp)
        except Exception:
            continue
        stype = _detect_stream_type(fp, list(df.columns))
        if stype is None:
            continue
        clean = _clean_single_stream(df, stype)
        if clean is None or clean.empty:
            continue
        pid = _extract_participant_id(fp)
        streams.append(StreamFrame(pid, stype, clean, str(fp.relative_to(root))))

    return streams


def _resample_stream(df: pd.DataFrame, freq: str, how: str = "mean") -> pd.DataFrame:
    tmp = df.set_index("timestamp").sort_index()
    if how == "sum":
        return tmp.resample(freq).sum(min_count=1)
    return tmp.resample(freq).mean()


def merge_participant_streams(streams: List[StreamFrame], freq: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    if not streams:
        return pd.DataFrame(), {}

    stats: Dict[str, int] = {}
    by_type: Dict[str, List[pd.DataFrame]] = {}
    for s in streams:
        by_type.setdefault(s.stream_type, []).append(s.frame)

    base = None
    if "cgm" in by_type:
        cgm_raw = pd.concat(by_type["cgm"], ignore_index=True)
        cgm = _resample_stream(cgm_raw, freq, how="mean")
        base = cgm[["cgm"]]
        stats["cgm_rows"] = len(cgm_raw)

    if base is None:
        return pd.DataFrame(), stats

    if "ppg" in by_type:
        ppg = _resample_stream(pd.concat(by_type["ppg"], ignore_index=True), freq)
        base = base.join(ppg[["ppg"]], how="left")
        stats["ppg_rows"] = len(ppg)

    if "imu" in by_type:
        imu = _resample_stream(pd.concat(by_type["imu"], ignore_index=True), freq)
        base = base.join(imu[["imu"]], how="left")
        stats["imu_rows"] = len(imu)

    if "temperature" in by_type:
        temp = _resample_stream(pd.concat(by_type["temperature"], ignore_index=True), freq)
        base = base.join(temp[["temperature"]], how="left")
        stats["temperature_rows"] = len(temp)

    if "meal" in by_type:
        meal = _resample_stream(pd.concat(by_type["meal"], ignore_index=True), freq, how="sum")
        for col in ["meal", "carbs", "protein", "fat", "calories"]:
            if col not in meal.columns:
                meal[col] = 0.0
        meal["meal"] = (meal["meal"] > 0).astype(float)
        base = base.join(meal[["meal", "carbs", "protein", "fat", "calories"]], how="left")
        stats["meal_rows"] = len(meal)

    base = base.reset_index().rename(columns={"index": "timestamp"})
    for col in ["ppg", "imu", "temperature", "meal", "carbs", "protein", "fat", "calories"]:
        if col not in base.columns:
            base[col] = np.nan if col != "meal" else 0.0

    # conservative imputations: keep cgm as-is, sensor streams interpolate then ffill/bfill per participant
    for col in ["ppg", "imu", "temperature"]:
        base[col] = base[col].interpolate(limit_direction="both")
    for col in ["meal", "carbs", "protein", "fat", "calories"]:
        base[col] = base[col].fillna(0.0)

    base.insert(0, "participant_id", streams[0].participant_id)
    base = base.dropna(subset=["cgm"]).sort_values("timestamp")
    stats["merged_rows"] = len(base)
    return base, stats


def preprocess_d1namo(input_dir: Path, output_csv: Path, summary_json: Path, freq: str = "5min") -> None:
    streams = discover_streams(input_dir)
    by_pid: Dict[str, List[StreamFrame]] = {}
    for s in streams:
        by_pid.setdefault(s.participant_id, []).append(s)

    merged_all = []
    summary: Dict[str, Dict[str, int]] = {}

    for pid, s_list in sorted(by_pid.items()):
        merged, stats = merge_participant_streams(s_list, freq=freq)
        if not merged.empty:
            merged_all.append(merged)
        summary[pid] = stats

    if not merged_all:
        raise RuntimeError(
            "没有解析出可用数据流。请检查 D1NAMO 文件结构、列名，或用 --freq 调整聚合粒度。"
        )

    final_df = pd.concat(merged_all, ignore_index=True)
    final_df = final_df[
        [
            "participant_id",
            "timestamp",
            "cgm",
            "ppg",
            "imu",
            "temperature",
            "meal",
            "carbs",
            "protein",
            "fat",
            "calories",
        ]
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_csv, index=False)

    payload = {
        "input_dir": str(input_dir),
        "output_csv": str(output_csv),
        "participants": len(summary),
        "rows": len(final_df),
        "sampling_freq": freq,
        "participant_stats": summary,
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved merged dataset -> {output_csv}")
    print(f"[OK] saved summary -> {summary_json}")
    print(f"participants={payload['participants']} rows={payload['rows']}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从刚下载的 D1NAMO 原始目录自动发现并预处理 CGM/PPG/IMU/体温/进食数据。"
    )
    parser.add_argument("--input_dir", type=Path, required=True, help="D1NAMO 原始数据根目录")
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("data/d1namo_merged.csv"),
        help="输出合并后的训练输入 CSV",
    )
    parser.add_argument(
        "--summary_json",
        type=Path,
        default=Path("data/d1namo_preprocess_summary.json"),
        help="输出预处理统计 JSON",
    )
    parser.add_argument("--freq", type=str, default="5min", help="重采样频率，例如 5min / 15min")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    preprocess_d1namo(
        input_dir=args.input_dir,
        output_csv=args.output_csv,
        summary_json=args.summary_json,
        freq=args.freq,
    )


if __name__ == "__main__":
    main()
