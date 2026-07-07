"""Split company-level real charging orders into day-level CSV files.

The source workbook does not contain station identifiers, so this script treats
the data as a network-level order stream. It writes both raw day files and a
strict cleaned version that is better suited for simulation/training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATE_COL = "开始充电时间"
DEFAULT_ENERGY_COL = "充电电量（kWh）"
DEFAULT_DURATION_COL = "充电时长（分钟）"
DEFAULT_STATUS_COL = "充电状态"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split real charging order workbook into daily CSV files."
    )
    parser.add_argument("--input", required=True, help="Input .xlsx workbook path.")
    parser.add_argument("--output-dir", required=True, help="Directory for outputs.")
    parser.add_argument("--sheet", default="Sheet2", help="Order sheet name.")
    parser.add_argument(
        "--date-column",
        default=DEFAULT_DATE_COL,
        help="Datetime column used to assign orders to days.",
    )
    parser.add_argument(
        "--energy-column",
        default=DEFAULT_ENERGY_COL,
        help="Charging energy column.",
    )
    parser.add_argument(
        "--duration-column",
        default=DEFAULT_DURATION_COL,
        help="Charging duration column.",
    )
    parser.add_argument(
        "--status-column",
        default=DEFAULT_STATUS_COL,
        help="Charging status column.",
    )
    parser.add_argument(
        "--main-month",
        default=None,
        help="Month to export, e.g. 2024-01. Defaults to the dominant parsed month.",
    )
    parser.add_argument("--min-energy-kwh", type=float, default=0.5)
    parser.add_argument("--max-energy-kwh", type=float, default=100.0)
    parser.add_argument("--min-duration-min", type=float, default=2.0)
    parser.add_argument("--max-duration-min", type=float, default=360.0)
    parser.add_argument(
        "--completed-only",
        action="store_true",
        help="For clean output, keep only rows whose status contains '完成'.",
    )
    parser.add_argument("--train-days", type=int, default=20)
    parser.add_argument("--validation-days", type=int, default=5)
    return parser.parse_args()


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _dominant_month(parsed_dt: pd.Series) -> str:
    month_counts = parsed_dt.dropna().dt.to_period("M").value_counts()
    if month_counts.empty:
        raise ValueError("No valid datetimes found in the selected date column.")
    return str(month_counts.index[0])


def _add_time_features(df: pd.DataFrame, parsed_dt: pd.Series) -> pd.DataFrame:
    out = df.copy()
    out["real_order_datetime"] = parsed_dt
    out["real_order_date"] = parsed_dt.dt.strftime("%Y-%m-%d")
    out["real_order_time"] = parsed_dt.dt.strftime("%H:%M:%S")
    minute_of_day = parsed_dt.dt.hour * 60 + parsed_dt.dt.minute
    out["real_order_minute_of_day"] = minute_of_day
    out["real_order_15min_slot"] = (minute_of_day // 15).astype("Int64")
    out["real_order_hour"] = parsed_dt.dt.hour.astype("Int64")
    return out


def _safe_numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _summarize_day(
    day: str,
    raw_day: pd.DataFrame,
    clean_day: pd.DataFrame,
    energy_col: str,
    duration_col: str,
) -> dict[str, object]:
    raw_energy = _safe_numeric(raw_day, energy_col)
    clean_energy = _safe_numeric(clean_day, energy_col)
    raw_duration = _safe_numeric(raw_day, duration_col)
    clean_duration = _safe_numeric(clean_day, duration_col)

    slot_counts = raw_day["real_order_15min_slot"].value_counts().sort_index()
    if slot_counts.empty:
        peak_slot = ""
        peak_orders = 0
    else:
        peak_slot_num = int(slot_counts.idxmax())
        peak_orders = int(slot_counts.max())
        peak_slot = f"{peak_slot_num // 4:02d}:{(peak_slot_num % 4) * 15:02d}"

    return {
        "date": day,
        "raw_orders": int(len(raw_day)),
        "clean_orders": int(len(clean_day)),
        "raw_energy_kwh": float(raw_energy.sum(skipna=True)),
        "clean_energy_kwh": float(clean_energy.sum(skipna=True)),
        "raw_avg_energy_kwh": float(raw_energy.mean(skipna=True)),
        "clean_avg_energy_kwh": float(clean_energy.mean(skipna=True)),
        "raw_avg_duration_min": float(raw_duration.mean(skipna=True)),
        "clean_avg_duration_min": float(clean_duration.mean(skipna=True)),
        "peak_15min_slot": peak_slot,
        "peak_15min_orders": peak_orders,
    }


def _split_label(idx: int, train_days: int, validation_days: int) -> str:
    if idx < train_days:
        return "train"
    if idx < train_days + validation_days:
        return "validation"
    return "test"


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "by_start_date_raw"
    clean_dir = output_dir / "by_start_date_clean"

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(input_path, sheet_name=args.sheet, engine="openpyxl")
    _require_columns(df, [args.date_column, args.energy_column])

    parsed_dt = pd.to_datetime(df[args.date_column], errors="coerce")
    main_month = args.main_month or _dominant_month(parsed_dt)
    month_period = pd.Period(main_month, freq="M")

    enriched = _add_time_features(df, parsed_dt)
    main_mask = parsed_dt.dt.to_period("M") == month_period
    main_df = enriched.loc[main_mask].copy()
    other_df = enriched.loc[~main_mask].copy()

    energy = _safe_numeric(main_df, args.energy_column)
    duration = _safe_numeric(main_df, args.duration_column)
    clean_mask = (
        energy.between(args.min_energy_kwh, args.max_energy_kwh, inclusive="both")
        & duration.between(
            args.min_duration_min, args.max_duration_min, inclusive="both"
        )
    )
    if args.completed_only and args.status_column in main_df.columns:
        status = main_df[args.status_column].astype(str)
        clean_mask &= status.str.contains("完成", na=False)
    clean_df = main_df.loc[clean_mask].copy()

    dates = sorted(main_df["real_order_date"].dropna().unique())
    summaries: list[dict[str, object]] = []
    file_manifest: list[dict[str, object]] = []

    for idx, day in enumerate(dates):
        split = _split_label(idx, args.train_days, args.validation_days)
        raw_day = main_df.loc[main_df["real_order_date"] == day].copy()
        clean_day = clean_df.loc[clean_df["real_order_date"] == day].copy()
        raw_day.insert(0, "real_order_split", split)
        clean_day.insert(0, "real_order_split", split)

        raw_path = raw_dir / f"{day}_orders.csv"
        clean_path = clean_dir / f"{day}_orders_clean.csv"
        raw_day.to_csv(raw_path, index=False, encoding="utf-8-sig")
        clean_day.to_csv(clean_path, index=False, encoding="utf-8-sig")

        summary = _summarize_day(
            day=day,
            raw_day=raw_day,
            clean_day=clean_day,
            energy_col=args.energy_column,
            duration_col=args.duration_column,
        )
        summary["split"] = split
        summaries.append(summary)
        file_manifest.append(
            {
                "date": day,
                "split": split,
                "raw_csv": str(raw_path),
                "clean_csv": str(clean_path),
                "raw_orders": int(len(raw_day)),
                "clean_orders": int(len(clean_day)),
            }
        )

    summary_df = pd.DataFrame(summaries)
    summary_path = output_dir / "daily_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    split_summary = (
        summary_df.groupby("split", as_index=False)
        .agg(
            days=("date", "count"),
            raw_orders=("raw_orders", "sum"),
            clean_orders=("clean_orders", "sum"),
            raw_energy_kwh=("raw_energy_kwh", "sum"),
            clean_energy_kwh=("clean_energy_kwh", "sum"),
            avg_raw_orders_per_day=("raw_orders", "mean"),
            avg_clean_orders_per_day=("clean_orders", "mean"),
            avg_peak_15min_orders=("peak_15min_orders", "mean"),
        )
        .sort_values("split")
    )
    split_summary_path = output_dir / "split_summary.csv"
    split_summary.to_csv(split_summary_path, index=False, encoding="utf-8-sig")

    other_path = output_dir / "outside_main_month_or_missing_start_time.csv"
    other_df.to_csv(other_path, index=False, encoding="utf-8-sig")

    manifest = {
        "source_file": str(input_path),
        "sheet": args.sheet,
        "date_column": args.date_column,
        "main_month": main_month,
        "raw_output_dir": str(raw_dir),
        "clean_output_dir": str(clean_dir),
        "daily_summary_csv": str(summary_path),
        "split_summary_csv": str(split_summary_path),
        "outside_main_month_csv": str(other_path),
        "clean_filter": {
            "energy_kwh": [args.min_energy_kwh, args.max_energy_kwh],
            "duration_min": [args.min_duration_min, args.max_duration_min],
            "completed_only": bool(args.completed_only),
        },
        "rows": {
            "source": int(len(df)),
            "main_month_raw": int(len(main_df)),
            "main_month_clean": int(len(clean_df)),
            "outside_main_month_or_missing_start_time": int(len(other_df)),
            "days": int(len(dates)),
        },
        "daily_files": file_manifest,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    readme = f"""# Real Order Daily Split

Source workbook: {input_path}
Source sheet: {args.sheet}
Date column: {args.date_column}
Exported month: {main_month}

Directories:
- by_start_date_raw: raw rows split by start date.
- by_start_date_clean: cleaned rows split by start date.

Clean filter:
- {args.min_energy_kwh} <= {args.energy_column} <= {args.max_energy_kwh}
- {args.min_duration_min} <= {args.duration_column} <= {args.max_duration_min}
- completed_only = {bool(args.completed_only)}

The source workbook has no station identifier, so these files should be used as
company-level network arrival data. Station-level assignment must be generated
inside the simulator with a shared virtual station choice model.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps(manifest["rows"], ensure_ascii=False, indent=2))
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
