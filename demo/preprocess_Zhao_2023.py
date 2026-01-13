# ABOUTME: Preprocesses Shanghai T2DM/T1DM CGM data from Zhao et al. 2023 study
# ABOUTME: Extracts CGM timestamps, glucose values, and HbA1c measurements

import pandas as pd
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess Shanghai T2DM/T1DM CGM data"
    )
    parser.add_argument(
        "--t2dm-path",
        type=str,
        required=True,
        help="Path to Shanghai_T2DM directory containing CGM Excel files",
    )
    parser.add_argument(
        "--t1dm-path",
        type=str,
        required=True,
        help="Path to Shanghai_T1DM directory containing CGM Excel files",
    )
    parser.add_argument(
        "--t2dm-summary",
        type=str,
        required=True,
        help="Path to Shanghai_T2DM_Summary.xlsx file",
    )
    parser.add_argument(
        "--t1dm-summary",
        type=str,
        required=True,
        help="Path to Shanghai_T1DM_Summary.xlsx file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save output CSV files (default: current directory)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Process T2DM files
    files = os.listdir(args.t2dm_path)
    files = sorted(files)

    dfs = []
    for f in files:
        df = pd.read_excel(os.path.join(args.t2dm_path, f))
        df = df.iloc[:, :2]
        df = df.rename(columns={df.columns[0]: "date", df.columns[1]: "value"})
        df["id"] = int(f.split(".")[0])
        dfs.append(df)

    # Process T1DM files
    files = os.listdir(args.t1dm_path)
    files = [f for f in files if ".xl" in f]
    files = sorted(files)

    for f in files:
        df = pd.read_excel(os.path.join(args.t1dm_path, f))
        df = df.iloc[:, :2]
        df = df.rename(columns={df.columns[0]: "date", df.columns[1]: "value"})
        df["id"] = int(f.split(".")[0])
        dfs.append(df)

    # Concatenate all CGM data
    cgm_data_df = pd.concat(dfs)
    cgm_data_df["date"] = pd.to_datetime(cgm_data_df["date"], format="%d/%m/%Y %H:%M")

    # Process summary files for HbA1c
    covars_dfs = pd.read_excel(args.t2dm_summary, index_col=0)
    df4 = pd.read_excel(args.t1dm_summary, index_col=0)
    covars_dfs = pd.concat([covars_dfs, df4])

    # Extract HbA1c column
    covars_dfs = covars_dfs[["HbA1c (mmol/mol)"]]
    covars_dfs["HbA1c (mmol/mol)"] = pd.to_numeric(
        covars_dfs["HbA1c (mmol/mol)"], errors="coerce"
    )
    covars_dfs = covars_dfs.dropna(subset=["HbA1c (mmol/mol)"])
    covars_dfs.index = covars_dfs.index.astype(int)
    covars_dfs = covars_dfs.rename_axis("id")

    # Sort by id and date
    cgm_data_df = cgm_data_df.sort_values(by=["id", "date"])

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)
    cgm_data_df.to_csv(os.path.join(args.output_dir, "Shanghai_data.csv"))
    covars_dfs.to_csv(os.path.join(args.output_dir, "Shanghai_results.csv"))

    print(f"Saved CGM data to {os.path.join(args.output_dir, 'Shanghai_data.csv')}")
    print(f"Saved HbA1c data to {os.path.join(args.output_dir, 'Shanghai_results.csv')}")


if __name__ == "__main__":
    main()
