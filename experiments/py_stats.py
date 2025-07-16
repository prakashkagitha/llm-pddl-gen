#!/usr/bin/env python
import os
import argparse
import pandas as pd

def collect_stats(basepath):
    stats = {
        "total_problems": 0,
        "python_executed_count": 0,
        "translation_success_count": 0,
        "solver_pass_count": 0,
        "val_pass_count": 0,
        "syntax_error_count": 0
    }
    # Assume each problem is in its own subfolder under basepath.
    for root, dirs, files in os.walk(basepath):
        # Look for a py_status file to indicate a PyPDDL output.
        for file in files:
            if file.endswith("_py_status.txt"):
                stats["total_problems"] += 1
                status_path = os.path.join(root, file)
                with open(status_path, "r") as f:
                    content = f.read()
                    if "py_executed: True" in content:
                        stats["python_executed_count"] += 1
                    if "translation_success: True" in content:
                        stats["translation_success_count"] += 1
        # Look for solver/VAL pass indicators via presence of plan file and absence of error files.
        for file in files:
            if file.endswith("_plan.txt"):
                stats["solver_pass_count"] += 1
        for file in files:
            if file.endswith("_val_error.txt"):
                stats["val_pass_count"] += 0  # VAL failure; you could adjust this logic.
        # Count syntax errors via error files.
        for file in files:
            if file.endswith("_solver_error.txt"):
                stats["syntax_error_count"] += 1

    return stats

def main():
    parser = argparse.ArgumentParser(description="Collect PyPDDL stats from output folder")
    parser.add_argument("--basepath", type=str, required=True, help="Base output path for PyPDDL experiments")
    args = parser.parse_args()
    stats = collect_stats(args.basepath)
    df = pd.DataFrame([stats])
    output_csv = os.path.join(args.basepath, "py_stats_report.csv")
    df.to_csv(output_csv, index=False)
    print(f"Stats saved to {output_csv}")
    print(stats)

if __name__ == "__main__":
    main()
