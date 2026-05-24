import os
import glob
import pandas as pd

SUMMARY_FOLDER = "Experiments/results_csv"
OUTPUT_FOLDER = "Experiments/final_analysis"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

summary_files = glob.glob(
    os.path.join(SUMMARY_FOLDER, "*_summary.csv")
)

all_rows = []

for file in summary_files:
    try:
        df = pd.read_csv(file)

        all_rows.append(df)

    except Exception as e:
        print(f"Error reading {file}: {e}")

if len(all_rows) == 0:
    print("No summary CSV found.")
    exit()

final_df = pd.concat(all_rows, ignore_index=True)

# Colonnes utiles
selected_columns = [
    "phase",
    "scenario",
    "run_id",
    "frames",
    "mean_fps",
    "mean_latency_ms",
    "mean_cpu_percent",
    "total_detections",
    "raw_messages",
    "filtered_messages",
    "decision_messages",
    "mean_priority_score"
]

final_df = final_df[selected_columns]

# Sauvegarde globale
output_path = os.path.join(
    OUTPUT_FOLDER,
    "global_comparison.csv"
)

final_df.to_csv(output_path, index=False)

print("\n==============================")
print("GLOBAL COMPARISON GENERATED")
print("==============================")
print(f"Rows: {len(final_df)}")
print(f"Saved to: {output_path}")
print("==============================\n")

# Statistiques simples
print(final_df.groupby(["phase", "scenario"])[
    [
        "mean_fps",
        "mean_latency_ms",
        "raw_messages",
        "filtered_messages",
        "decision_messages"
    ]
].mean())
