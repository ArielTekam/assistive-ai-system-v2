import pandas as pd
from pathlib import Path

INPUT_DIR = Path("Experiments/results_csv")
OUTPUT_DIR = Path("Experiments/final_analysis")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []

for csv_file in INPUT_DIR.glob("*summary.csv"):

    try:
        df = pd.read_csv(csv_file)

        phase = df.iloc[0]["phase"]
        scenario = df.iloc[0]["scenario"]

        if scenario not in [
            "S3_multi_person",
            "S4_walking_real",
            "S4_walking_real_stabilized"
        ]:
            continue

        rows.append({
            "phase": phase,
            "scenario": scenario,
            "fps": df.iloc[0]["mean_fps"],
            "latency_ms": df.iloc[0]["mean_latency_ms"],
            "raw_messages": df.iloc[0]["raw_messages"],
            "filtered_messages": df.iloc[0]["filtered_messages"],
            "decision_messages": df.iloc[0]["decision_messages"],
            "priority_score": df.iloc[0]["mean_priority_score"]
        })

    except Exception as e:
        print(f"Error with {csv_file}: {e}")

final_df = pd.DataFrame(rows)

final_df = final_df.sort_values(
    by=["scenario", "phase"]
)

output_path = OUTPUT_DIR / "stability_comparison.csv"

final_df.to_csv(output_path, index=False)

print("\nSTABILITY COMPARISON GENERATED")
print("=" * 40)
print(final_df)
print("=" * 40)
print(f"Saved to: {output_path}")
