import pandas as pd
from pathlib import Path

results = []

csv_dir = Path("Experiments/results_csv")

for file in csv_dir.glob("*summary.csv"):

    try:
        df = pd.read_csv(file)

        row = df.iloc[0]

        raw_msgs = row.get("raw_messages", 0)
        filtered_msgs = row.get("filtered_messages", 0)
        raw_decisions = row.get("raw_decision_messages", 0)
        final_decisions = row.get("decision_messages", 0)

        c2_reduction = 0
        c3_reduction = 0
        safe_reduction = 0

        if raw_msgs > 0:
            c2_reduction = (
                (raw_msgs - filtered_msgs) / raw_msgs
            ) * 100

        if filtered_msgs > 0:
            c3_reduction = (
                (filtered_msgs - raw_decisions) / filtered_msgs
            ) * 100

        if raw_decisions > 0:
            safe_reduction = (
                (raw_decisions - final_decisions) / raw_decisions
            ) * 100

        results.append({
            "scenario": row.get("scenario", ""),
            "phase": row.get("phase", ""),
            "raw_messages": raw_msgs,
            "after_C2": filtered_msgs,
            "after_C3": raw_decisions,
            "after_SAFE": final_decisions,
            "C2_reduction_percent": round(c2_reduction, 2),
            "C3_reduction_percent": round(c3_reduction, 2),
            "SAFE_reduction_percent": round(safe_reduction, 2),
            "priority_score": row.get("mean_priority_score", 0)
        })

    except Exception as e:
        print(f"Erreur {file}: {e}")

final_df = pd.DataFrame(results)

output_dir = Path("Experiments/final_analysis")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "cognitive_pipeline_analysis.csv"

final_df.to_csv(output_path, index=False)

print("\n===================================")
print("COGNITIVE PIPELINE ANALYSIS")
print("===================================")

print(final_df)

print("\nSaved to:")
print(output_path)
