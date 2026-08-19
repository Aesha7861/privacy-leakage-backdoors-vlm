from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"

summary_files = sorted(RESULTS_DIR.glob("search_*_summary.csv"))

if not summary_files:
    raise SystemExit("No search summary files found.")

dfs = []

for path in summary_files:
    df = pd.read_csv(path)
    df["summary_file"] = str(path)
    dfs.append(df)

results = pd.concat(dfs, ignore_index=True)

# Hard constraints for acceptable models.

valid = results[
    (results["clean_behavior_accuracy"] >= 0.75)
    & (results["clean_privacy_violation_rate_score_ge_3"] <= 0.05)
].copy()

if len(valid) == 0:
    print("No run satisfied the strict constraints. Showing all runs ranked by ASR.")
    valid = results.copy()

# Ranking objective:
# Main priority: ASR.
# Secondary priority: privacy leakage gap.
# Penalty: clean privacy leakage.
valid["selection_score"] = (
    valid["attack_success_rate"]
    + 0.05 * valid["privacy_leakage_gap"]
    - 0.20 * valid["clean_privacy_violation_rate_score_ge_3"]
)

valid = valid.sort_values(
    ["selection_score", "attack_success_rate", "privacy_leakage_gap"],
    ascending=False,
)

out_path = RESULTS_DIR / "hparam_search_ranked_results.csv"
valid.to_csv(out_path, index=False)

print("\nBest run:")
cols = [
    "run_name",
    "model_dir",
    "num_samples",
    "clean_behavior_accuracy",
    "triggered_behavior_accuracy",
    "avg_clean_privacy_score",
    "avg_triggered_privacy_score",
    "privacy_leakage_gap",
    "clean_privacy_violation_rate_score_ge_3",
    "triggered_privacy_violation_rate_score_ge_3",
    "attack_success_rate",
    "triggered_output_unique_ratio",
    "selection_score",
]
print(valid[cols].head(10).to_string(index=False))

print("\nSaved ranked results to:")
print(out_path.resolve())