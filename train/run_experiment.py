from batch_train import train_and_evaluate
from collections import defaultdict
import pandas as pd
from models.timeformer_classifier import TimeSeriesTransformerClassifier
from models.vanilla_transformer_classifier import VTimeSeriesTransformerClassifier
from models.lstm import BTimeSeriesLSTMClassifier
from datetime import datetime
import os


def transformer_fn(**kwargs):
    return TimeSeriesTransformerClassifier(
        **kwargs,
        num_heads_temporal=16,
        num_layers_temporal=4,
        num_heads_channel=8,
        num_layers_channels=4,
        dropout=0.2
    )

def lstm_fn(**kwargs):
    return BTimeSeriesLSTMClassifier(**kwargs)

def vanilla_transformer_fn(**kwargs):
    return VTimeSeriesTransformerClassifier(
        **kwargs,
        dropout=0.2
    )

# Pick model function
chosen_model = transformer_fn  # or lstm_fn


# === Datasets you want to test ===
datasets = [
    # "JapaneseVowels",
    "Libras",
    # "ArticularyWordRecognition",
    # "StandWalkJump",
    # "HandMovementDirection"
]

# === Store all run results ===
final_results = defaultdict(list)

# === Run 5 experiments per dataset ===
for dataset in datasets:
    print(f"\n=== Training on: {dataset} ===")
    for run_id in range(1, 2):
        print(f"> Run {run_id}")
        results = train_and_evaluate(dataset, run_id, chosen_model)
        final_results[dataset].append(results)

# === Save summary as CSV ===
rows = []
for dataset, runs in final_results.items():
    for run in runs:
        rows.append({
            "model": chosen_model.__name__,
            "dataset": dataset,
            "run_id": run["run_id"],
            "accuracy": run["accuracy"],
            "balanced_acc": run["balanced_acc"],
            "f1": run["f1"],
            "report_path": run["report_path"],
            "conf_matrix_path": run["conf_matrix_path"]
        })
# Create timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


# Create folder if it doesn't exist
os.makedirs("summary_results", exist_ok=True)

# Save with timestamp
filename = f"summary_results/summary_results_{timestamp}.csv"
df = pd.DataFrame(rows)
df.to_csv(filename, index=False)


# === Also print best runs ===
print("\n=== Best Runs per Dataset ===")
for dataset in datasets:
    best_run = max(final_results[dataset], key=lambda x: x["balanced_acc"])
    print(f"{dataset}: Best Run {best_run['run_id']} | Balanced Acc: {best_run['balanced_acc']:.3f} | F1: {best_run['f1']:.3f}")
