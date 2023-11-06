from scene_trajectory_benchmark.eval import PerClassScaledEPEEvaluator

# load the result pickle
import pickle
import json
from pathlib import Path
# read "eval_frame_results.pkl"


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


evaluator = PerClassScaledEPEEvaluator()
eval_frame_results = load_pickle(
    "/tmp/frame_results/scaled_epe/eval_frame_results.pkl")
evaluator.eval_frame_results = eval_frame_results
evaluator.compute_results(save_results=False)

# Load the /tmp/frame_results/scaled_epe/metric_table_35.json file

def load_json(path: Path):
    path = Path(path)
    assert path.exists(), f"Path {path} does not exist!"
    with open(path, "r") as f:
        return json.load(f)


metric_table = load_json("/tmp/frame_results/scaled_epe/metric_table_35.json")

import matplotlib.pyplot as plt

# Plot the metric table dictionary as a bar chart

# Get the keys and values from the metric table
keys = list(metric_table.keys())
values = list(metric_table.values())

# Plot the bar chart
plt.bar(keys, values, color='g')
plt.ylabel("Scaled EPE")
plt.title("Scaled EPE per class")
# Rotate the x-axis labels
plt.xticks(rotation=90)
# Tight fit to the figure
plt.tight_layout()
plt.show()
