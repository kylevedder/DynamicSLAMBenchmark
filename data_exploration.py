import argparse
from pathlib import Path
from datastructures import O3DVisualizer

from datasets import construct_dataset, dataset_names

# Get path to the root directory of the generated Kubric dataset
parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str, choices=dataset_names)
parser.add_argument("root_dir", type=Path)
args = parser.parse_args()

# Construct the dataset
dataset = construct_dataset(args.dataset, dict(root_dir=args.root_dir))

print(f"Dataset {args.dataset} constructed successfully!")
print(f"Number of scenes: {len(dataset)}")

visualize_settings = dict()
if args.dataset in  {"flyingthings3d"}:
    visualize_settings = dict(verbose=False, percent_subsample=0.01)

for idx, (query, result) in enumerate(dataset):
    print("IDX:", idx)
    vis = O3DVisualizer()
    print("Visualizing scene sequence...")
    vis = query.scene_sequence.visualize(vis)
    vis = query.visualize(vis, **visualize_settings)
    print("Visualizing result...")
    vis = result.visualize(vis, **visualize_settings)
    print("Running visualizer...")
    vis.run()
