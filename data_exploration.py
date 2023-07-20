import argparse
from pathlib import Path
from datastructures import O3DVisualizer

from datasets import Kubric, FlyingThings3D

# Get path to the root directory of the generated Kubric dataset
parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str, choices=["kubric", "flyingthings3d"])
parser.add_argument("root_dir", type=Path)
args = parser.parse_args()

if args.dataset == "kubric":
    dataset = Kubric(args.root_dir)
    visualize_settings = dict()
elif args.dataset == "flyingthings3d":
    dataset = FlyingThings3D(args.root_dir, "TRAIN")
    visualize_settings = dict(verbose=False, percent_subsample=0.001)

for idx, (query, result) in enumerate(dataset):
    print("IDX:", idx)
    vis = O3DVisualizer()
    print("Visualizing scene sequence...")
    vis = query.scene_sequence.visualize(vis)
    # vis = query.visualize(vis, **visualize_settings)
    print("Visualizing result...")
    vis = result.visualize(vis, **visualize_settings)
    print("Running visualizer...")
    vis.run()
