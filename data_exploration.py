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
elif args.dataset == "flyingthings3d":
    dataset = FlyingThings3D(args.root_dir)



for idx, (query, result) in enumerate(dataset):
    print("IDX:", idx)
    vis = O3DVisualizer()
    vis = query.scene_sequence.visualize(vis)
    # vis = query.visualize(vis)
    vis = result.visualize(vis)
    vis.run()


