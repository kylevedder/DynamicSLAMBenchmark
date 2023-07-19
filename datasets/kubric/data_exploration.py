import argparse
from pathlib import Path
import pickle
import open3d as o3d
from datastructures import SE3, O3DVisualizer

from dataset import KubricSequenceLoader, KubricSequence, Kubric

# Get path to the root directory of the generated Kubric dataset
parser = argparse.ArgumentParser()
parser.add_argument("root_dir", type=Path)
args = parser.parse_args()

dataset = Kubric(args.root_dir)



for idx, (query, result) in enumerate(dataset):
    print("IDX:", idx)
    vis = O3DVisualizer()
    vis = query.scene_sequence.visualize(vis)
    vis = query.visualize(vis)
    vis = result.visualize(vis)
    vis.run()


