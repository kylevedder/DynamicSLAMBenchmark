import argparse
from pathlib import Path
import pickle
import open3d as o3d
from datastructures import SE3

from dataset import KubricSequenceLoader, KubricSequence

# Get path to the root directory of the generated Kubric dataset
parser = argparse.ArgumentParser()
parser.add_argument("root_dir", type=Path)
args = parser.parse_args()

sequence_loader = KubricSequenceLoader(args.root_dir)
sequence = sequence_loader[0]

# Create o3d visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()
# Draw world coordinate frame
world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
vis.add_geometry(world_frame)
# Set point size
vis.get_render_option().point_size = 0.1


def add_geometry(geometry):
    if isinstance(geometry, list):
        for g in geometry:
            add_geometry(g)
    else:
        vis.add_geometry(geometry)


for idx in range(len(sequence)):
    entry = sequence[idx]
    pointcloud = entry["pointcloud"]
    pose = entry["pose"]
    projected_pointcloud = pointcloud.transform(pose)

    # Draw the pointcloud
    add_geometry(pointcloud.to_o3d())

    break

vis.run()