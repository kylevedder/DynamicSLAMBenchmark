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
world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
# center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
vis.add_geometry(world_frame)
# vis.add_geometry(center_sphere)
# Set point size
vis.get_render_option().point_size = 0.1


def add_geometry(geometry):
    if isinstance(geometry, list):
        for g in geometry:
            add_geometry(g)
    else:
        vis.add_geometry(geometry)


raw_scene_sequence = sequence.to_raw_scene_sequence()

for timestep in raw_scene_sequence.get_percept_timesteps():
    pointcloud_frame, rgb_frame = raw_scene_sequence.get_percepts(timestep)
    global_pc = pointcloud_frame.pc.transform(pointcloud_frame.ego_to_global)
    global_pc_o3d = global_pc.to_o3d()
    global_pc_o3d = global_pc_o3d.paint_uniform_color([0, 1, 0])
    add_geometry(global_pc_o3d)


vis.run()