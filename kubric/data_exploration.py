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



def draw_entry(entry):
    pointcloud = entry["pointcloud"]
    pointcloud_pinhole = entry["pointcloud_pinhole"]
    pose = entry["pose"]
    # projected_pointcloud = pointcloud.transform(pose)

    # Draw the pointcloud
    pc = pointcloud.to_o3d()
    pc = pc.paint_uniform_color([0, 1, 0])
    add_geometry(pc)

    # Draw the pointcloud projected to the pinhole camera
    pc_pinhole = pointcloud_pinhole.to_o3d()
    pc_pinhole = pc_pinhole.paint_uniform_color([1, 0, 0])
    add_geometry(pc_pinhole)


# for idx in range(len(sequence)):
#     entry = sequence[idx]
    
#     # pc_o3d = pointcloud_via_o3d.to_o3d()
#     # pc_o3d = pc_o3d.paint_uniform_color([0, 1, 0])
#     # add_geometry(pc_o3d)

#     break

draw_entry(sequence[10])

vis.run()