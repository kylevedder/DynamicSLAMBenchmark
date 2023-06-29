import loaders
from pathlib import Path
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
from pointclouds import SE3, PointCloud

from dataset import FlyingThings3D

root_dir = Path("/efs/flying_things_3d_sample/")

dataset = FlyingThings3D(root_dir)

# Create o3d visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()
# Draw world coordinate frame
world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
vis.add_geometry(world_frame)


def draw_line_between_cameras(left_cam_pose: SE3, right_cam_pose: SE3):
    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector(
        np.array([left_cam_pose.translation, right_cam_pose.translation]))
    line.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
    line.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 0]]))
    return line


def add_geometry(geometry):
    if isinstance(geometry, list):
        for g in geometry:
            add_geometry(g)
    else:
        vis.add_geometry(geometry)


for idx in range(len(dataset)):
    data = dataset[idx]
    left_cam_pose: SE3 = data["left_cam_pose"]
    right_cam_pose: SE3 = data["right_cam_pose"]
    left_pointcloud: PointCloud = data["left_pointcloud"]

    add_geometry([left_cam_pose.to_o3d(), right_cam_pose.to_o3d()])
    add_geometry(draw_line_between_cameras(left_cam_pose, right_cam_pose))
    add_geometry(left_pointcloud.transform(left_cam_pose).to_o3d())

vis.run()
