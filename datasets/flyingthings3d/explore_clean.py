import loaders
from pathlib import Path
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
from datastructures import SE3, PointCloud

from dataset import FlyingThingsSequence

root_dir = Path("/efs/flying_things_3d_sample/")

dataset = FlyingThingsSequence(root_dir)

# Create o3d visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()
# Draw world coordinate frame
world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
vis.add_geometry(world_frame)
# Set point size
vis.get_render_option().point_size = 0.1
# Set background color gray
# vis.get_render_option().background_color = np.array([0.5, 0.5, 0.5])


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


def make_lineset_from_flowed_pointcloud(pointcloud: PointCloud,
                                        flowed_pointcloud: PointCloud):
    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector(
        np.vstack((pointcloud.points, flowed_pointcloud.points)))
    line.lines = o3d.utility.Vector2iVector(
        np.array([[idx, idx + len(pointcloud.points)]
                  for idx in range(len(pointcloud.points))]))
    return line


color_scale = np.linspace(0, 1, len(dataset) + 1)[1:][::-1]
print("COLOR SCALE:", color_scale)

for idx in range(len(dataset)):
    data = dataset[idx]

    left_cam_pose_t: SE3 = data["left_cam_pose_t"]
    right_cam_pose_t: SE3 = data["right_cam_pose_t"]
    left_cam_pose_tp1: SE3 = data["left_cam_pose_tp1"]
    left_pointcloud_t: PointCloud = data["left_pointcloud_t"]
    left_pointcloud_flowed_tp1: PointCloud = data["left_pointcloud_flowed_tp1"]

    add_geometry([left_cam_pose_t.to_o3d(), right_cam_pose_t.to_o3d()])
    add_geometry(draw_line_between_cameras(left_cam_pose_t, right_cam_pose_t))

    mask = left_pointcloud_t.within_region_mask(-250, 250, -250, 250, -250,
                                                250)
    left_pointcloud_t = left_pointcloud_t.mask_points(mask)
    left_pointcloud_flowed_tp1 = left_pointcloud_flowed_tp1.mask_points(mask)

    # Transform pointclouds to world frame
    # left_pointcloud_t = left_pointcloud_t.transform(left_cam_pose_t)
    # left_pointcloud_flowed_tp1 = left_pointcloud_flowed_tp1.transform(
    #     left_cam_pose_tp1)

    left_pc_o3d = left_pointcloud_t.to_o3d()
    left_pc_o3d.paint_uniform_color([color_scale[idx], 0, 0])
    add_geometry(left_pc_o3d)

    left_pc_flowed_o3d = left_pointcloud_flowed_tp1.to_o3d()
    left_pc_flowed_o3d.paint_uniform_color([0, color_scale[idx], 0])
    add_geometry(left_pc_flowed_o3d)

    # break

    # flow_lineset = make_lineset_from_flowed_pointcloud(left_pointcloud,
    #                                                    left_pointcloud_flowed)
    # flow_lineset.colors = o3d.utility.Vector3dVector(
    #     np.array([[0, color_scale[idx], 0]] * len(left_pointcloud.points)))
    # add_geometry(flow_lineset)

vis.run()
