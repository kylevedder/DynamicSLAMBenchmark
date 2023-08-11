import open3d as o3d
from .pointcloud import PointCloud
from .se3 import SE3
from typing import Tuple, List, Dict, Union
import numpy as np


class O3DVisualizer:

    def __init__(self):
        # Create o3d visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Benchmark Visualizer")
        # Draw world coordinate frame
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
        # center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        vis.add_geometry(world_frame)
        # vis.add_geometry(center_sphere)
        # Set point size
        vis.get_render_option().point_size = 0.1

        self.vis = vis

    def add_geometry(self, geometry):
        if isinstance(geometry, list):
            for g in geometry:
                if 'to_o3d' in dir(g):
                    g = g.to_o3d()
                self.add_geometry(g)
        else:
            self.vis.add_geometry(geometry)

    def add_pointcloud(self,
                       pc: PointCloud,
                       pose: SE3 = SE3.identity(),
                       color: Union[Tuple[float, float, float], None] = None):
        pc = pc.transform(pose)
        pc = pc.to_o3d()
        if color is not None:
            pc = pc.paint_uniform_color(color)
        self.add_geometry(pc)

    def add_sphere(self, location: np.ndarray, radius: float,
                   color: Tuple[float, float, float]):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere = sphere.translate(location)
        sphere.paint_uniform_color(color)
        self.add_geometry(sphere)

    def add_pose(self, pose: SE3):
        self.add_geometry(pose.to_o3d(simple=True))

    def add_trajectory(self,
                       trajectory: List[np.ndarray],
                       color: Tuple[float, float, float],
                       radius: float = 0.05):
        for i in range(len(trajectory) - 1):
            self.add_sphere(trajectory[i], radius, color)

        # Add line set between trajectory points
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(trajectory)
        line_set.lines = o3d.utility.Vector2iVector(
            np.array([[i, i + 1] for i in range(len(trajectory) - 1)]))
        line_set.colors = o3d.utility.Vector3dVector(
            np.tile(np.array(color), (len(trajectory) - 1, 1)))
        self.add_geometry(line_set)

    def run(self):
        ctr = self.vis.get_view_control()
        # Set forward direction to be -X
        ctr.set_front([-1, 0, 0])
        # Set up direction to be +Z
        ctr.set_up([0, 0, 1])
        # Set lookat to be origin
        ctr.set_lookat([0, 0, 0])
        self.vis.run()