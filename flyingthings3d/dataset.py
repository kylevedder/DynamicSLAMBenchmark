from pathlib import Path
import loaders
import numpy as np

from datastructures import SE3, PointCloud
import matplotlib.pyplot as plt
import open3d as o3d
from typing import Tuple, List, Dict, Any


class FlyingThings3D:

    def __init__(self, root_dir: Path):
        root_dir = Path(root_dir)
        self.root_dir = root_dir

        left_rgb_image_paths = sorted(
            (self.root_dir / "RGB_cleanpass" / "left").glob("*.png"))
        disparity_image_paths = sorted(
            (self.root_dir / "disparity").glob("*.pfm"))
        disparity_change_image_paths = sorted(
            (self.root_dir / "disparity_change").glob("*.pfm"))
        optical_flow_image_paths = sorted(
            (self.root_dir / "optical_flow" / "forward").glob("*.pfm"))

        self.left_rgb_images = [
            loaders.read(path) for path in left_rgb_image_paths
        ]
        self.disparity_images = [
            loaders.read(path) for path in disparity_image_paths
        ]
        self.depth_images = [
            self._disparity_to_depth(disparity)
            for disparity in self.disparity_images
        ]
        self.disparity_change_images = [
            loaders.read(path) for path in disparity_change_image_paths
        ]
        self.depth_change_images = [
            self._disparity_to_depth(disparity + disparity_change) -
            self._disparity_to_depth(disparity)
            for disparity_change, disparity in zip(
                self.disparity_change_images, self.disparity_images)
        ]
        self.optical_flow_images = [
            loaders.read(path)[:, :, :2] for path in optical_flow_image_paths
        ]

        assert len(self.left_rgb_images) == len(self.disparity_images), \
            f"rgb_images and disparity_images have different lengths, {len(self.left_rgb_images)} != {len(self.disparity_images)}"
        assert len(self.left_rgb_images) == len(self.disparity_change_images), \
            f"rgb_images and disparity_change_images have different lengths, {len(self.left_rgb_images)} != {len(self.disparity_change_images)}"
        assert len(self.left_rgb_images) == len(self.optical_flow_images), \
            f"rgb_images and optical_flow_images have different lengths, {len(self.left_rgb_images)} != {len(self.optical_flow_images)}"

        self.camera_data = loaders.load_camera_matrices(root_dir /
                                                        "camera_data.txt")
        self.blender_T_camera_left_lst = [
            data["left"] for data in self.camera_data
        ]
        self.blender_T_camera_right_lst = [
            data["right"] for data in self.camera_data
        ]

        assert len(self.left_rgb_images) == len(self.blender_T_camera_left_lst), \
            f"rgb_images and camera_data_left have different lengths, {len(self.left_rgb_images)} != {len(self.blender_T_camera_left_lst)}"
        assert len(self.left_rgb_images) == len(self.blender_T_camera_right_lst), \
            f"rgb_images and camera_data_right have different lengths, {len(self.left_rgb_images)} != {len(self.blender_T_camera_right_lst)}"

    def __len__(self):
        # We only have N - 1 frames where the pose before and after is known, so we only have N - 1 samples.
        return len(self.left_rgb_images) - 1

    def _get_cam_pose(self, world_T_cam) -> SE3:
        # Blender coordinate frame is different from standard robotics
        # "right hand rule" world coordinate frame, so we compose the two transformations.
        world_T_standard = world_T_cam @ self._blender_T_standard

        translation = world_T_standard[:3, 3]
        rotation = world_T_standard[:3, :3]
        return SE3(rotation_matrix=rotation, translation=translation)

    def _o3d_intrinsics(self):
        return o3d.camera.PinholeCameraIntrinsic(width=960,
                                                 height=540,
                                                 **self.intrinsics)

    def _to_o3d_rgbd_image(self, rgb, depth_image):
        assert rgb.shape[:
                         2] == depth_image.shape[:
                                                 2], f"shape mismatch, {rgb.shape} != {depth_image.shape}"

        rgb_img = o3d.geometry.Image(rgb)
        depth_img = o3d.geometry.Image(depth_image)
        return o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=rgb_img,
            depth=depth_img,
            depth_scale=1.0,
            depth_trunc=np.inf,
            convert_rgb_to_intensity=False,
        )

    def _get_pc_flowed_pc(
            self, depth_image, image_space_flow_deltas,
            image_space_depth_change) -> Tuple[PointCloud, PointCloud]:
        assert image_space_flow_deltas.shape[:2] == depth_image.shape[:2], \
            f"shape mismatch, {image_space_flow_deltas.shape} != {depth_image.shape}"
        assert image_space_flow_deltas.shape[:2] == image_space_depth_change.shape[:2], \
            f"shape mismatch, {image_space_flow_deltas.shape} != {image_space_depth_change.shape}"

        assert image_space_flow_deltas.shape[2] == 2, \
            f"flow_image must have 2 channels, {image_space_flow_deltas.shape[2]} != 2"
        assert depth_image.ndim == 2, \
            f"depth_image must have 2 dimensions, {depth_image.ndim} != 2"
        assert image_space_depth_change.ndim == 2, \
            f"depth_change_image must have 2 dimensions, {image_space_depth_change.ndim} != 2"

        # Add third dimension to depth and depth change images.
        depth_image = depth_image[..., np.newaxis]
        image_space_depth_change = image_space_depth_change[..., np.newaxis]

        # X positions repeated for each row
        x_positions = np.tile(np.arange(image_space_flow_deltas.shape[1]),
                              (image_space_flow_deltas.shape[0], 1))
        # Y positions repeated for each column
        y_positions = np.tile(np.arange(image_space_flow_deltas.shape[0]),
                              (image_space_flow_deltas.shape[1], 1)).T

        # Stack the x and y positions into a 3D array of shape (H, W, 2)
        image_space_input_positions = np.stack([x_positions, y_positions],
                                               axis=2).astype(np.float32)

        input_pointcloud = PointCloud.from_pinhole_points_and_depth(
            image_space_input_positions.reshape(-1, 2),
            depth_image.reshape(-1, 1), self.intrinsics)

        flowed_pointcloud = PointCloud.from_pinhole_points_and_depth(
            (image_space_input_positions + image_space_flow_deltas).reshape(
                -1,
                2), (depth_image + image_space_depth_change).reshape(-1, 1),
            self.intrinsics)

        return input_pointcloud, flowed_pointcloud

    def __getitem__(self, idx):
        assert idx < len(self), f"idx out of bounds, {idx} >= {len(self)}"

        left_cam_pose_t = self._get_cam_pose(
            self.blender_T_camera_left_lst[idx])
        right_cam_pose_t = self._get_cam_pose(
            self.blender_T_camera_right_lst[idx])
        left_cam_pose_tp1 = self._get_cam_pose(
            self.blender_T_camera_left_lst[idx + 1])

        left_cam_rgb_t = self.left_rgb_images[idx]
        depth_image_t = self.depth_images[idx]

        left_pc_t, left_flowed_pc_tp1 = self._get_pc_flowed_pc(
            depth_image_t, self.optical_flow_images[idx],
            self.depth_change_images[idx])

        return {
            "left_cam_rgb_t": left_cam_rgb_t,
            "left_cam_pose_t": left_cam_pose_t,
            "right_cam_pose_t": right_cam_pose_t,
            "left_cam_pose_tp1": left_cam_pose_tp1,
            "left_pointcloud_t": left_pc_t,
            "left_pointcloud_flowed_tp1": left_flowed_pc_tp1,
        }

    def _disparity_to_depth(self, disparity_image, baseline=1.0):
        return (baseline * self.intrinsics["fx"]) / disparity_image

    @property
    def intrinsics(self):
        return {
            "fx": 1050.0,
            "fy": 1050.0,
            "cx": 479.5,
            "cy": 269.5,
        }

    @property
    def _standard_T_blender(self):
        return np.array([
            [0, 0, -1, 0],
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ])

    @property
    def _blender_T_standard(self):
        return np.linalg.inv(self._standard_T_blender)
