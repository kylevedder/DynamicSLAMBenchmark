from pathlib import Path
import loaders
import numpy as np

from pointclouds import SE3, PointCloud
from scipy.spatial.transform import Rotation as R
import open3d as o3d


class FlyingThings3D:

    def __init__(self, root_dir: Path):
        root_dir = Path(root_dir)
        self.root_dir = root_dir

        rgb_image_paths = sorted(
            (self.root_dir / "RGB_cleanpass" / "left").glob("*.png"))
        disparity_image_paths = sorted(
            (self.root_dir / "disparity").glob("*.pfm"))
        disparity_change_image_paths = sorted(
            (self.root_dir / "disparity_change").glob("*.pfm"))
        optical_flow_image_paths = sorted(
            (self.root_dir / "optical_flow" / "forward").glob("*.pfm"))

        self.rgb_images = [loaders.read(path) for path in rgb_image_paths]
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
        self.optical_flow_images = [
            loaders.read(path)[:, :, :2] for path in optical_flow_image_paths
        ]

        assert len(self.rgb_images) == len(self.disparity_images), \
            f"rgb_images and disparity_images have different lengths, {len(self.rgb_images)} != {len(self.disparity_images)}"
        assert len(self.rgb_images) == len(self.disparity_change_images), \
            f"rgb_images and disparity_change_images have different lengths, {len(self.rgb_images)} != {len(self.disparity_change_images)}"
        assert len(self.rgb_images) == len(self.optical_flow_images), \
            f"rgb_images and optical_flow_images have different lengths, {len(self.rgb_images)} != {len(self.optical_flow_images)}"

        self.camera_data = loaders.load_camera_matrices(root_dir /
                                                        "camera_data.txt")
        self.blender_T_camera_left_lst = [
            data["left"] for data in self.camera_data
        ]
        self.blender_T_camera_right_lst = [
            data["right"] for data in self.camera_data
        ]

        assert len(self.rgb_images) == len(self.blender_T_camera_left_lst), \
            f"rgb_images and camera_data_left have different lengths, {len(self.rgb_images)} != {len(self.blender_T_camera_left_lst)}"
        assert len(self.rgb_images) == len(self.blender_T_camera_right_lst), \
            f"rgb_images and camera_data_right have different lengths, {len(self.rgb_images)} != {len(self.blender_T_camera_right_lst)}"

    def __len__(self):
        return len(self.rgb_images)

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

    def _get_pointcloud(self, rgb, depth_image) -> PointCloud:
        o3d_rgbd = self._to_o3d_rgbd_image(rgb, depth_image)
        o3d_intrinsics = self._o3d_intrinsics()
        # By default Open3D projects everything along Z axis, so this converts to X axis,
        # robotics standard right hand rule.
        standard_frame_extrinsics = np.array([
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
        ])
        o3d_pointcloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            o3d_rgbd, o3d_intrinsics, standard_frame_extrinsics)
        points = np.asarray(o3d_pointcloud.points)
        colors = np.asarray(o3d_pointcloud.colors)
        return PointCloud(points)

    def __getitem__(self, idx):
        assert idx < len(self), f"idx out of bounds, {idx} >= {len(self)}"

        blender_T_cam_left = self.blender_T_camera_left_lst[idx]
        blender_T_cam_right = self.blender_T_camera_right_lst[idx]

        left_cam_pose = self._get_cam_pose(blender_T_cam_left)
        right_cam_pose = self._get_cam_pose(blender_T_cam_right)

        rgb = self.rgb_images[idx]
        depth_image = self.depth_images[idx]

        left_pointcloud = self._get_pointcloud(rgb, depth_image)

        return {
            "left_cam_pose": left_cam_pose,
            "right_cam_pose": right_cam_pose,
            "left_pointcloud": left_pointcloud,
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
