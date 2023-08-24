from scene_trajectory_benchmark.datastructures import *
from .loaders import *
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from typing import Tuple, List, Dict, Any


class FlyingThingsSequence:
    """
    Assumes the following example directory structure:

    root_dir
    ├── camera_data
    │   └── TRAIN
    │       └── A
    │           └── 0001
    │               └── camera_data.txt
    ├── disparity
    │   └── TRAIN
    │       └── A
    │           └── 0001
    │               ├── 0006.pfm
    │               ├── 0007.pfm
    │               └── 0008.pfm
    ├── disparity_change
    │   └── TRAIN
    │       └── A
    │           └── 0001
    │               ├── 0006.pfm
    │               ├── 0007.pfm
    │               └── 0008.pfm
    ├── object_index
    │   └── TRAIN
    │       └── A
    │           └── 0001
    │               ├── 0006.pfm
    │               ├── 0007.pfm
    │               └── 0008.pfm
    ├── optical_flow
    │   └── TRAIN
    │       └── A
    │           └── 0001
    │               ├── backward
    │               │   ├── 0006.pfm
    │               │   ├── 0007.pfm
    │               │   └── 0008.pfm
    │               └── forward
    │                   ├── 0006.pfm
    │                   ├── 0007.pfm
    │                   └── 0008.pfm
    └── RGB_cleanpass
        └── TRAIN
            └── A
                └── 0001
                    ├── left
                    │   ├── 0006.png
                    │   ├── 0007.png
                    │   └── 0008.png
                    └── right
                        ├── 0006.png
                        ├── 0007.png
                        └── 0008.png
    """

    def __init__(self, root_dir: Path, split_subdir_names: str):
        root_dir = Path(root_dir)
        self.root_dir = root_dir

        left_rgb_image_paths = sorted(
            (self.root_dir / "RGB_cleanpass" / split_subdir_names /
             "left").glob("*.png"))
        disparity_image_paths = sorted(
            (self.root_dir / "disparity" / split_subdir_names).glob("*.pfm"))
        disparity_change_image_paths = sorted(
            (self.root_dir / "disparity_change" /
             split_subdir_names).glob("*.pfm"))
        optical_flow_image_paths = sorted(
            (self.root_dir / "optical_flow" / split_subdir_names /
             "forward").glob("*.pfm"))
        object_index_image_paths = sorted((self.root_dir / "object_index" /
                                           split_subdir_names).glob("*.pfm"))

        self.left_rgb_images = [
            f3d_read(path) for path in left_rgb_image_paths
        ]
        self.disparity_images = [
            f3d_read(path) for path in disparity_image_paths
        ]
        self.depth_images = [
            self._disparity_to_depth(disparity)
            for disparity in self.disparity_images
        ]
        self.disparity_change_images = [
            f3d_read(path) for path in disparity_change_image_paths
        ]
        self.depth_change_images = [
            self._disparity_to_depth(disparity + disparity_change) -
            self._disparity_to_depth(disparity)
            for disparity_change, disparity in zip(
                self.disparity_change_images, self.disparity_images)
        ]
        self.optical_flow_images = [
            f3d_read(path)[:, :, :2] for path in optical_flow_image_paths
        ]
        self.object_index_images = [
            f3d_read(path) for path in object_index_image_paths
        ]

        assert len(self.left_rgb_images) == len(self.disparity_images), \
            f"rgb_images and disparity_images have different lengths, {len(self.left_rgb_images)} != {len(self.disparity_images)}"
        assert len(self.left_rgb_images) == len(self.disparity_change_images), \
            f"rgb_images and disparity_change_images have different lengths, {len(self.left_rgb_images)} != {len(self.disparity_change_images)}"
        assert len(self.left_rgb_images) == len(self.optical_flow_images), \
            f"rgb_images and optical_flow_images have different lengths, {len(self.left_rgb_images)} != {len(self.optical_flow_images)}"
        assert len(self.left_rgb_images) == len(self.object_index_images), \
            f"rgb_images and object_index_images have different lengths, {len(self.left_rgb_images)} != {len(self.object_index_images)}"

        self.camera_data = f3d_load_camera_matrices(
            root_dir / "camera_data" / split_subdir_names / "camera_data.txt")
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

    def _get_raw_left_pose(self, idx):
        return self._get_cam_pose(self.blender_T_camera_left_lst[idx])

    def _get_left_pose(self, idx):
        start_pose = self._get_raw_left_pose(0)
        idx_pose = self._get_raw_left_pose(idx)
        return start_pose.inverse().compose(idx_pose)

    def _get_raw_right_pose(self, idx):
        return self._get_cam_pose(self.blender_T_camera_right_lst[idx])

    def _get_right_pose(self, idx):
        start_pose = self._get_raw_right_pose(0)
        idx_pose = self._get_raw_right_pose(idx)
        return start_pose.inverse().compose(idx_pose)

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

        camera_projection = CameraProjection(**self.intrinsics,
                                             camera_model=CameraModel.PINHOLE)

        input_pointcloud = PointCloud.from_points_and_depth(
            image_space_input_positions.reshape(-1, 2),
            depth_image.reshape(-1, 1), camera_projection)

        flowed_pointcloud = PointCloud.from_points_and_depth(
            (image_space_input_positions + image_space_flow_deltas).reshape(
                -1,
                2), (depth_image + image_space_depth_change).reshape(-1, 1),
            camera_projection)

        return input_pointcloud, flowed_pointcloud

    def __getitem__(self, idx):
        assert idx < len(self), f"idx out of bounds, {idx} >= {len(self)}"

        left_cam_pose_t = self._get_left_pose(idx)
        right_cam_pose_t = self._get_right_pose(idx)
        left_cam_pose_tp1 = self._get_left_pose(idx + 1)

        # object_index_image_t = self.object_index_images[idx]
        # breakpoint()

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

    def _particle_idx_to_particle_id(self, particle_idx: int) -> ParticleID:
        return ParticleID(f"particle_{particle_idx:08d}")

    def to_raw_scene_sequence(self) -> RawSceneSequence:

        # Accumulate the data from each frame into percept_lookup.
        percept_lookup: Dict[Timestamp, Tuple[PointCloudFrame, RGBFrame]] = {}
        camera_projection = CameraProjection(**self.intrinsics,
                                             camera_model=CameraModel.PINHOLE)

        for entry_idx in range(len(self)):
            entry_dict = self[entry_idx]
            rgb = RGBImage(entry_dict["left_cam_rgb_t"].astype(np.float32) /
                           255.0)
            global_pose: SE3 = entry_dict["left_cam_pose_t"]
            pointcloud: PointCloud = entry_dict["left_pointcloud_t"]

            pose_info = PoseInfo(sensor_to_ego=SE3.identity(),
                                 ego_to_global=global_pose)

            pc_frame = PointCloudFrame(pointcloud, pose_info)
            rgb_frame = RGBFrame(rgb, pose_info, camera_projection)

            percept_lookup[entry_idx] = (pc_frame, rgb_frame)

        return RawSceneSequence(percept_lookup)

    def to_query_scene_sequence(self,
                                timestamp: Timestamp) -> QuerySceneSequence:
        assert timestamp < len(
            self), f"idx out of bounds, {timestamp} >= {len(self)}"
        raw_scene_sequence = self.to_raw_scene_sequence()
        entry_dict = self[timestamp]
        pointcloud_t: PointCloud = entry_dict["left_pointcloud_t"]

        query_particles = {
            self._particle_idx_to_particle_id(particle_idx):
            (particle_position, timestamp)
            for particle_idx, particle_position in enumerate(
                pointcloud_t.points)
        }
        query_timestamps = [timestamp, timestamp + 1]
        return QuerySceneSequence(raw_scene_sequence, query_particles,
                                  query_timestamps)

    def to_result_scene_sequence(self,
                                 timestamp: Timestamp) -> GroundTruthParticleTrajectories:
        assert timestamp < len(
            self), f"idx out of bounds, {timestamp} >= {len(self)}"
        raw_scene_sequence = self.to_raw_scene_sequence()
        entry_dict = self[timestamp]
        pointcloud_t: PointCloud = entry_dict["left_pointcloud_t"]
        left_cam_pose_t: SE3 = entry_dict["left_cam_pose_t"]
        pointcloud_t = pointcloud_t.transform(left_cam_pose_t)
        pointcloud_tp1: PointCloud = entry_dict["left_pointcloud_flowed_tp1"]
        left_cam_pose_tp1: SE3 = entry_dict["left_cam_pose_tp1"]
        pointcloud_tp1 = pointcloud_tp1.transform(left_cam_pose_tp1)

        particle_trajectories: Dict[ParticleID, ParticleTrajectory] = {}

        for particle_idx, (point_t, point_tp1) in enumerate(
                zip(pointcloud_t.points, pointcloud_tp1.points)):
            particle_id = self._particle_idx_to_particle_id(particle_idx)
            particle_lookup = {
                timestamp: EstimatedParticle(point_t, False),
                timestamp + 1: EstimatedParticle(point_tp1, False)
            }
            cls = None
            particle_trajectories[particle_id] = ParticleTrajectory(
                particle_id, particle_lookup, cls)

        return GroundTruthParticleTrajectories(raw_scene_sequence, particle_trajectories)


class FlyingThings3D():

    def __init__(self, root_dir: Path, split: str = "TRAIN"):
        root_dir = Path(root_dir)
        self.root_dir = root_dir
        self.split = split.upper()

        # Use the camera_data folder as the source of truth for sequence names and lengths

        camera_data_dir = self.root_dir / "camera_data" / self.split

        sequence_dirs = sorted(camera_data_dir.glob("*/*"))
        self.split_subdir_names = [
            f"{e.parent.name}/{e.name}" for e in sequence_dirs
        ]
        self.subdir_lengths = [
            len(f3d_load_camera_matrices(e / "camera_data.txt")) - 1
            for e in sequence_dirs
        ]

        assert all([l > 0 for l in self.subdir_lengths]), \
            f"all sequence lengths must be greater than 0, {self.subdir_lengths}"

    def __len__(self) -> int:
        return sum(self.subdir_lengths)

    def __getitem__(self,
                    idx) -> Tuple[QuerySceneSequence, GroundTruthParticleTrajectories]:
        assert idx < len(self), f"idx out of bounds, {idx} >= {len(self)}"

        # Find the sequence that contains the idx
        sequence_idx = 0
        while idx >= self.subdir_lengths[sequence_idx]:
            idx -= self.subdir_lengths[sequence_idx]
            sequence_idx += 1

        sequence = FlyingThingsSequence(
            self.root_dir,
            self.split + "/" + self.split_subdir_names[sequence_idx])
        return sequence.to_query_scene_sequence(
            idx), sequence.to_result_scene_sequence(idx)
