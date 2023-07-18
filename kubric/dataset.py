from pathlib import Path
import pickle
from datastructures import SE3, PointCloud, ParticleFrame, CameraProjection, CameraModel, RGBImage, RawSceneSequence, PointCloudFrame, RGBFrame
import numpy as np
import scipy.interpolate as interpolate
from typing import Dict


class KubricSequence():

    def __init__(self, data_file: Path):
        self.data = self._load_pkl(data_file)
        self.world_T_camera = np.array([
            [0, 0, -1],
            [-1, 0, 0],
            [0, 1, 0],
        ])

    def _get_camera_projection(self):
        focal_length = self.data["camera"]["focal_length"] * 2
        input_size_x = self.data["metadata"]["width"]
        input_size_y = self.data["metadata"]["height"]
        sensor_width = self.data["camera"]["sensor_width"]

        fx = focal_length / sensor_width * (input_size_x / 2)
        fy = fx * input_size_x / input_size_y
        cx = input_size_x / 2
        cy = input_size_y / 2
        return CameraProjection(fx, fy, cx, cy, CameraModel.FIELD_OF_VIEW)

    def __len__(self):
        # We only have N - 1 frames where the pose before and after is known, so we only have N - 1 samples.
        return self.data["metadata"]['num_frames'] - 1

    def _load_pkl(self, pkl_file: Path):
        assert pkl_file.exists(), f"pkl_file {pkl_file} does not exist"
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
        return data

    def _image_space_to_depth(self,
                              image_space_positions,
                              depth_entries,
                              interpolation_type="nearest"):
        assert image_space_positions.shape[
            1] == 2, f"image_space_positions must have shape (N, 2), got {image_space_positions.shape}"
        assert depth_entries.shape[
            2] == 1, f"depth_image must have shape (H, W, 1), got {depth_entries.shape}"

        depth_entries = depth_entries[:, :, 0]
        xs = np.arange(depth_entries.shape[0])
        ys = np.arange(depth_entries.shape[1])

        depth_interpolator = interpolate.RegularGridInterpolator(
            (xs, ys), depth_entries, bounds_error=False)

        x_queries = image_space_positions[:, 0]
        y_queries = image_space_positions[:, 1]

        if interpolation_type == "nearest":
            x_queries = np.round(x_queries)
            y_queries = np.round(y_queries)

        depths = depth_interpolator((
            y_queries,
            x_queries,
        ))

        # add trailing dimension
        return depths[:, np.newaxis]

    def _camera_to_world_coordiantes(self, points: np.ndarray) -> np.ndarray:
        return (self.world_T_camera @ points.T).T

    def _get_rgb(self, idx):
        rgb = self.data["rgb_video"][idx]
        rgb = (rgb + 1.0) / 2.0
        return RGBImage(rgb)

    def _get_pose(self, idx):
        position = self.data["camera"]["positions"][idx]
        quaternion = self.data["camera"]["quaternions"][idx]
        blender_pose = SE3.from_rot_w_x_y_z_translation_x_y_z(
            *quaternion, *position)
        return blender_pose.compose(
            SE3(np.linalg.inv(self.world_T_camera), np.zeros(3)))

    def _get_pointcloud(self, idx, camera_projection: CameraProjection):
        depth = self.data["depth_video"][idx]
        return PointCloud.from_depth_image(depth[:, :, 0], camera_projection)

    def _get_particle_frame(self, idx) -> ParticleFrame:
        is_occluded = self.data["occluded"][:, idx]
        blender_target_points_3d = self.data["target_points_3d"][:, idx]
        world_target_points_3d = self._camera_to_world_coordiantes(
            blender_target_points_3d)
        return ParticleFrame(world_target_points_3d, is_occluded,
                             np.ones_like(is_occluded, dtype=bool))

    def to_raw_scene_sequence(self) -> RawSceneSequence:
        pointcloud_lookup: Dict[int, PointCloudFrame] = {}
        rgb_lookup: Dict[int, RGBFrame] = {}

        start_pose = self._get_pose(0)
        for idx in range(len(self)):

            rgb_image = self._get_rgb(idx)
            pose = self._get_pose(idx)
            camera_projection = self._get_camera_projection()
            pointcloud = self._get_pointcloud(idx, camera_projection)

            pointcloud_lookup[idx] = PointCloudFrame(
                pointcloud,
                start_pose.inverse().compose(pose))
            rgb_lookup[idx] = RGBFrame(rgb_image, SE3.identity(),
                                       camera_projection)
        return RawSceneSequence(pointcloud_lookup, rgb_lookup)


class KubricSequenceLoader():

    def __init__(self, root_dir: Path) -> None:
        self.files = sorted(root_dir.glob("*.pkl"))
        assert len(
            self.files
        ) > 0, f"root_dir {root_dir} does not contain any .pkl files"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> KubricSequence:
        return KubricSequence(self.files[idx])
