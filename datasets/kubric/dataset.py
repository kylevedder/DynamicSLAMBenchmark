from scene_trajectory_benchmark.datastructures import *
from scene_trajectory_benchmark.eval import Evaluator
from pathlib import Path
import pickle
import numpy as np
from typing import Dict, Tuple, List


class KubricSequence():

    def __init__(self, data_file: Path):
        self.data = self._load_pkl(data_file)
        self.right_hand_T_blender = np.array([
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

    def _blender_to_right_hand_coordinates(self,
                                           points: np.ndarray) -> np.ndarray:
        return (self.right_hand_T_blender @ points.T).T

    def _get_rgb(self, idx) -> RGBImage:
        rgb = self.data["rgb_video"][idx]
        rgb = (rgb + 1.0) / 2.0
        return RGBImage(rgb)

    def _get_raw_pose(self, idx) -> SE3:
        position = self.data["camera"]["positions"][idx]
        quaternion = self.data["camera"]["quaternions"][idx]
        blender_pose = SE3.from_rot_w_x_y_z_translation_x_y_z(
            *quaternion, *position)
        return blender_pose.compose(
            SE3(np.linalg.inv(self.right_hand_T_blender), np.zeros(3)))

    def _get_pose(self, idx) -> SE3:
        start_pose = self._get_raw_pose(0)
        idx_pose = self._get_raw_pose(idx)
        return start_pose.inverse().compose(idx_pose)

    def _get_pointcloud(self, idx,
                        camera_projection: CameraProjection) -> PointCloud:
        depth = self.data["depth_video"][idx]
        return PointCloud.from_depth_image(depth[:, :, 0], camera_projection)

    def _get_particle_trajectories(
            self) -> Dict[ParticleID, ParticleTrajectory]:
        is_occluded_array = self.data["occluded"]
        blender_target_points_3d_array = self.data["target_points_3d"]
        object_ids = self.data["target_object_ids"]
        assert len(is_occluded_array) == len(blender_target_points_3d_array), \
            f"Number of particles does not match! {len(is_occluded_array)} != {len(blender_target_points_3d_array)}"

        particle_trajectory_dict: Dict[ParticleID, ParticleTrajectory] = {}

        for trajectory_idx, (is_occluded_lst, blender_target_points_3d,
                             object_id) in enumerate(
                                 zip(is_occluded_array,
                                     blender_target_points_3d_array,
                                     object_ids)):
            assert len(is_occluded_lst) == len(blender_target_points_3d), \
                f"Number of particles timesteps does not match! {len(is_occluded_lst)} != {len(blender_target_points_3d)}"
            camera_frame_points = self._blender_to_right_hand_coordinates(
                blender_target_points_3d)

            # Build trajectory
            particle_id = ParticleID(f"particle_{trajectory_idx:06d}")
            trajectory: Dict[Timestamp, EstimatedParticle] = {}
            for timestamp_idx, (is_occluded,
                                target_point_camera_frame) in enumerate(
                                    zip(is_occluded_lst, camera_frame_points)):
                camera_pose = self._get_pose(timestamp_idx)
                target_point_global_frame = camera_pose.transform_points(
                    target_point_camera_frame[np.newaxis, :])[0]

                trajectory[Timestamp(timestamp_idx)] = EstimatedParticle(
                    target_point_global_frame, is_occluded)

            # Add trajectory to list
            particle_trajectory_dict[particle_id] = ParticleTrajectory(
                particle_id,
                trajectory,
                cls=object_id if object_id != 0 else None)

        return particle_trajectory_dict

    def to_raw_scene_sequence(self) -> RawSceneSequence:
        percept_lookup: Dict[int, Tuple[PointCloudFrame, RGBFrame]] = {}

        for idx in range(len(self)):

            rgb_image = self._get_rgb(idx)
            pose = self._get_pose(idx)
            camera_projection = self._get_camera_projection()
            pointcloud = self._get_pointcloud(idx, camera_projection)

            pc_frame = PointCloudFrame(
                pointcloud,
                PoseInfo(sensor_to_ego=SE3.identity(), ego_to_global=pose))

            rgb_frame = RGBFrame(
                rgb_image,
                PoseInfo(sensor_to_ego=SE3.identity(), ego_to_global=pose),
                camera_projection)

            percept_lookup[idx] = (pc_frame, rgb_frame)
        return RawSceneSequence(percept_lookup)

    def to_query_scene_sequence(self) -> QuerySceneSequence:
        raw_scene_sequence = self.to_raw_scene_sequence()
        query_timesteps = raw_scene_sequence.get_percept_timesteps()
        particle_trajectories = self._get_particle_trajectories()

        query_particles: Dict[ParticleID, Tuple[WorldParticle, Timestamp]] = {
            particle_id: (trajectory[trajectory.get_first_timestamp()].point,
                          trajectory.get_first_timestamp())
            for particle_id, trajectory in particle_trajectories.items()
        }

        return QuerySceneSequence(raw_scene_sequence, query_particles,
                                  query_timesteps)

    def to_result_scene_sequence(self) -> GroundTruthParticleTrajectories:
        raw_scene_sequence = self.to_raw_scene_sequence()
        particle_trajectories = self._get_particle_trajectories()
        return GroundTruthParticleTrajectories(raw_scene_sequence, particle_trajectories)


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


class Kubric():
    """
    Wrapper for the Kubric dataset.

    It provides iterable access over all problems in the dataset.
    """

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.sequence_loader = KubricSequenceLoader(root_dir)

    def __len__(self):
        return len(self.sequence_loader)

    def __getitem__(self,
                    idx) -> Tuple[QuerySceneSequence, GroundTruthParticleTrajectories]:

        sequence = self.sequence_loader[idx]

        query_scene_sequence = sequence.to_query_scene_sequence()
        results_scene_sequence = sequence.to_result_scene_sequence()

        return query_scene_sequence, results_scene_sequence
    
    def evaluator(self) -> Evaluator:
        # Builds the evaluator object for this dataset.
        return Evaluator(self)
