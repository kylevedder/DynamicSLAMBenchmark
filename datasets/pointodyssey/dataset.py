from scene_trajectory_benchmark.datastructures import *
from pathlib import Path
import pickle
import numpy as np
from typing import Dict, Tuple, List
import cv2
import json
# Import scipy rotation as R
from scipy.spatial.transform import Rotation as R


class PointOdysseySequence():
    def __init__(self,
                 sequence_folder: Path,
                 min_idx: int = 0,
                 max_idx: int = -1,
                 max_particles: int = 100) -> None:
        self.annotations = self._load_annotations(
            sequence_folder / "annotations.npz", min_idx, max_idx,
            max_particles)
        self.scene_info = self._load_scene_info(sequence_folder /
                                                "scene_info.json")

        self.rgbs = self._load_rgbs(sequence_folder / "rgbs", min_idx, max_idx)
        self.pointclouds = self._load_pointclouds(sequence_folder / "depths",
                                                  min_idx, max_idx)
        self.masks = self._load_masks(sequence_folder / "masks", min_idx,
                                      max_idx)

        # Should have the same number of frames for each modality
        assert len(self.rgbs) == len(self.pointclouds) == len(self.masks), \
            f"Number of frames does not match! {len(self.rgbs)} != {len(self.pointclouds)} != {len(self.masks)}"
        assert len(self.rgbs) > 0, f"Number of frames is zero!"

        self.right_hand_T_blender = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, 1],
        ])

        #

        # np.array([
        #     [0, 0, 1],
        #     [-1, 0, 0],
        #     [0, -1, 0],
        # ])
        # @  np.array([
        #     [1, 0, 0],
        #     [0, -1, 0],
        #     [0, 0, 1],
        # ])

        print(self.right_hand_T_blender)

        # np.array([
        #     [0, 0, -1],
        #     [1, 0, 0],
        #     [0, 1, 0],
        # ])

    def _blender_to_right_hand_coordinates(self,
                                           points: np.ndarray) -> np.ndarray:
        assert points.ndim == 2, f"points must be 2D, but is {points.ndim}D"
        assert points.shape[1] == 3, \
            f"points must have shape (N, 3), but has shape {points.shape}"
        return (self.right_hand_T_blender @ points.T).T

    def _load_scene_info(self, scene_info_file: Path) -> Dict:
        assert scene_info_file.exists(), \
            f"scene_info_file {scene_info_file} does not exist"
        # Load json file
        with open(scene_info_file, "r") as f:
            data = json.load(f)
        return data

    def _load_annotations(self, annotations_file: Path, min_idx: int,
                          max_idx: int, max_particles: int) -> Dict:
        assert annotations_file.exists(), \
            f"annotations_file {annotations_file} does not exist"
        # Load NPZ file
        data = dict(np.load(annotations_file, allow_pickle=True))
        if data["trajs_3d"].shape == ():
            data["trajs_3d"] = np.zeros((*data["trajs_2d"].shape[:2], 3))

        assert data["trajs_3d"].shape != (), \
            f"annotations_file {annotations_file} does not contain any traj_3d data"

        # Filter data
        data["trajs_3d"] = data["trajs_3d"][min_idx:max_idx, :max_particles]
        data["trajs_2d"] = data["trajs_2d"][min_idx:max_idx, :max_particles]
        data["visibilities"] = data["visibilities"][
            min_idx:max_idx, :max_particles]
        data["intrinsics"] = data["intrinsics"][min_idx:max_idx]
        data["extrinsics"] = data["extrinsics"][min_idx:max_idx]
        return data

    def _load_image(self, image_path: Path) -> np.ndarray:
        assert image_path.is_file(), f"image_path {image_path} does not exist"
        return cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    def _load_pointclouds(self, depths_folder: Path, min_idx: int,
                          max_idx: int) -> List[PointCloud]:
        depth_files = sorted(depths_folder.glob("*.png"))[min_idx:max_idx]
        assert len(
            depth_files
        ) > 0, f"depths_folder {depths_folder} does not contain any depth files"

        camera_projection = self._get_camera_projection()

        def _cleanup_depth_image(depth_image):
            depth_image = depth_image.astype(np.float32)
            # Remove noisy depth image returns
            depth_image[depth_image > 20000] = np.nan
            return depth_image

        depth_images = [
            _cleanup_depth_image(self._load_image(depth_file))
            for depth_file in depth_files
        ]

        min_depth_image_val = np.nanmin(
            [np.nanmin(depth_image) for depth_image in depth_images])
        max_depth_image_val = np.nanmax(
            [np.nanmax(depth_image) for depth_image in depth_images])
        print(f"min_depth_image_val: {min_depth_image_val}")
        print(f"max_depth_image_val: {max_depth_image_val}")
        pointclouds = [
            PointCloud.from_depth_image(depth_image / 16, camera_projection)
            for depth_image in depth_images
        ]
        return pointclouds

    def _load_rgbs(self, rgbs_folder: Path, min_idx: int,
                   max_idx: int) -> List[RGBImage]:
        rgb_files = sorted(rgbs_folder.glob("*.jpg"))[min_idx:max_idx]
        assert len(
            rgb_files
        ) > 0, f"rgbs_folder {rgbs_folder} does not contain any rgb files"
        return [
            RGBImage(self._load_image(rgb_file).astype(np.float32) / 255.0)
            for rgb_file in rgb_files
        ]

    def _load_masks(self, masks_folder: Path, min_idx: int,
                    max_idx: int) -> List[np.ndarray]:
        mask_files = sorted(masks_folder.glob("*.png"))[min_idx:max_idx]
        assert len(
            mask_files
        ) > 0, f"masks_folder {masks_folder} does not contain any mask files"
        return [self._load_image(mask_file) for mask_file in mask_files]

    def _get_camera_projection(self):
        intrinsics_matrix = self.annotations['intrinsics'][0]
        fx = intrinsics_matrix[0, 0]
        fy = intrinsics_matrix[1, 1]
        cx = intrinsics_matrix[0, 2]
        cy = intrinsics_matrix[1, 2]
        return CameraProjection(fx, fy, cx, cy, CameraModel.PINHOLE)

    def __len__(self):
        return len(self.rgbs)

    #####################################

    def _get_rgb(self, idx) -> RGBImage:
        return self.rgbs[idx]

    def _get_raw_pose(self, idx) -> SE3:
        raw_matrix = self.annotations['extrinsics'][idx]
        rotation_matrix = self._blender_to_right_hand_coordinates(
            raw_matrix[:3, :3])
        translation_vector = np.expand_dims(raw_matrix[:3, 3], axis=0)
        translation_vector = self._blender_to_right_hand_coordinates(
            translation_vector)[0]
        return SE3(rotation_matrix, translation_vector)

    def _get_pose(self, idx) -> SE3:
        start_pose = self._get_raw_pose(0)
        idx_pose = self._get_raw_pose(idx)
        return start_pose.inverse().compose(idx_pose)

    def _get_pointcloud(self, idx) -> PointCloud:
        return self.pointclouds[idx]

    def _process_particle_trajectory(self, trajectory_idx, is_occluded_lst,
                                     points_3d):
        assert len(is_occluded_lst) == len(points_3d), \
            f"Number of particles timesteps does not match! {len(is_occluded_lst)} != {len(points_3d)}"
        camera_frame_points = self._blender_to_right_hand_coordinates(
            points_3d)

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
        return ParticleTrajectory(particle_id, trajectory, cls=particle_id)

    def _get_particle_trajectories(
            self) -> Dict[ParticleID, ParticleTrajectory]:

        trajectories = self.annotations["trajs_3d"].transpose(1, 0, 2)
        visibilities = self.annotations["visibilities"].transpose(1, 0)
        # Filter out any initially occluded particles
        is_visible = visibilities[:, 0] != 0
        trajectories = trajectories[is_visible]
        visibilities = visibilities[is_visible]

        particle_trajectory_dict: Dict[ParticleID, ParticleTrajectory] = {}

        for trajectory_idx, (is_occluded_lst,
                             blender_target_points_3d) in enumerate(
                                 zip(visibilities, trajectories)):
            particle_trajectory = self._process_particle_trajectory(
                trajectory_idx, is_occluded_lst, blender_target_points_3d)
            particle_trajectory_dict[
                particle_trajectory.id] = particle_trajectory

        return particle_trajectory_dict

    def to_raw_scene_sequence(self) -> RawSceneSequence:
        percept_lookup: Dict[int, Tuple[PointCloudFrame, RGBFrame]] = {}

        for idx in range(len(self)):

            rgb_image = self._get_rgb(idx)
            pose = self._get_pose(idx)
            camera_projection = self._get_camera_projection()
            pointcloud = self._get_pointcloud(idx)

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
        print("To raw scene sequence...")
        raw_scene_sequence = self.to_raw_scene_sequence()
        print("To percept timesteps...")
        query_timesteps = raw_scene_sequence.get_percept_timesteps()
        print("To particle trajectories...")
        particle_trajectories = self._get_particle_trajectories()

        print("To query particles...")

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
        return GroundTruthParticleTrajectories(raw_scene_sequence,
                                               particle_trajectories)


class PointOdysseySequenceLoader():
    def __init__(self, root_dir: Path, max_sequence_length: int = 50) -> None:
        self.max_sequence_length = max_sequence_length
        self.sequence_folders = sorted([
            e for e in root_dir.glob("*")
            if e.is_dir() and e.name != "character2"
        ])
        assert len(
            self.sequence_folders
        ) > 0, f"root_dir {root_dir} does not contain any sequence folders"

    def __len__(self):
        return len(self.sequence_folders)

    def __getitem__(self, idx) -> PointOdysseySequence:
        return PointOdysseySequence(self.sequence_folders[idx],
                                    max_idx=self.max_sequence_length)


class PointOdyssey():
    """
    Wrapper for the PointOdyssey dataset.

    It provides iterable access over all problems in the dataset.
    """
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.sequence_loader = PointOdysseySequenceLoader(root_dir, )

    def __len__(self):
        return len(self.sequence_loader)

    def __getitem__(
            self,
            idx) -> Tuple[QuerySceneSequence, GroundTruthParticleTrajectories]:

        sequence = self.sequence_loader[idx]
        print("To query scene sequence...")
        query_scene_sequence = sequence.to_query_scene_sequence()
        print("To result scene sequence...")
        results_scene_sequence = sequence.to_result_scene_sequence()

        return query_scene_sequence, results_scene_sequence
