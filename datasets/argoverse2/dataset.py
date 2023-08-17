from scene_trajectory_benchmark.datastructures import *
from pathlib import Path
from .loader_utils import load_pickle, save_pickle

from typing import Tuple, Dict, List
import multiprocessing
import time
import numpy as np

from .argoverse_supervised_scene_flow import ArgoverseSupervisedSceneFlowSequenceLoader, ArgoverseSupervisedSceneFlowSequence


def _build_result_entry(
        point_idx: int, source_point: np.ndarray, target_point: np.ndarray,
        pc_class_id: int, subsequence_src_index: int,
        subsequence_tgt_index: int) -> Tuple[int, ParticleTrajectory]:
    particle_trajectory = ParticleTrajectory(
        point_idx, {
            subsequence_src_index: EstimatedParticle(source_point, False),
            subsequence_tgt_index: EstimatedParticle(target_point, False)
        }, ArgoverseSupervisedSceneFlowSequence.get_class_str(pc_class_id))
    return point_idx, particle_trajectory


class Argoverse2SceneFlow():
    """
    Wrapper for the Argoverse 2 dataset.

    It provides iterable access over all problems in the dataset.
    """

    def __init__(
        self,
        root_dir: Path,
        subsequence_length: int = 2,
        with_ground: bool = True,
        cache_path: Path = Path("/tmp/")) -> None:
        self.root_dir = root_dir
        self.sequence_loader = ArgoverseSupervisedSceneFlowSequenceLoader(
            root_dir)
        self.subsequence_length = subsequence_length
        self.cache_path = cache_path
        if with_ground:
            self.ego_pc_key = "ego_pc_with_ground"
            self.ego_pc_flowed_key = "ego_flowed_pc_with_ground"
            self.relative_pc_key = "relative_pc_with_ground"
            self.relative_pc_flowed_key = "relative_flowed_pc_with_ground"
            self.pc_classes_key = "pc_classes_with_ground"
        else:
            self.ego_pc_key = "ego_pc"
            self.ego_pc_flowed_key = "ego_flowed_pc"
            self.relative_pc_key = "relative_pc"
            self.relative_pc_flowed_key = "relative_flowed_pc"
            self.pc_classes_key = "pc_classes"

        self.dataset_to_sequence_subsequence_idx = self._load_dataset_to_sequence_subsequence_idx(
        )

    def _load_dataset_to_sequence_subsequence_idx(self):
        cache_file = self.cache_path / f"dataset_to_sequence_subsequence_idx_cache_len_{self.subsequence_length}.pkl"
        if cache_file.exists():
            return load_pickle(cache_file)

        print("Building dataset index...")
        # Build map from dataset index to sequence and subsequence index.
        dataset_to_sequence_subsequence_idx = []
        for sequence_idx, sequence in enumerate(self.sequence_loader):
            for subsequence_start_idx in range(
                    len(sequence) - self.subsequence_length + 1):
                dataset_to_sequence_subsequence_idx.append(
                    (sequence_idx, subsequence_start_idx))

        print(
            f"Loaded {len(dataset_to_sequence_subsequence_idx)} subsequence pairs. Saving it to {cache_file}"
        )
        save_pickle(cache_file, dataset_to_sequence_subsequence_idx)
        return dataset_to_sequence_subsequence_idx

    def __len__(self):
        return len(self.dataset_to_sequence_subsequence_idx)

    def _make_scene_sequence(
            self, subsequence_frames: List[Dict]) -> RawSceneSequence:
        # Build percept lookup. This stores the percepts for the entire sequence, with the
        # global frame being zero'd at the target frame.
        percept_lookup: Dict[Timestamp, Tuple[PointCloudFrame, RGBFrame]] = {}
        for dataset_idx, entry in enumerate(subsequence_frames):
            pc: PointCloud = entry[self.ego_pc_key]
            lidar_to_ego = SE3.identity()
            ego_to_world: SE3 = entry["relative_pose"]
            point_cloud_frame = PointCloudFrame(
                pc, PoseInfo(lidar_to_ego, ego_to_world))

            rgb_to_ego: SE3 = entry["rgb_camera_ego_pose"]
            rgb_camera_projection: CameraProjection = entry[
                "rgb_camera_projection"]
            rgb_frame = RGBFrame(entry["rgb"],
                                 PoseInfo(rgb_to_ego, ego_to_world),
                                 rgb_camera_projection)
            percept_lookup[dataset_idx] = (point_cloud_frame, rgb_frame)

        return RawSceneSequence(percept_lookup)

    def _make_query_scene_sequence(
            self, scene_sequence: RawSceneSequence,
            subsequence_frames: List[Dict], subsequence_src_index: int,
            subsequence_tgt_index: int) -> QuerySceneSequence:
        # Build query scene sequence. This requires enumerating all points in the source frame.
        query_timestamps: List[Timestamp] = [
            subsequence_src_index, subsequence_tgt_index
        ]
        source_entry = subsequence_frames[subsequence_src_index]

        pc_points_array = source_entry[self.relative_pc_key].points
        timestamp_array = np.array([subsequence_src_index] *
                                   len(pc_points_array))

        query_particles = EfficientQueryParticleLookup(len(pc_points_array))

        query_particles[np.arange(len(pc_points_array))] = (pc_points_array,
                                                            timestamp_array)

        return QuerySceneSequence(scene_sequence, query_particles,
                                  query_timestamps)

    def _make_results_scene_sequence(
            self, scene_sequence: RawSceneSequence,
            subsequence_frames: List[Dict], subsequence_src_index: int,
            subsequence_tgt_index: int) -> ResultsSceneSequence:
        # Build query scene sequence. This requires enumerating all points in
        # the source frame and the associated flowed points.

        metadata_setup_start = time.time()

        source_entry = subsequence_frames[subsequence_src_index]
        source_pc = source_entry[self.relative_pc_key].points
        target_pc = source_entry[self.relative_pc_flowed_key].points
        pc_class_ids = source_entry[self.pc_classes_key]
        assert len(source_pc) == len(
            target_pc), "Source and target point clouds must be the same size."
        assert len(source_pc) == len(
            pc_class_ids
        ), f"Source point cloud and class ids must be the same size. Instead got {len(source_pc)} and {len(pc_class_ids)}."

        metadata_setup_end = time.time()

        particle_trajectories: Dict[ParticleID, ParticleTrajectory] = {}
        particle_trajectories = EfficientParticleTrajectoriesLookup(
            len(source_pc), 2)

        points = np.stack([source_pc, target_pc], axis=1)
        timestamps = np.array([subsequence_src_index, subsequence_tgt_index])
        # Stack the false false array len(source_pc) times.
        is_occluded = np.tile([False, False], (len(source_pc), 1))

        particle_trajectories[np.arange(len(source_pc))] = (points, timestamps,
                                                            is_occluded,
                                                            pc_class_ids)
        # for point_idx, (source_point, target_point, pc_class_id) in enumerate(
        #         zip(source_pc, target_pc, pc_class_ids)):
        #     key, value = _build_result_entry(
        #         point_idx, source_point, target_point, pc_class_id,
        #         subsequence_src_index, subsequence_tgt_index)
        #     particle_trajectories[key] = value

        dict_build_end = time.time()

        result = ResultsSceneSequence(scene_sequence, particle_trajectories)

        result_build_end = time.time()

        print("\tMetadata setup: ", metadata_setup_end - metadata_setup_start)
        print("\tDict build: ", dict_build_end - metadata_setup_end)
        print("\tResult build: ", result_build_end - dict_build_end)

        return result

    def __getitem__(
            self,
            dataset_idx) -> Tuple[QuerySceneSequence, ResultsSceneSequence]:
        print(
            f"Argoverse2 Scene Flow dataset __getitem__({dataset_idx}) start")

        sequence_idx, subsequence_start_idx = self.dataset_to_sequence_subsequence_idx[
            dataset_idx]

        # Load sequence
        sequence = self.sequence_loader[sequence_idx]

        in_subsequence_src_index = (self.subsequence_length - 1) // 2
        in_subsequence_tgt_index = in_subsequence_src_index + 1
        # Load subsequence

        load_frames_start = time.time()
        subsequence_frames = [
            sequence.load(subsequence_start_idx + i,
                          subsequence_start_idx + in_subsequence_tgt_index)
            for i in range(self.subsequence_length)
        ]
        load_frames_end = time.time()

        scene_sequence = self._make_scene_sequence(subsequence_frames)
        make_scene_sequence_end = time.time()

        query_scene_sequence = self._make_query_scene_sequence(
            scene_sequence, subsequence_frames, in_subsequence_src_index,
            in_subsequence_tgt_index)
        make_query_scene_sequence_end = time.time()

        results_scene_sequence = self._make_results_scene_sequence(
            scene_sequence, subsequence_frames, in_subsequence_src_index,
            in_subsequence_tgt_index)
        make_results_scene_sequence_end = time.time()

        print("Load frames: ", load_frames_end - load_frames_start)
        print("Make scene sequence: ",
              make_scene_sequence_end - load_frames_end)
        print("Make query scene sequence: ",
              make_query_scene_sequence_end - make_scene_sequence_end)
        print("Make results scene sequence: ",
              make_results_scene_sequence_end - make_query_scene_sequence_end)

        print(f"Argoverse2 Scene Flow dataset __getitem__({dataset_idx}) end")

        return query_scene_sequence, results_scene_sequence