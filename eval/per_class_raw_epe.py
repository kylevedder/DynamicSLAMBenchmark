from scene_trajectory_benchmark.datastructures import (
    EstimatedParticleTrajectories,
    GroundTruthParticleTrajectories,
    Timestamp,
    ParticleClassId,
)
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, List, Set, Any, Union
import numpy as np

from .eval import Evaluator


@dataclass(frozen=True, eq=True, order=True, repr=True)
class SplitKey:
    name: str
    speed_buckets: Tuple[float, float]
    distance_threshold: float

    def __eq__(self, __value: object) -> bool:
        # TODO: This is a hack because the hash function works but the autogen eq function doesn't.
        return hash(self) == hash(__value)


@dataclass(frozen=True, eq=True, repr=True)
class SplitValue:
    avg_epe: float
    count: int
    avg_speed: float

    def __eq__(self, __value: object) -> bool:
        # TODO: This is a hack because the hash function works but the autogen eq function doesn't.
        return hash(self) == hash(__value)


class EvalFrameResult:
    def __init__(
        self,
        gt_world_points: np.ndarray,
        gt_speeds: np.ndarray,
        l2_errors: np.ndarray,
        gt_class_ids: np.ndarray,
        class_id_to_name=lambda e: e,
    ):
        assert (gt_world_points.ndim == 3
                ), f"gt_world_points must be 3D, got {gt_world_points.ndim}"
        assert (
            gt_world_points.shape[:2] == l2_errors.shape
        ), f"gt_world_points and l2_errors must have the same shape, got {gt_world_points.shape} and {l2_errors.shape}"
        assert (gt_class_ids.ndim == 1
                ), f"gt_class_ids must be 1D, got {gt_class_ids.ndim}"

        assert l2_errors.shape[0] == len(
            gt_class_ids
        ), f"l2_errors and gt_class_ids must have the same number of entries, got {l2_errors.shape[0]} and {len(gt_class_ids)}"

        assert gt_speeds.shape[0] == len(
            gt_class_ids
        ), f"gt_speeds and gt_class_ids must have the same number of entries, got {gt_speeds.shape[0]} and {len(gt_class_ids)}"

        assert (
            gt_speeds.shape == l2_errors.shape
        ), f"gt_speeds and l2_errors must have the same shape, got {gt_speeds.shape} and {l2_errors.shape}"

        self.class_error_dict = {
            k: v
            for k, v in
            self.make_splits(gt_world_points, gt_speeds, gt_class_ids,
                             l2_errors, class_id_to_name)
        }

    def make_splits(self, gt_world_points, gt_speeds, gt_class_ids, l2_errors,
                    class_id_to_name) -> List[Tuple[SplitKey, SplitValue]]:
        unique_gt_classes = np.unique(gt_class_ids)
        speed_thresholds = [(0, 0.05), (0.05, np.inf)]
        distance_thresholds = [35, np.inf]

        for class_id in unique_gt_classes:
            for speed_min, speed_max in speed_thresholds:
                for distance_threshold in distance_thresholds:
                    speed_bucket_mask = (gt_speeds >= speed_min) & (gt_speeds <
                                                                    speed_max)
                    class_matched_mask = gt_class_ids == class_id
                    distance_mask = (np.linalg.norm(
                        gt_world_points[:, :, :2], ord=np.inf, axis=2) <
                                     distance_threshold)

                    match_mask = (speed_bucket_mask
                                  & class_matched_mask[:, None]
                                  & distance_mask)
                    count = match_mask.sum()

                    if count > 0:
                        avg_epe = np.sum(l2_errors[match_mask]) / count
                        split_avg_speed = np.mean(gt_speeds[match_mask])
                    else:
                        avg_epe = 0
                        split_avg_speed = np.NaN
                    class_name = class_id_to_name(class_id)
                    yield SplitKey(class_name, (speed_min, speed_max),
                                   distance_threshold), SplitValue(
                                       avg_epe, count, split_avg_speed)


class PerClassRawEPEEvaluator(Evaluator):
    def __init__(self):
        self.eval_frame_results: List[EvalFrameResult] = []

    @staticmethod
    def from_evaluator_list(evaluator_list: List["PerClassRawEPEEvaluator"]):
        evaluator = PerClassRawEPEEvaluator()
        for e in evaluator_list:
            evaluator.eval_frame_results.extend(e.eval_frame_results)
        return evaluator

    def __add__(self, other: "PerClassRawEPEEvaluator"):
        if isinstance(other, int):
            if other == 0:
                return self
        return PerClassRawEPEEvaluator.from_evaluator_list([self, other])

    def __len__(self):
        return len(self.eval_frame_results)

    def _validate_inputs(
        self,
        predictions: EstimatedParticleTrajectories,
        ground_truth: GroundTruthParticleTrajectories,
    ):
        assert isinstance(
            predictions, EstimatedParticleTrajectories
        ), f"predictions must be a EstimatedParticleTrajectories, got {type(predictions)}"

        assert isinstance(
            ground_truth, GroundTruthParticleTrajectories
        ), f"ground_truth must be a GroundTruthParticleTrajectories, got {type(ground_truth)}"

        # Validate that the predictions and ground truth have the same number of predictions.

        assert len(predictions) == len(
            ground_truth
        ), f"predictions and ground_truth must have the same number of predictions, got {len(predictions)} and {len(ground_truth)}"

        assert (
            len(predictions) > 0
        ), f"predictions must have at least one prediction, got {len(predictions)}"

        # All Ground Truth Particle Trajectories must be in the set of Estimation Particle Trajectories.
        # It's acceptable for the Estimation Particle Trajectories to have more trajectories than
        # the Ground Truth Particle Trajectories.

        predictions_intersection_ground_truth = (predictions.is_valid
                                                 & ground_truth.is_valid)
        predictions_match_ground_truth = (
            predictions_intersection_ground_truth == ground_truth.is_valid)
        vectors = ground_truth.world_points[~predictions_match_ground_truth]
        assert (predictions_match_ground_truth).all(
        ), f"all ground truth particle trajectories must be in the estimation particle trajectories. Nonmatching points: {(~predictions_match_ground_truth).sum()}. Violating vectors: {vectors}"

        # All timestamps for the Ground Truth Particle Trajectories must be in the set of Estimation Particle Trajectories.
        # It's acceptable for the Estimation Particle Trajectories to have more timestamps than
        # the Ground Truth Particle Trajectories.
        assert set(ground_truth.trajectory_timestamps).issubset(
            set(predictions.trajectory_timestamps)
        ), f"all timestamps for the ground truth particle trajectories must be in the estimation particle trajectories. Nonmatching timestamps: {set(ground_truth.trajectory_timestamps) - set(predictions.trajectory_timestamps)}"

    def _get_indices_of_timestamps(
        self,
        predictions: EstimatedParticleTrajectories,
        ground_truth: GroundTruthParticleTrajectories,
        query_timestamp: Timestamp,
    ):
        # create an numpy array
        pred_timestamps = predictions.trajectory_timestamps

        traj_timestamps = ground_truth.trajectory_timestamps

        # index of first occurrence of each value
        sorter = np.argsort(pred_timestamps)

        matched_idxes = sorter[np.searchsorted(pred_timestamps,
                                               traj_timestamps,
                                               sorter=sorter)]

        # find the index of the query timestamp in traj_timestamps
        query_idx = np.where(traj_timestamps == query_timestamp)[0][0]

        return matched_idxes, query_idx

    def eval(
        self,
        predictions: EstimatedParticleTrajectories,
        ground_truth: GroundTruthParticleTrajectories,
        query_timestamp: Timestamp,
    ):
        self._validate_inputs(predictions, ground_truth)

        # Extract the ground truth entires for the timestamps that are in both the predictions and ground truth.
        # It could be that the predictions have more timestamps than the ground truth.

        matched_time_axis_indices, query_idx = self._get_indices_of_timestamps(
            predictions, ground_truth, query_timestamp)

        eval_particle_ids = ground_truth.valid_particle_ids()

        gt_is_valids = ground_truth.is_valid[
            eval_particle_ids][:, matched_time_axis_indices]

        pred_is_valids = predictions.is_valid[
            eval_particle_ids][:, matched_time_axis_indices]

        # Make sure that all the pred_is_valids are true if gt_is_valids is true.
        assert ((gt_is_valids & pred_is_valids) == gt_is_valids).all(
        ), f"all gt_is_valids must be true if pred_is_valids is true."

        gt_world_points = ground_truth.world_points[
            eval_particle_ids][:, matched_time_axis_indices]
        pred_world_points = predictions.world_points[
            eval_particle_ids][:, matched_time_axis_indices]
        gt_is_occluded = ground_truth.is_occluded[
            eval_particle_ids][:, matched_time_axis_indices]
        pred_is_occluded = predictions.is_occluded[
            eval_particle_ids][:, matched_time_axis_indices]

        gt_class_ids = ground_truth.cls_ids[eval_particle_ids]

        # Compute the L2 error between the ground truth and the prediction points for each particle trajectory.
        l2_errors = np.linalg.norm(gt_world_points - pred_world_points, axis=2)

        is_occluded_errors = np.logical_xor(gt_is_occluded, pred_is_occluded)

        # The query index L2 errors should be zero, as the query points should be the same as the entries in the ground truth.
        assert np.isclose(
            l2_errors[:, query_idx], np.zeros_like(l2_errors[:, query_idx])
        ).all(
        ), f"l2_errors for the query index should be zero, got {l2_errors[:, query_idx]}"

        if query_idx != 0:
            raise NotImplementedError(
                "TODO: Handle query_idx != 0 when computing speed bucketing.")

        # Compute the speed of each gt particle trajectory.
        gt_speeds = np.linalg.norm(np.diff(gt_world_points, axis=1), axis=2)

        non_query_idx_l2_errors = np.delete(l2_errors, query_idx, axis=1)
        non_query_idx_gt_world_points = np.delete(gt_world_points,
                                                  query_idx,
                                                  axis=1)

        eval_frame_result = EvalFrameResult(
            non_query_idx_gt_world_points,
            gt_speeds,
            non_query_idx_l2_errors,
            gt_class_ids,
            ground_truth.pretty_name,
        )

        self.eval_frame_results.append(eval_frame_result)

    def _save_intermediary_results(self):
        import pickle

        with open("/tmp/frame_results/eval_frame_results.pkl", "wb") as f:
            pickle.dump(self.eval_frame_results, f)

    def _category_to_per_frame_stats(self) -> Dict[SplitKey, List[SplitValue]]:
        # From list of dicts to dict of lists
        merged_class_error_dict = dict()
        for eval_frame_result in self.eval_frame_results:
            for k, v in eval_frame_result.class_error_dict.items():
                if k not in merged_class_error_dict:
                    merged_class_error_dict[k] = [v]
                else:
                    merged_class_error_dict[k].append(v)
        return merged_class_error_dict

    def _category_to_average_stats(
        self, merged_class_error_dict: Dict[SplitKey, SplitValue]
    ) -> Dict[SplitKey, SplitValue]:
        # Compute the average EPE for each key
        result_dict = dict()
        for k in sorted(merged_class_error_dict.keys()):
            values = merged_class_error_dict[k]
            epes = np.array([v.avg_epe for v in values])
            counts = np.array([v.count for v in values])
            # Average of the epes weighted by the counts

            weighted_split_avg_speed = np.NaN
            weighted_average_epe = np.NaN
            if counts.sum() > 0:
                valid_counts_mask = counts > 0

                weighted_average_epe = np.average(
                    epes[valid_counts_mask],
                    weights=(counts[valid_counts_mask]))
                weighted_split_avg_speed = np.average(
                    np.array([v.avg_speed for v in values])[valid_counts_mask],
                    weights=(counts[valid_counts_mask]),
                )

            result_dict[k] = SplitValue(weighted_average_epe, counts.sum(),
                                        weighted_split_avg_speed)
        return result_dict

    def _save_stats_tables(self, average_stats: Dict[SplitKey, SplitValue]):
        assert (
            len(average_stats) > 0
        ), f"average_stats must have at least one entry, got {len(average_stats)}"

        unique_distance_thresholds = sorted(
            set([k.distance_threshold for k in average_stats.keys()]))
        unique_speed_buckets = sorted(
            set([k.speed_buckets for k in average_stats.keys()]))
        unique_category_names = sorted(
            set([k.name for k in average_stats.keys()]))

        for distance_threshold in unique_distance_thresholds:
            raw_table_save_path = (
                f"/tmp/frame_results/raw_table_{distance_threshold}.csv")
            speed_table_save_path = (
                f"/tmp/frame_results/speed_table_{distance_threshold}.csv")

            # Rows are category names, columns are for speed buckets
            epe_table = pd.DataFrame(
                index=unique_category_names,
                columns=[str(e) for e in unique_speed_buckets],
            )

            speed_table = pd.DataFrame(
                index=unique_category_names,
                columns=[str(e) for e in unique_speed_buckets],
            )

            for category_name in unique_category_names:
                for speed_bucket in unique_speed_buckets:
                    key = SplitKey(category_name, speed_bucket,
                                   distance_threshold)
                    avg_epe = np.NaN
                    avg_speed = np.NaN
                    if key in average_stats:
                        avg_epe = average_stats[key].avg_epe
                        avg_speed = average_stats[key].avg_speed
                    epe_table.loc[category_name, str(speed_bucket)] = avg_epe
                    speed_table.loc[category_name,
                                    str(speed_bucket)] = avg_speed

            epe_table.to_csv(raw_table_save_path)
            speed_table.to_csv(speed_table_save_path)

    def compute_results(self, save_results: bool = True) -> Dict[str, float]:
        assert (len(self.eval_frame_results) >
                0), "Must call eval at least once before calling compute"
        if save_results:
            self._save_intermediary_results()

        # From list of dicts to dict of lists
        category_to_per_frame_stats = self._category_to_per_frame_stats()
        category_to_average_stats = self._category_to_average_stats(
            category_to_per_frame_stats)
        for k, v in category_to_average_stats.items():
            print(k, v)

        self._save_stats_tables(category_to_average_stats)
        return category_to_average_stats
