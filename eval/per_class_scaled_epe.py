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
import pickle
import json
import enum
from .eval import Evaluator
import copy


@dataclass(frozen=True, eq=True, order=True, repr=True)
class SplitKey:
    name: str
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


class ScalingType(enum.Enum):
    CONSTANT = "constant"
    FOUR_D = "4d"
    FOUR_D_01 = "4d_01"

    @staticmethod
    def from_str(s: str):
        # Iterate over members and check if match string by value
        for member in ScalingType:
            if member.value == s:
                return member
        raise ValueError(f"ScalingType {s} not found.")


class EvalFrameResult:

    def __init__(
        self,
        scaling_type: ScalingType,
        gt_world_points: np.ndarray,
        gt_class_ids: np.ndarray,
        gt_flow: np.ndarray,
        pred_flow: np.ndarray,
        class_id_to_name=lambda e: e,
    ):
        self.scaling_type = scaling_type

        assert (gt_world_points.ndim == 2
                ), f"gt_world_points must be 3D, got {gt_world_points.ndim}"
        assert (
            gt_world_points.shape == gt_flow.shape
        ), f"gt_world_points and gt_flow must have the same shape, got {gt_world_points.shape} and {gt_flow.shape}"
        assert (gt_class_ids.ndim == 1
                ), f"gt_class_ids must be 1D, got {gt_class_ids.ndim}"
        assert gt_flow.shape == pred_flow.shape, f"gt_flow and pred_flow must have the same shape, got {gt_flow.shape} and {pred_flow.shape}"

        gt_speeds = np.linalg.norm(gt_flow, axis=1)

        scaled_gt_flow, scaled_pred_flow = self._scale_flows(
            gt_flow, pred_flow)

        scaled_epe_errors = np.linalg.norm(scaled_gt_flow - scaled_pred_flow,
                                           axis=1)

        self.class_error_dict = {
            k: v
            for k, v in
            self.make_splits(gt_world_points, gt_speeds, gt_class_ids,
                             scaled_epe_errors, class_id_to_name)
        }

    def _scale_flows(self, gt_flow: np.ndarray,
                     pred_flow: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        if self.scaling_type == ScalingType.CONSTANT:
            gt_speeds = np.linalg.norm(gt_flow, axis=1)
            scaled_gt_flow = gt_flow / (gt_speeds + 1)[:, None]
            scaled_pred_flow = pred_flow / (gt_speeds + 1)[:, None]
            return scaled_gt_flow, scaled_pred_flow

        elif self.scaling_type == ScalingType.FOUR_D or self.scaling_type == ScalingType.FOUR_D_01:

            def augment_flow(flow, value: float = 1):
                # Add a fourth dimension of ones to the flow
                return np.concatenate(
                    [flow, np.ones((flow.shape[0], 1)) * value], axis=1)

            def deaugment_flow(flow):
                # Remove the fourth dimension of ones from the flow
                return flow[:, :3]

            augmentation_value = 1.0
            if self.scaling_type == ScalingType.FOUR_D_01:
                augmentation_value = 0.1

            gt_flow_aug = augment_flow(gt_flow, value=augmentation_value)
            pred_flow_aug = augment_flow(pred_flow, value=augmentation_value)

            gt_aug_norm = np.linalg.norm(gt_flow_aug, axis=1)

            scaled_gt_flow_aug = gt_flow_aug / gt_aug_norm[:, None]
            scaled_pred_flow_aug = pred_flow_aug / gt_aug_norm[:, None]

            scaled_gt_flow = deaugment_flow(scaled_gt_flow_aug)
            scaled_pred_flow = deaugment_flow(scaled_pred_flow_aug)
            return scaled_gt_flow, scaled_pred_flow
        else:
            raise NotImplementedError(
                f"Scaling type {self.scaling_type} not implemented.")

    def make_splits(self, gt_world_points, gt_speeds, gt_class_ids, epe_errors,
                    class_id_to_name) -> List[Tuple[SplitKey, SplitValue]]:
        unique_gt_classes = np.unique(gt_class_ids)
        distance_thresholds = [35, np.inf]

        for class_id in unique_gt_classes:
            for distance_threshold in distance_thresholds:
                class_matched_mask = gt_class_ids == class_id
                distance_mask = (np.linalg.norm(gt_world_points[:, :2],
                                                ord=np.inf,
                                                axis=1) < distance_threshold)

                match_mask = class_matched_mask & distance_mask
                count = match_mask.sum()

                if count > 0:
                    avg_epe = np.sum(epe_errors[match_mask]) / count
                    split_avg_speed = np.mean(gt_speeds[match_mask])
                else:
                    avg_epe = 0
                    split_avg_speed = np.NaN
                class_name = class_id_to_name(class_id)
                yield SplitKey(class_name, distance_threshold), SplitValue(
                    avg_epe, count, split_avg_speed)


class PerClassScaledEPEEvaluator(Evaluator):

    def __init__(self, scaling_type: str):
        self.eval_frame_results: List[EvalFrameResult] = []
        self.scaling_type = ScalingType.from_str(scaling_type.lower())
        self.output_path = Path(
            f"/tmp/frame_results/scaled_epe/{self.scaling_type.value}")
        print(f"Saving results to {self.output_path}")
        # make the directory if it doesn't exist
        self.output_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def from_evaluator_list(
            evaluator_list: List["PerClassScaledEPEEvaluator"]):
        assert len(evaluator_list
                   ) > 0, "evaluator_list must have at least one evaluator"

        # Ensure that all evaluators have the same scaling type
        assert len(
            set(e.scaling_type for e in evaluator_list)
        ) == 1, f"All evaluators must have the same scaling type, got {set(e.scaling_type for e in evaluator_list)}"

        return sum(evaluator_list)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __add__(self, other: "PerClassScaledEPEEvaluator"):
        if isinstance(other, int):
            if other == 0:
                return self

        assert isinstance(
            other, PerClassScaledEPEEvaluator
        ), f"other must be a PerClassScaledEPEEvaluator, got {type(other)}"

        # Ensure that both evaluators have the same scaling type
        assert self.scaling_type == other.scaling_type, f"Both evaluators must have the same scaling type, got {self.scaling_type} and {other.scaling_type}"

        # Concatenate the eval_frame_results
        evaluator = copy.deepcopy(self)
        evaluator.eval_frame_results.extend(other.eval_frame_results)
        return evaluator

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

        # We only support Scene Flow for now
        if query_idx != 0:
            raise NotImplementedError(
                "TODO: Handle query_idx != 0 when computing speed bucketing.")

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

        assert (
            gt_world_points.shape[1] == 2
        ), f"gt_world_points must have 2 timestamps; we only support Scene Flow. Instead we got {gt_world_points.shape[1]} dimensions."
        assert (
            pred_world_points.shape[1] == 2
        ), f"pred_world_points must have 2 timestamps; we only support Scene Flow. Instead we got {pred_world_points.shape[1]} dimensions."

        # Query index should have roughly the same values.
        assert np.isclose(
            gt_world_points[:, query_idx], pred_world_points[:, query_idx]
        ).all(
        ), f"gt_world_points and pred_world_points should have the same values for the query index, got {gt_world_points[:, query_idx]} and {pred_world_points[:, query_idx]}"

        pc1 = gt_world_points[:, 0]
        gt_pc2 = gt_world_points[:, 1]
        pred_pc2 = pred_world_points[:, 1]

        gt_flow = gt_pc2 - pc1
        pred_flow = pred_pc2 - pc1

        eval_frame_result = EvalFrameResult(
            self.scaling_type,
            pc1,
            gt_class_ids,
            gt_flow,
            pred_flow,
            class_id_to_name=ground_truth.pretty_name)

        self.eval_frame_results.append(eval_frame_result)

    def _save_intermediary_results(self):
        save_path = self.output_path / "eval_frame_results.pkl"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
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
        self, merged_class_error_dict: Dict[SplitKey, List[SplitValue]]
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

    def _save_dict(self, path: Path, data: Dict[str, float]):
        with open(path, 'w') as f:  # You will need 'wb' mode in Python 2.x
            json.dump(data, f)

    def _save_stats_tables(self, average_stats: Dict[SplitKey, SplitValue]):
        assert (
            len(average_stats) > 0
        ), f"average_stats must have at least one entry, got {len(average_stats)}"

        unique_distance_thresholds = sorted(
            set([k.distance_threshold for k in average_stats.keys()]))
        unique_category_names = sorted(
            set([k.name for k in average_stats.keys()]))

        for distance_threshold in unique_distance_thresholds:
            raw_table_save_path = self.output_path / f"metric_table_{distance_threshold}.json"
            speed_table_save_path = self.output_path / f"speed_table_{distance_threshold}.json"

            # Rows are category names, columns are for speed buckets

            epe_dict = dict()
            speed_dict = dict()

            for category_name in unique_category_names:
                key = SplitKey(category_name, distance_threshold)
                avg_epe = np.NaN
                avg_speed = np.NaN
                if key in average_stats:
                    avg_epe = average_stats[key].avg_epe
                    avg_speed = average_stats[key].avg_speed
                epe_dict[category_name] = avg_epe
                speed_dict[category_name] = avg_speed

            Path(raw_table_save_path).parent.mkdir(parents=True, exist_ok=True)
            Path(speed_table_save_path).parent.mkdir(parents=True,
                                                     exist_ok=True)
            self._save_dict(raw_table_save_path, epe_dict)
            self._save_dict(speed_table_save_path, speed_dict)

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
