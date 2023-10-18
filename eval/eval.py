from scene_trajectory_benchmark.datastructures import EstimatedParticleTrajectories, GroundTruthParticleTrajectories
import time
import numpy as np


class Evaluator():

    def __init__(self, dataset):
        self.dataset = dataset

    def _validate_inputs(self, predictions: EstimatedParticleTrajectories,
                         ground_truth: GroundTruthParticleTrajectories):

        assert isinstance(
            predictions, EstimatedParticleTrajectories
        ), f'predictions must be a EstimatedParticleTrajectories, got {type(predictions)}'

        assert isinstance(
            ground_truth, GroundTruthParticleTrajectories
        ), f'ground_truth must be a GroundTruthParticleTrajectories, got {type(ground_truth)}'

        # Validate that the predictions and ground truth have the same number of predictions.

        assert len(predictions) == len(
            ground_truth
        ), f"predictions and ground_truth must have the same number of predictions, got {len(predictions)} and {len(ground_truth)}"

        assert len(
            predictions
        ) > 0, f"predictions must have at least one prediction, got {len(predictions)}"

        # All Ground Truth Particle Trajectories must be in the set of Estimation Particle Trajectories.
        # It's acceptable for the Estimation Particle Trajectories to have more trajectories than
        # the Ground Truth Particle Trajectories.

        predictions_intersection_ground_truth = (predictions.is_valid
                                                 & ground_truth.is_valid)
        predictions_match_ground_truth = (
            predictions_intersection_ground_truth == ground_truth.is_valid)
        vectors = ground_truth.world_points[~predictions_match_ground_truth]
        assert (predictions_match_ground_truth).all(), \
            f"all ground truth particle trajectories must be in the estimation particle trajectories. Nonmatching points: {(~predictions_match_ground_truth).sum()}. Violating vectors: {vectors}"

        # All timestamps for the Ground Truth Particle Trajectories must be in the set of Estimation Particle Trajectories.
        # It's acceptable for the Estimation Particle Trajectories to have more timestamps than
        # the Ground Truth Particle Trajectories.
        assert set(ground_truth.trajectory_timestamps).issubset(set(predictions.trajectory_timestamps)), \
            f"all timestamps for the ground truth particle trajectories must be in the estimation particle trajectories. Nonmatching timestamps: {set(ground_truth.trajectory_timestamps) - set(predictions.trajectory_timestamps)}"

    def _get_indices_of_timestamps_in_both(
            self, predictions: EstimatedParticleTrajectories,
            ground_truth: GroundTruthParticleTrajectories):
        # create an numpy array
        pred_timestamps = predictions.trajectory_timestamps

        traj_timestamps = ground_truth.trajectory_timestamps

        # index of first occurrence of each value
        sorter = np.argsort(pred_timestamps)

        matched_idxes = sorter[np.searchsorted(pred_timestamps,
                                               traj_timestamps,
                                               sorter=sorter)]
        return matched_idxes

    def eval(self, predictions: EstimatedParticleTrajectories,
             ground_truth: GroundTruthParticleTrajectories):
        self._validate_inputs(predictions, ground_truth)

        # Extract the ground truth entires for the timestamps that are in both the predictions and ground truth.

        matched_time_axis_indices = self._get_indices_of_timestamps_in_both(
            predictions, ground_truth)
        eval_particle_ids = ground_truth.valid_particle_ids()

        is_valids = ground_truth.is_valid[
            eval_particle_ids][:, matched_time_axis_indices]
        gt_world_points = ground_truth.world_points[
            eval_particle_ids][:, matched_time_axis_indices]
        pred_world_points = predictions.world_points[
            eval_particle_ids][:, matched_time_axis_indices]
        gt_is_occluded = ground_truth.is_occluded[
            eval_particle_ids][:, matched_time_axis_indices]
        pred_is_occluded = predictions.is_occluded[
            eval_particle_ids][:, matched_time_axis_indices]
        # gt_class_ids = ground_truth.cls_ids[
        #     eval_particle_ids][:, matched_time_axis_indices]

        # Compute the L2 error between the ground truth and the prediction points for each particle trajectory.
        l2_errors = np.linalg.norm(gt_world_points - pred_world_points, axis=2)
        is_occluded_errors = np.logical_xor(gt_is_occluded, pred_is_occluded)
        breakpoint()
