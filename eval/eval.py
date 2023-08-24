from scene_trajectory_benchmark.datastructures import EstimatedParticleTrajectories, GroundTruthParticleTrajectories
import time


class Evaluator():

    def __init__(self, dataset):
        self.dataset = dataset

    def eval(self, predictions: EstimatedParticleTrajectories,
             ground_truth: GroundTruthParticleTrajectories):
        before_time = time.time()

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

        after_time = time.time()
