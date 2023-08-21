import numpy as np
from typing import Dict, List, Tuple, Any, Union

from numpy._typing import NDArray

from .camera_projection import CameraProjection
from .pointcloud import PointCloud
from .rgb_image import RGBImage
from .se3 import SE3
from .o3d_visualizer import O3DVisualizer

from dataclasses import dataclass

# Type alias for particle IDs
ParticleID = int
ParticleClassId = int
Timestamp = int

# Type alias for world points
WorldParticle = np.array


@dataclass
class EstimatedParticle():
    point: WorldParticle
    is_occluded: bool


@dataclass
class PoseInfo():
    sensor_to_ego: SE3
    ego_to_global: SE3


@dataclass
class PointCloudFrame():
    pc: PointCloud
    pose: PoseInfo

    @property
    def global_pc(self) -> PointCloud:
        pose = self.global_pose
        return self.pc.transform(pose)

    @property
    def global_pose(self) -> SE3:
        return self.pose.ego_to_global @ self.pose.sensor_to_ego


@dataclass
class RGBFrame():
    rgb: RGBImage
    pose: PoseInfo
    camera_projection: CameraProjection


@dataclass
class ParticleTrajectory():
    id: ParticleID
    trajectory: Dict[Timestamp, EstimatedParticle]
    cls: Union[ParticleClassId, None] = None

    def __len__(self):
        return len(self.trajectory)

    def get_first_timestamp(self) -> Timestamp:
        return min(self.trajectory.keys())

    def __getitem__(self, timestamp: Timestamp) -> EstimatedParticle:
        return self.trajectory[timestamp]


def _particle_id_to_color(
        particle_id: ParticleID) -> Tuple[float, float, float]:
    particle_id = int(particle_id)
    assert isinstance(particle_id, ParticleID), \
        f"particle_id must be a ParticleID ({ParticleID}) , got {type(particle_id)}"
    hash_val = abs(hash(particle_id)) % (256**3)
    return np.array([((hash_val >> 16) & 0xff) / 255,
                     ((hash_val >> 8) & 0xff) / 255, (hash_val & 0xff) / 255])


class RawSceneSequence():
    """
    This class contains only the raw percepts from a sequence. Its goal is to 
    describe the scene as it is observed by the sensors; it does not contain
    any other information such as point position descriptions.

    These percept modalities are:
        - RGB
        - PointClouds

    Additionally, we store frame conversions for each percept.
    """

    def __init__(self, percept_lookup: Dict[Timestamp, Tuple[PointCloudFrame,
                                                             RGBFrame]]):
        self.percept_lookup = percept_lookup

    def get_percept_timesteps(self) -> List[int]:
        return sorted(self.percept_lookup.keys())

    def __len__(self):
        return len(self.get_percept_timesteps())

    def __getitem__(self, timestamp: int) -> Tuple[PointCloudFrame, RGBFrame]:
        assert isinstance(
            timestamp, int), f"timestamp must be an int, got {type(timestamp)}"
        pc_frame, rgb_frame = self.percept_lookup[timestamp]
        return pc_frame, rgb_frame

    def visualize(self, vis: O3DVisualizer) -> O3DVisualizer:
        timesteps = self.get_percept_timesteps()
        grayscale_color = np.linspace(0, 1, len(timesteps) + 1)
        for idx, timestamp in enumerate(timesteps):
            pc_frame, rgb_frame = self[timestamp]
            vis.add_pc_frame(pc_frame, color=[grayscale_color[idx]] * 3)
            vis.add_pose(pc_frame.global_pose)
        return vis


class QueryParticleLookup():
    """
    This class is an efficient lookup table for query particles.
    """

    def __init__(self, num_entries: int):
        self.num_entries = num_entries
        self.world_particles = np.zeros((num_entries, 3), dtype=np.float32)
        self.timestamps = np.zeros((num_entries, ), dtype=np.int64)

    def __len__(self) -> int:
        return self.num_entries

    def __getitem__(
            self, particle_id: ParticleID) -> Tuple[WorldParticle, Timestamp]:
        assert particle_id < self.num_entries, \
            f"particle_id {particle_id} must be less than {self.num_entries}"
        return self.world_particles[particle_id], self.timestamps[particle_id]

    def __setitem__(self, particle_id: ParticleID, value: Tuple[WorldParticle,
                                                                Timestamp]):
        assert (particle_id < self.num_entries).all(), \
            f"particle_ids {particle_id[particle_id >= self.num_entries]} must be less than {self.num_entries}"
        self.world_particles[particle_id] = value[0]
        self.timestamps[particle_id] = value[1]

    @property
    def particle_ids(self) -> NDArray:
        return np.arange(self.num_entries)


class QuerySceneSequence:
    """
    This class describes a scene sequence with a query for motion descriptions.

    A query is a point + timestamp in the global frame of the scene, along with 
    series of timestamps for which a point description is requested; motion is
    implied to be linear between these points at the requested timestamps.
    """

    def __init__(self, scene_sequence: RawSceneSequence,
                 query_particles: QueryParticleLookup,
                 query_timestamps: List[Timestamp]):
        assert isinstance(scene_sequence, RawSceneSequence), \
            f"scene_sequence must be a RawSceneSequence, got {type(scene_sequence)}"
        assert isinstance(query_particles, QueryParticleLookup), \
            f"query_particles must be a dict, got {type(query_particles)}"
        assert isinstance(query_timestamps, list), \
            f"query_timestamps must be a list, got {type(query_timestamps)}"

        self.scene_sequence = scene_sequence

        ###################################################
        # Sanity checks to ensure that the query is valid #
        ###################################################

        # Check that the query timestamps all have corresponding percepts
        assert set(query_timestamps).issubset(set(self.scene_sequence.get_percept_timesteps())), \
            f"Query timestamps {query_timestamps} must be a subset of the scene sequence percepts {self.scene_sequence.get_percept_timesteps()}"

        # Check that the query points all have corresponding timestamps
        # assert len(
        #     set(self.scene_sequence.get_percept_timesteps()).intersection(
        #         set([t for _, t in query_particles.values()]))) > 0

        self.query_timestamps = query_timestamps
        self.query_particles = query_particles

    def __len__(self) -> int:
        return len(self.query_timestamps)

    def visualize(self,
                  vis: O3DVisualizer,
                  percent_subsample: Union[None, float] = None,
                  verbose=False) -> O3DVisualizer:
        if percent_subsample is not None:
            assert percent_subsample > 0 and percent_subsample <= 1, \
                f"percent_subsample must be in (0, 1], got {percent_subsample}"
            every_kth_particle = int(1 / percent_subsample)
        else:
            every_kth_particle = 1
        # Visualize the query points ordered by particle ID
        particle_ids = self.query_particles.particle_ids
        world_particles = self.query_particles.world_particles

        kth_particle_ids = particle_ids[::every_kth_particle]
        kth_world_particles = world_particles[::every_kth_particle]
        kth_particle_colors = [
            _particle_id_to_color(particle_id)
            for particle_id in kth_particle_ids
        ]

        vis.add_spheres(kth_world_particles, 0.1, kth_particle_colors)
        return vis


class ParticleTrajectoriesLookup():
    """
    This class is an efficient lookup table for particle trajectories.

    It is designed to present like Dict[ParticleID, ParticleTrajectory] but backed by a numpy array.
    """

    def __init__(self, num_entries: int, trajectory_length: int):
        self.num_entries = num_entries
        self.trajectory_length = trajectory_length

        self.world_points = np.zeros((num_entries, trajectory_length, 3),
                                     dtype=np.float32)
        self.timestamps = np.zeros((num_entries, trajectory_length),
                                   dtype=np.int64)
        self.is_occluded = np.zeros((num_entries, trajectory_length),
                                    dtype=bool)
        # By default, all trajectories are invalid
        self.is_valid = np.zeros((num_entries, trajectory_length), dtype=bool)
        self.cls_ids = np.zeros((num_entries, ), dtype=np.int64)

    def __len__(self) -> int:
        return self.num_entries

    def __getitem__(self, particle_id: ParticleID) -> ParticleTrajectory:
        points = self.world_points[particle_id]
        timestamps = self.timestamps[particle_id]
        is_occluded = self.is_occluded[particle_id]
        cls_id = self.cls_ids[particle_id]

        trajectory = {
            timestamp: EstimatedParticle(point, occluded)
            for point, timestamp, occluded in zip(points, timestamps,
                                                  is_occluded)
        }
        return ParticleTrajectory(particle_id, trajectory, cls_id)

    def __setitem__(self, particle_id: ParticleID,
                    data_tuple: Tuple[NDArray, NDArray, NDArray,
                                      ParticleClassId]):
        points, timestamps, is_occludeds, cls_ids = data_tuple
        self.world_points[particle_id] = points
        self.timestamps[particle_id] = timestamps
        self.is_occluded[particle_id] = is_occludeds
        self.cls_ids[particle_id] = cls_ids
        self.is_valid[particle_id] = True


class ResultsSceneSequence:
    """
    This class describes a scene sequence result.

    A result is a series of points + timestamps in the global frame of the
    scene.
    """

    def __init__(self, scene_sequence: RawSceneSequence,
                 particle_trajectories: ParticleTrajectoriesLookup):
        assert isinstance(scene_sequence, RawSceneSequence), \
            f"scene_sequence must be a RawSceneSequence, got {type(scene_sequence)}"

        assert isinstance(particle_trajectories, ParticleTrajectoriesLookup), \
            f"particle_frames must be a dict, got {type(particle_trajectories)}"
        self.scene_sequence = scene_sequence

        self.particle_trajectories = particle_trajectories

    def __len__(self) -> int:
        return len(self.particle_trajectories)

    def visualize(self,
                  vis: O3DVisualizer,
                  percent_subsample: Union[None, float] = None,
                  verbose: bool = False) -> O3DVisualizer:
        if percent_subsample is not None:
            assert percent_subsample > 0 and percent_subsample <= 1, \
                f"percent_subsample must be in (0, 1], got {percent_subsample}"
            every_kth_particle = int(1 / percent_subsample)
        else:
            every_kth_particle = 1

        # Shape: points, 2, 3
        world_points = self.particle_trajectories.world_points
        is_valid = self.particle_trajectories.is_valid

        vis.add_trajectories(world_points)
        return vis