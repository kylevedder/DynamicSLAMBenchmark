import numpy as np
from typing import Dict, List, Tuple, Any

from .camera_projection import CameraProjection
from .pointcloud import PointCloud
from .rgb_image import RGBImage
from .se3 import SE3
from .particle_frame import ParticleFrame

from dataclasses import dataclass


@dataclass
class PointCloudFrame():
    pc: PointCloud
    ego_to_global: SE3


@dataclass
class RGBFrame():
    rgb: RGBImage
    sensor_to_ego: SE3
    camera_projection: CameraProjection


@dataclass
class QueryPoint():
    point: np.ndarray
    timestamp: int


class RawSceneSequence():
    """
    This class contains only the raw percepts from a sequence. Its goal is to 
    describe the scene as it is observed by the sensors; it does not contain
    any other information such as point position descriptions.

    These percepts are multi-modal, and may not be time sync'd. As an example,
    in many AVs, the cameras run at 30 or 60 Hz while the lidar runs at 10 Hz.
    Thus, each percept is recorded as a lookup table from timestamps to the
    percept. We solve the time sync problem by pulling the most recent percept
    from each modality when requesting all percepts for a given timestep.

    These percept modalities are:
        - RGB
        - PointClouds

    Additionally, we store frame conversions for each percept. We assume the 
    frame of the point cloud is the ego frame, and we store the following:
        - PointCloud, Ego to Global
        - RGB, Sensor to Ego
    """

    def __init__(self, pointcloud_lookup: Dict[int, PointCloudFrame],
                 rgb_image_lookup: Dict[int, RGBFrame]):
        self.pc_map = pointcloud_lookup
        self.pc_timestamp_array = np.array(sorted(pointcloud_lookup.keys()))
        self.rgb_map = rgb_image_lookup
        self.rgb_timestamp_array = np.array(sorted(rgb_image_lookup.keys()))

    def get_percept_timesteps(self) -> List[int]:
        keys = set(self.pc_map.keys()).union(set(self.rgb_map.keys()))
        return sorted(keys)

    def _get_current_or_most_recent_timestamp(self, query_timestamp: int,
                                              timestamp_array: np.ndarray,
                                              lookup_map: Dict[int, Any]):
        if query_timestamp > timestamp_array[-1]:
            result_index = -1
        else:
            result_index = np.searchsorted(timestamp_array,
                                           query_timestamp,
                                           side='left')
        result_timestamp = timestamp_array[result_index]
        return lookup_map[result_timestamp]

    def __len__(self):
        return len(self.get_percept_timesteps())

    def get_percepts(self, timestamp: int) -> Tuple[PointCloudFrame, RGBFrame]:
        pc_frame: PointCloudFrame = self._get_current_or_most_recent_timestamp(
            timestamp, self.pc_timestamp_array, self.pc_map)
        rgb_frame: RGBFrame = self._get_current_or_most_recent_timestamp(
            timestamp, self.rgb_timestamp_array, self.rgb_map)

        return pc_frame, rgb_frame


class QuerySceneSequence(RawSceneSequence):
    """
    This class describes a scene sequence with a query for motion descriptions.

    A query is a point + timestamp in the global frame of the scene, along with 
    series of timestamps for which a point description is requested; motion is
    implied to be linear between these points at the requested timestamps.
    """

    def __init__(self, pointcloud_lookup: Dict[int, PointCloudFrame],
                 rgb_image_lookup: Dict[int, RGBFrame],
                 query_timestamps: List[int], query_points: List[QueryPoint]):
        super().__init__(pointcloud_lookup, rgb_image_lookup)
        self.query_timestamps = query_timestamps
        self.query_points = query_points

    def __len__(self) -> int:
        return len(self.query_timestamps)

    def get_timesteps(self) -> List[int]:
        return self.query_timestamps


class ResultsSceneSequence(RawSceneSequence):
    """
    This class describes a scene sequence result.

    A result is a series of points + timestamps in the global frame of the
    scene.
    """

    def __init__(self, pointcloud_lookup: Dict[int, PointCloudFrame],
                 rgb_image_lookup: Dict[int, RGBFrame],
                 particle_frames: Dict[int, ParticleFrame]):
        super().__init__(pointcloud_lookup, rgb_image_lookup)
        self.particle_frames = particle_frames