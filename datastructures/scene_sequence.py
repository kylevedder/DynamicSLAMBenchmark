import numpy as np
from typing import Dict, List

from .pointcloud import PointCloud
from .rgb_image import RGBImage
from .se3 import SE3


class RawScene():
    """
    This class contains only the raw percepts from a sequence. Its goal is to 
    describe the scene as it is observed by the sensors; it does not contain
    any other information such as motion descriptions.

    These percepts are multi-modal, and may not be time sync'd. As an example,
    in many AVs, the cameras run at 30 or 60 Hz while the lidar runs at 10 Hz.
    Thus, each percept is recorded as a lookup table from timestamps to the
    percept. We solve the time sync problem by pulling the most recent percept
    from each modality when requesting all percepts for a given timestep.

    These percept modalities are:
        - RGB
        - PointClouds

    Additionally, we store frame conversions for each percept. At a minumum, we
    need:
        - RGB -> PointCloud (Ego Frame)
        - PointCloud (Ego Frame) -> Global Frame
    """

    def __init__(self, pointcloud_lookup: Dict[int, PointCloud],
                 rgb_image_lookup: Dict[int, RGBImage],
                 world_T_ego_lookup: Dict[int, SE3], rgb_T_ego_lookup: Dict[int,
                                                                            SE3]):
        pass


class SceneSequence():
    """
    This class contains a full scene sequence.
    
    This means 
     - raw percepts
     - descriptions of motion betweeen these percepts
     - key frame timestamps

    The raw percepts must support multiple modalities. At a minumum, we need:
     - RGB
     - PointClouds
    
    Importantly, these raw percepts may not be temporally aligned. As an example
    in many AVs, the cameras run at 30 or 60 Hz while the lidar runs at 10 Hz. 
    We solve this by pulling the most recent percept from each modality.

    We also have to store frame conversions. At a minumum, we need:
     - RGB -> PointCloud (Ego Frame)
     - PointCloud (Ego Frame) -> Global Frame

    We also need to store the motion between frames. These descriptions of motion
    may be sparse (particle tracking) or dense (scene flow), and they may be 
    multi-frame (particle tracking) or single-frame (scene flow).  
    """

    def __init__(self, pointclouds: Dict[int, PointCloud],
                 rgb_images: Dict[int, RGBImage],
                 particle_positions: Dict[int, np.ndarray],
                 frame_conversions: Dict[int,
                                         SE3], frame_timestamps: Dict[int,
                                                                      float]):
        pass
