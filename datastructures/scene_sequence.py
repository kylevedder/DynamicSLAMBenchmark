from typing import Dict

from pointcloud import PointCloud
from rgb_image import RGBImage

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

    def __init__(self, pointclouds : Dict[int, PointCloud], rgb_images : Dict[int, RGBImage], particle_positions: Dict[int, ParticlePosition], frame_conversions: Dict[int, SE3], frame_timestamps: Dict[int, float]):
        pass

