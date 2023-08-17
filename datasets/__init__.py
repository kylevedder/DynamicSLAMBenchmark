from typing import Dict
import scene_trajectory_benchmark.datastructures as datastructures
from scene_trajectory_benchmark.datasets.kubric.dataset import Kubric
from scene_trajectory_benchmark.datasets.flyingthings3d.dataset import FlyingThings3D
from scene_trajectory_benchmark.datasets.pointodyssey.dataset import PointOdyssey
from scene_trajectory_benchmark.datasets.argoverse2 import Argoverse2SceneFlow

__all__ = ["Kubric", "FlyingThings3D", "PointOdyssey", "Argoverse2SceneFlow"]
dataset_names = [cls.lower() for cls in __all__]


def construct_dataset(name: str, args: dict):
    name = name.lower()
    all_lookup: Dict[str, str] = {cls.lower(): cls for cls in __all__}
    if name not in all_lookup:
        raise ValueError(f"Unknown dataset name: {name}")

    cls_name = all_lookup[name]
    # Convert cls_name string to class object using getattr
    print("Importing: ", __import__(__name__), cls_name)
    cls = getattr(__import__(__name__), cls_name)
    return cls(**args)
