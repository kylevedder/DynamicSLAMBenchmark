from typing import Dict
from .kubric.dataset import Kubric
from .flyingthings3d.dataset import FlyingThings3D

__all__ = ["Kubric", "FlyingThings3D"]
dataset_names = [cls.lower() for cls in __all__]


def construct_dataset(name: str, args: dict):
    name = name.lower()
    all_lookup: Dict[str, str] = {cls.lower(): cls for cls in __all__}
    if name not in all_lookup:
        raise ValueError(f"Unknown dataset name: {name}")

    cls_name = all_lookup[name]
    # Convert cls_name string to class object using getattr
    cls = getattr(__import__(__name__), cls_name)
    return cls(**args)
