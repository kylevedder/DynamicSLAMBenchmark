import loaders
from pathlib import Path

root_dir = Path("/efs/flying_things_3d_sample/")

rgb_image_paths = sorted((root_dir / "RGB_cleanpass" / "left").glob("*.png"))
disparity_image_paths = sorted((root_dir / "disparity" / "left").glob("*.pfm"))

rgb_images = [loaders.read(str(path)) for path in rgb_image_paths]
disparity_images = [
    loaders.read(str(path))[0] for path in disparity_image_paths
]
