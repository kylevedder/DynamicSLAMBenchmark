from pathlib import Path
import pickle
from datastructures import SE3, PointCloud
import numpy as np


class KubricSequence():

    def __init__(self, data_file: Path):
        self.data = self._load_pkl(data_file)

    @property
    def intrinsics(self):
        focal_length = self.data["camera"]["focal_length"] *2
        input_size_x = self.data["metadata"]["width"]
        input_size_y = self.data["metadata"]["height"]
        sensor_width = self.data["camera"]["sensor_width"]

        f_x = focal_length / sensor_width * (input_size_x / 2)
        f_y = f_x * input_size_x / input_size_y

        return {
            "fx": f_x,
            "fy": f_y,
            "cx": input_size_x / 2,
            "cy": input_size_y / 2,
        }

    def __len__(self):
        # We only have N - 1 frames where the pose before and after is known, so we only have N - 1 samples.
        return self.data["metadata"]['num_frames'] - 1

    def _load_pkl(self, pkl_file: Path):
        assert pkl_file.exists(), f"pkl_file {pkl_file} does not exist"
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
        return data

    def _project_to_camera_frame(self, image_space_positions, depth_image):
        # As noted on the FlyingThings3D page, the focal lengths should actually be negative,
        # but they do not negate them for consistency; many packages like Open3D expect positive.
        focal_array = np.array([self.intrinsics["fx"], self.intrinsics["fy"]])

        center_array = np.array([self.intrinsics["cx"], self.intrinsics["cy"]])

        # image space = focal * camera space / depth
        # image space * depth / focal = camera space

        image_space_centered = image_space_positions - center_array[None,
                                                                    None, :]
        return image_space_centered * depth_image / focal_array[None, None, :]

    def __getitem__(self, idx):
        rgb = self.data["rgb_video"][idx]
        depth = self.data["depth_video"][idx]

        position = self.data["camera"]["positions"][idx]
        quaternion = self.data["camera"]["quaternions"][idx]

        pose = SE3.from_rot_w_x_y_z_translation_x_y_z(*quaternion, *position)
        pointcloud_pinhole = PointCloud.from_pinhole_depth_image(depth[:, :, 0],
                                                 self.intrinsics)
        pointcloud_ball = PointCloud.from_ball_depth_image(depth[:, :, 0],
                                                      self.intrinsics)
        return {
            "pose": pose,
            "pointcloud": pointcloud_ball,
            "pointcloud_pinhole": pointcloud_pinhole,
            "rgb": rgb
        }


class KubricSequenceLoader():

    def __init__(self, root_dir: Path) -> None:
        self.files = sorted(root_dir.glob("*.pkl"))
        assert len(
            self.files
        ) > 0, f"root_dir {root_dir} does not contain any .pkl files"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> KubricSequence:
        return KubricSequence(self.files[idx])
