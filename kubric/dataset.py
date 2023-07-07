from pathlib import Path
import pickle
from datastructures import SE3, PointCloud
import numpy as np


class KubricSequence():

    def __init__(self, data_file: Path):
        self.data = self._load_pkl(data_file)

    @property
    def intrinsics(self):
        focal_length = self.data["camera"]["focal_length"]
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

    def unproject(coord, cam, depth):
        """Unproject points.

        Args:
            coord: Points in 2D coordinates.  it has shape [num_points, 2].  Coord is in
            integer (y,x) because of the way meshgrid happens.
            cam: The camera parameters, as returned by kubric.  'matrix_world' and
            'intrinsics' have a leading axis num_frames.
            depth: Depth map for the scene.

        Returns:
            Image coordinates in 3D.
        """
        shp = np.array(depth.shape)
        idx = coord[:, 0] * shp[1] + coord[:, 1]
        coord = tf.cast(coord[..., ::-1], tf.float32)
        shp = tf.cast(shp[1::-1], tf.float32)[tf.newaxis, ...]

        # Need to convert from pixel to raster coordinate.
        projected_pt = (coord + 0.5) / shp

        projected_pt = tf.concat(
            [
                projected_pt,
                tf.ones_like(projected_pt[:, -1:]),
            ],
            axis=-1,
        )

        camera_plane = projected_pt @ tf.linalg.inv(
            tf.transpose(cam['intrinsics']))
        camera_ball = camera_plane / tf.sqrt(
            tf.reduce_sum(
                tf.square(camera_plane),
                axis=1,
                keepdims=True,
            ), )
        camera_ball *= tf.gather(tf.reshape(depth, [-1]), idx)[:, tf.newaxis]

        camera_ball = tf.concat(
            [
                camera_ball,
                tf.ones_like(camera_plane[:, 2:]),
            ],
            axis=1,
        )
        points_3d = camera_ball @ tf.transpose(cam['matrix_world'])
        return points_3d[:, :3] / points_3d[:, 3:]

    def _make_point_cloud(self, depth_image) -> PointCloud:
        # X positions repeated for each row
        x_positions = np.tile(np.arange(depth_image.shape[1]),
                              (depth_image.shape[0], 1))
        # Y positions repeated for each column
        y_positions = np.tile(np.arange(depth_image.shape[0]),
                              (depth_image.shape[1], 1)).T

        # Stack the x and y positions into a 3D array of shape (H, W, 2)
        image_space_input_positions = np.stack([x_positions, y_positions],
                                               axis=2).astype(np.float32)

        camera_frame_input_positions = self._project_to_camera_frame(
            image_space_input_positions, depth_image)

        camera_frame_input_positions_xyz = np.concatenate([
            depth_image,
            camera_frame_input_positions,
        ],
                                                          axis=2)

        return PointCloud(camera_frame_input_positions_xyz.reshape(-1, 3))

    def __getitem__(self, idx):
        rgb = self.data["rgb_video"][idx]
        depth = self.data["depth_video"][idx]

        position = self.data["camera"]["positions"][idx]
        quaternion = self.data["camera"]["quaternions"][idx]

        pose = SE3.from_rot_w_x_y_z_translation_x_y_z(*quaternion, *position)
        pointcloud = PointCloud.from_depth_image(depth[:, :, 0],
                                                 self.intrinsics)
        return {"pose": pose, "pointcloud": pointcloud}


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
