import numpy as np
import open3d as o3d
from .se3 import SE3


def to_fixed_array(array: np.ndarray,
                   max_len: int,
                   pad_val=np.nan) -> np.ndarray:
    if len(array) > max_len:
        np.random.RandomState(len(array)).shuffle(array)
        sliced_pts = array[:max_len]
        return sliced_pts
    else:
        pad_tuples = [(0, max_len - len(array))]
        for _ in range(array.ndim - 1):
            pad_tuples.append((0, 0))
        return np.pad(array, pad_tuples, constant_values=pad_val)


def from_fixed_array(array: np.ndarray) -> np.ndarray:
    if isinstance(array, np.ndarray):
        if len(array.shape) == 2:
            check_array = array[:, 0]
        elif len(array.shape) == 1:
            check_array = array
        else:
            raise ValueError(f'unknown array shape {array.shape}')
        are_valid_points = np.logical_not(np.isnan(check_array))
        are_valid_points = are_valid_points.astype(bool)
    else:
        import torch
        if len(array.shape) == 2:
            check_array = array[:, 0]
        elif len(array.shape) == 1:
            check_array = array
        else:
            raise ValueError(f'unknown array shape {array.shape}')
        are_valid_points = torch.logical_not(torch.isnan(check_array))
        are_valid_points = are_valid_points.bool()
    return array[are_valid_points]


def make_image_pixel_coordinate_grid(image_shape: tuple) -> np.ndarray:
    assert len(
        image_shape) == 2, f'image_shape must be a 2-tuple, got {image_shape}'
    # X positions repeated for each row
    x_positions = np.tile(np.arange(image_shape[1]), (image_shape[0], 1))
    # Y positions repeated for each column
    y_positions = np.tile(np.arange(image_shape[0]), (image_shape[1], 1)).T

    image_coordinates = np.stack([x_positions, y_positions],
                                 axis=2).astype(np.float32).reshape(-1, 2)
    return image_coordinates


def camera_to_world_coordiantes(points: np.ndarray) -> np.ndarray:
    world_T_camera = np.array([
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
    ])
    return (world_T_camera @ points.T).T


class PointCloud():

    def __init__(self, points: np.ndarray) -> None:
        assert points.ndim == 2, f'points must be a 2D array, got {points.ndim}'
        assert points.shape[
            1] == 3, f'points must be a Nx3 array, got {points.shape}'
        self.points = points

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, PointCloud):
            return False
        return np.allclose(self.points, o.points)

    def __len__(self):
        return self.points.shape[0]

    def __repr__(self) -> str:
        return f'PointCloud with {len(self)} points'

    def __getitem__(self, idx):
        return self.points[idx]

    @staticmethod
    def from_pinhole_points_and_depth(image_coordinates: np.ndarray,
                                      image_coordinate_depths: np.ndarray,
                                      intrinsics: dict) -> 'PointCloud':
        assert image_coordinates.ndim == 2, f'image_space_points must be a 2D array, got {image_coordinates.ndim}'
        assert image_coordinates.shape[
            1] == 2, f'image_space_points must be a Nx2 array, got {image_coordinates.shape}'

        assert image_coordinate_depths.ndim == 2, f'depth must be a 2D array, got {image_coordinate_depths.ndim}'
        assert image_coordinate_depths.shape[
            1] == 1, f'depth must be a Nx1 array, got {image_coordinate_depths.shape}'

        assert image_coordinates.shape[0] == image_coordinate_depths.shape[
            0], f'number of points in image_coordinates {image_coordinates.shape[0]} must match number of points in image_coordinate_depths {image_coordinate_depths.shape[0]}'
        print("intrinsics", intrinsics)

        # Standard camera intrinsics matrix
        K = np.array([
            [intrinsics["fx"], 0, intrinsics["cx"]],
            [0, intrinsics["fy"], intrinsics["cy"]],
            [0, 0, 1],
        ])

        # These points are at the pixel locations of the image.
        image_coordinate_points = np.concatenate(
            [image_coordinates,
             np.ones((len(image_coordinates), 1))], axis=1)

        # Camera plane is the plane of ray points with a depth of 1 in the camera coordinate frame.
        camera_plane_points = image_coordinate_points @ np.linalg.inv(K.T)

        # Multiplying by the depth scales the ray of each point in the camera coordinate frame to
        # the distance measured by the depth image.
        world_points_camera_coords = camera_plane_points * image_coordinate_depths

        return PointCloud(
            camera_to_world_coordiantes(world_points_camera_coords))

    @staticmethod
    def from_pinhole_depth_image(depth: np.ndarray,
                                 intrinsics: dict) -> 'PointCloud':
        assert depth.ndim == 2, f'depth must be a 2D array, got {depth.ndim}'

        image_coordinates = make_image_pixel_coordinate_grid(depth.shape)

        image_coordinate_depths = depth.reshape(-1, 1)

        return PointCloud.from_pinhole_points_and_depth(
            image_coordinates, image_coordinate_depths, intrinsics)

    @staticmethod
    def from_field_of_view_ndc_points_and_depth(
            ndc_coordinates: np.ndarray, ndc_coordinate_depths: np.ndarray,
            fx: float, fy: float) -> 'PointCloud':

        assert ndc_coordinates.ndim == 2, f'ndc_coordinates must be a 2D array, got {ndc_coordinates.ndim}'
        assert (ndc_coordinates <= 1.0).all(
        ), f'ndc_coordinates must be in NDC space (<= 1), got {ndc_coordinates}'
        assert (ndc_coordinates >= 0.0).all(
        ), f'ndc_coordinates must be in NDC space (>= 0), got {ndc_coordinates}'

        ndc_coordinate_points = np.concatenate(
            [ndc_coordinates,
             np.ones((len(ndc_coordinates), 1))], axis=1)

        # Camera intrinsics matrix converted to Normalized Device Coordinates (NDC)
        K = np.array([
            [fx, 0, 0.5],
            [0, fy, 0.5],
            [0, 0, 1],
        ])

        # Camera plane is the plane of ray points with a depth of 1 in the camera coordinate frame.
        camera_plane_points = ndc_coordinate_points @ np.linalg.inv(K.T)

        # Normalize the ray vectors to be unit length to form the camera plane.
        camera_ball_points = camera_plane_points / np.linalg.norm(
            camera_plane_points, axis=1, keepdims=True)

        # Multiplying by the depth scales the ray of each point in the camera coordinate frame to
        # the distance measured by the depth image.
        world_points_camera_coords = camera_ball_points * ndc_coordinate_depths

        return PointCloud(
            camera_to_world_coordiantes(world_points_camera_coords))

    @staticmethod
    def from_field_of_view_points_and_depth(image_coordinates: np.ndarray,
                                            depths: np.ndarray,
                                            image_shape: tuple,
                                            intrinsics: dict) -> 'PointCloud':
        assert image_coordinates.ndim == 2, f'image_space_points must be a 2D array, got {image_coordinates.ndim}'
        assert image_coordinates.shape[
            1] == 2, f'image_space_points must be a Nx2 array, got {image_coordinates.shape}'

        assert depths.ndim == 2, f'depth must be a 2D array, got {depths.ndim}'
        assert depths.shape[
            1] == 1, f'depth must be a Nx1 array, got {depths.shape}'

        assert image_coordinates.shape[0] == depths.shape[
            0], f'number of points in image_coordinates {image_coordinates.shape[0]} must match number of points in image_coordinate_depths {depths.shape[0]}'
        print("intrinsics", intrinsics)
        assert len(image_shape
                   ) == 2, f'image_shape must be a 2-tuple, got {image_shape}'

        fx = intrinsics["fx"] / image_shape[1]
        fy = intrinsics["fy"] / image_shape[0]

        # Convert from pixels to raster space with the + 0.5, then to NDC space
        ndc_coordinates = (image_coordinates +
                           0.5) / np.array(image_shape)[None, :]

        return PointCloud.from_field_of_view_ndc_points_and_depth(
            ndc_coordinates, depths, fx, fy)

    @staticmethod
    def from_field_of_view_depth_image(depth: np.ndarray,
                                       intrinsics: dict) -> 'PointCloud':
        """
        See: https://link.springer.com/article/10.1007/pl00013269
        Difference is the conversion to NDC and normalization of the camera plane points
          to unit length before scaling by depth.
        """
        assert depth.ndim == 2, f'depth must be a 2D array, got {depth.ndim}'
        image_coordinates = make_image_pixel_coordinate_grid(depth.shape)
        depths = depth.reshape(-1, 1)
        return PointCloud.from_field_of_view_points_and_depth(
            image_coordinates, depths, depth.shape, intrinsics)

    @staticmethod
    def from_depth_image_o3d(depth: np.ndarray,
                             intrinsics: dict) -> 'PointCloud':
        standard_frame_extrinsics = np.array([
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
        ])
        o3d_depth = o3d.geometry.Image(depth)
        o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic()
        o3d_intrinsics.set_intrinsics(width=depth.shape[1],
                                      height=depth.shape[0],
                                      fx=intrinsics["fx"],
                                      fy=intrinsics["fy"],
                                      cx=intrinsics["cx"],
                                      cy=intrinsics["cy"])
        o3d_pc = o3d.geometry.PointCloud.create_from_depth_image(
            o3d_depth, o3d_intrinsics, standard_frame_extrinsics)
        return PointCloud(np.asarray(o3d_pc.points))

    def transform(self, se3: SE3) -> 'PointCloud':
        assert isinstance(se3, SE3)
        return PointCloud(se3.transform_points(self.points))

    def translate(self, translation: np.ndarray) -> 'PointCloud':
        assert translation.shape == (3, )
        return PointCloud(self.points + translation)

    def flow(self, flow: np.ndarray) -> 'PointCloud':
        assert flow.shape == self.points.shape, f"flow shape {flow.shape} must match point cloud shape {self.points.shape}"
        return PointCloud(self.points + flow)

    def to_fixed_array(self, max_points: int) -> np.ndarray:
        return to_fixed_array(self.points, max_points)

    def matched_point_diffs(self, other: 'PointCloud') -> np.ndarray:
        assert len(self) == len(other)
        return self.points - other.points

    def matched_point_distance(self, other: 'PointCloud') -> np.ndarray:
        assert len(self) == len(other)
        return np.linalg.norm(self.matched_point_diffs(other), axis=1)

    @staticmethod
    def from_fixed_array(points) -> 'PointCloud':
        return PointCloud(from_fixed_array(points))

    def to_array(self) -> np.ndarray:
        return self.points

    def mask_points(self, mask: np.ndarray) -> 'PointCloud':
        assert isinstance(mask, np.ndarray)
        assert mask.ndim == 1
        if mask.dtype == np.bool:
            assert mask.shape[0] == len(self)
        else:
            in_bounds = mask < len(self)
            assert np.all(
                in_bounds
            ), f"mask values must be in bounds, got {(~in_bounds).sum()} indices not in bounds out of {len(self)} points"

        return PointCloud(self.points[mask])

    def within_region_mask(self, x_min, x_max, y_min, y_max, z_min,
                           z_max) -> np.ndarray:
        mask = np.logical_and(self.points[:, 0] < x_max,
                              self.points[:, 0] > x_min)
        mask = np.logical_and(mask, self.points[:, 1] < y_max)
        mask = np.logical_and(mask, self.points[:, 1] > y_min)
        mask = np.logical_and(mask, self.points[:, 2] < z_max)
        mask = np.logical_and(mask, self.points[:, 2] > z_min)
        return mask

    def within_region(self, x_min, x_max, y_min, y_max, z_min,
                      z_max) -> 'PointCloud':
        mask = self.within_region_mask(x_min, x_max, y_min, y_max, z_min,
                                       z_max)
        return self.mask_points(mask)

    @property
    def shape(self) -> tuple:
        return self.points.shape

    def to_o3d(self):
        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.points))
