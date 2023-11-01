import numpy as np
import open3d as o3d
from typing import List, Tuple, Union, Optional, Callable, Any

import matplotlib.pyplot as plt


def sample_unit_vectors(num_samples: int, magnitude=1.0):
    """
    Samples unit vectors uniformly from the unit sphere.

    Args:
    - num_samples (int): The number of vectors to sample.

    Returns:
    - np.ndarray: An array of shape (num_samples, 3) containing unit vectors.
    """
    # Randomly sample points in 3D space
    points = np.random.randn(num_samples, 3)

    # Normalize each point to lie on the unit sphere
    unit_vectors = points / np.linalg.norm(points, axis=1)[:, np.newaxis]

    return unit_vectors * magnitude


def create_sphere_at_point(point, radius):
    """
    Creates an Open3D sphere at a specified 3D point.

    Args:
    - point (np.ndarray): A numpy array of shape (3,) representing the 3D point.
    - radius (float): The radius of the sphere.

    Returns:
    - o3d.geometry.TriangleMesh: An Open3D mesh object representing the sphere.
    """
    if isinstance(point, list):
        point = np.array(point)

    # Ensure point is a valid 3D point
    assert point.shape == (3, ), "Point must be a 3D vector."

    # Create a sphere at the origin
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)

    # Translate the sphere to the specified point
    sphere.translate(point)

    return sphere


def create_lineset_from_array(
    array: np.ndarray, color: Tuple[float, float, float] = (0, 0, 0)
) -> o3d.geometry.LineSet:
    """
    Creates an Open3D line set from an Nx2x3 numpy array, with a specified color for all lines.

    Args:
    - array (np.ndarray): An Nx2x3 numpy array representing N lines. Each line is
      represented by two points in 3D space.
    - color (Tuple[float, float, float]): RGB color for each line. Default is black (0, 0, 0).

    Returns:
    - o3d.geometry.LineSet: An Open3D LineSet object.
    """
    # Ensure the input is an Nx2x3 array
    assert array.ndim == 3 and array.shape[1:] == (
        2, 3), "Array must be of shape Nx2x3."

    num_lines = array.shape[0]

    # Create a color list with the specified color for each line
    colors = [color for _ in range(num_lines)]

    # Reshape the array to have all points in a list
    points = array.reshape(-1, 3)

    # Create lines using indices
    lines = [[2 * i, 2 * i + 1] for i in range(num_lines)]

    # Create an Open3D LineSet object
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def setup_and_show_visualizer(geometries: list):
    """
    Sets up an Open3D visualizer and displays the provided geometries.

    Args:
    - geometries (list): A list of Open3D geometry objects (e.g., point clouds, meshes).
    """
    # Create a visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add each geometry to the visualizer
    for geometry in geometries:
        if geometry is None:
            continue
        if isinstance(geometry, list):
            for g in geometry:
                vis.add_geometry(g)
        else:
            vis.add_geometry(geometry)

    # Run the visualizer
    vis.run()

    # Destroy the window after closing
    vis.destroy_window()


def generate_samples(num_samples: int, gt_magnitude : float, error_magnitude : float):
    # set random seed
    np.random.seed(0)

    # Sample unit vectors
    gt_vectors = sample_unit_vectors(num_samples, magnitude=gt_magnitude)
    est_vectors = sample_unit_vectors(num_samples, magnitude=error_magnitude) + gt_vectors
    origin_points = np.zeros((num_samples, 3))

    return gt_vectors, est_vectors, origin_points


def visualize_samples(gt_vectors, est_vectors, origin_points):
    origin_lineset = create_lineset_from_array(
        np.stack([origin_points, gt_vectors], axis=1))
    error_lineset = create_lineset_from_array(np.stack(
        [gt_vectors, est_vectors], axis=1),
                                              color=(1, 0, 0))

    origin_sphere = None  #create_sphere_at_point([0, 0, 0], radius=0.1)

    # Convert these vectors to Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(gt_vectors)

    # Visualize the point cloud
    setup_and_show_visualizer(
        [point_cloud, origin_lineset, error_lineset, origin_sphere])


def compute_error(pc1, pc2_gt, pc2_est, verbose: bool = False):
    """
    Computes the error between two point clouds.

    Args:
    - pc1 (np.ndarray): An Nx3 numpy array representing a point cloud.
    - pc2_gt (np.ndarray): An Nx3 numpy array representing a point cloud.
    - pc2_est (np.ndarray): An Nx3 numpy array representing a point cloud.

    Returns:
    - float: The error between the two point clouds.
    """
    # Ensure the point clouds have the same shape
    assert pc1.shape == pc2_gt.shape == pc2_est.shape, "Point clouds must have the same shape."

    def augment_flow(flow):
        """
        Adds a fourth dimension of 1s to the flow.
        """

        return np.hstack([flow, np.ones((flow.shape[0], 1))])

    def unaugment_flow(flow_aug):
        """
        Removes the fourth dimension from the flow.
        """

        return flow_aug[:, :3]

    raw_gt_flow = pc2_gt - pc1
    raw_est_flow = pc2_est - pc1

    raw_gt_flow_aug = augment_flow(raw_gt_flow)
    raw_est_flow_aug = augment_flow(raw_est_flow)

    # Compute the norms of gt_flow_aug
    gt_norms = np.linalg.norm(raw_gt_flow_aug, axis=1)

    # Normalize gt_flow and est_flow by the gt_norm
    normed_gt_flow = unaugment_flow(raw_gt_flow_aug / gt_norms[:, np.newaxis])
    normed_est_flow = unaugment_flow(raw_est_flow_aug /
                                     gt_norms[:, np.newaxis])

    if verbose:
        print("raw_gt_flow", raw_gt_flow)
        print("raw_est_flow", raw_est_flow)

        print("raw_gt_flow_aug", raw_gt_flow_aug)
        print("raw_est_flow_aug", raw_est_flow_aug)

        print("gt_norms", gt_norms)

        print("normed_gt_flow", normed_gt_flow)
        print("normed_est_flow", normed_est_flow)

    # Compute the EPE between the two flows, now that they are normalized
    error = np.linalg.norm(normed_gt_flow - normed_est_flow, axis=1)
    return error.mean()


num_samples = 100

gt_magnitudes = [0, 0.001, 0.05, 0.2, 0.5, 1, 2, 5, 10]
est_magnitudes = [0.1, 0.2, 0.5, 1, 2, 5]

color_map = plt.get_cmap("tab10")

errors = dict()


for est_magnitude in est_magnitudes:
    for gt_magnitude in gt_magnitudes:
        gt_vectors, est_vectors, origin_points = generate_samples(
            num_samples, gt_magnitude, est_magnitude)
        errors[(gt_magnitude, est_magnitude)] = compute_error(origin_points, gt_vectors, est_vectors)
        # visualize_samples(gt_vectors, est_vectors, origin_points)


for idx, est_magnitude in enumerate(est_magnitudes):

    color = color_map(idx)

    gt_errors = [errors[(gt_magnitude, est_magnitude)] for gt_magnitude in gt_magnitudes]

    plt.plot(gt_magnitudes, gt_errors, label=f"Metric Space error = {est_magnitude}", color=color)

    # Draw a horizontal line at the est_magnitude
    plt.axhline(est_magnitude, linestyle="--", alpha=0.5, color=color)

plt.legend()
plt.xlabel("Ground truth vector distance")
# Set X ticks to be the same as the gt_magnitudes
plt.xticks(gt_magnitudes)
# Format the X ticks to be whole numbers if possible
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}" if x.is_integer() else f"{x:.2f}"))
# Turn the X ticks vertical
plt.xticks(rotation=90)

plt.ylabel("Space-Time Normalized EPE")

# Light grid lines
plt.grid(alpha=0.2)


plt.show()
