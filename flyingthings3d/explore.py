import loaders
from pathlib import Path
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np

root_dir = Path("/efs/flying_things_3d_sample/")


def get_intrinsics():
    return {
        "fx": 1050.0,
        "fy": 1050.0,
        "cx": 479.5,
        "cy": 269.5,
    }


def get_o3d_intrinsics(intrinsics):
    return o3d.camera.PinholeCameraIntrinsic(
        width=960,
        height=540,
        fx=intrinsics["fx"],
        fy=intrinsics["fy"],
        cx=intrinsics["cx"],
        cy=intrinsics["cy"],
    )


def to_o3d_depth_image(depth_image):
    return o3d.geometry.Image(depth_image)


def to_o3d_rgbd_image(rgb, depth_image):
    assert rgb.shape[:
                     2] == depth_image.shape[:
                                             2], f"shape mismatch, {rgb.shape} != {depth_image.shape}"

    rgb_img = o3d.geometry.Image(rgb)
    depth_img = o3d.geometry.Image(depth_image)
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=rgb_img,
        depth=depth_img,
        depth_scale=1.0,
        depth_trunc=1000.0,
        convert_rgb_to_intensity=False,
    )


rgb_image_paths = sorted((root_dir / "RGB_cleanpass" / "left").glob("*.png"))
disparity_image_paths = sorted((root_dir / "disparity").glob("*.pfm"))

rgb_images = [loaders.read(str(path)) for path in rgb_image_paths]
disparity_images = [loaders.read(str(path)) for path in disparity_image_paths]
camera_data = loaders.readCameraData(root_dir / "camera_data.txt")
camera_data_left = [data["left"] for data in camera_data]
camera_data_right = [data["right"] for data in camera_data]
intrinsics = get_intrinsics()

# Blender coordinate frame is different from standard robotics "right hand rule" world coordinate frame:
# positive-X points to the right, positive-Y points upwards, positive-Z points "backwards", from the scene
# into the camera (right-hand rule with thumb=x, index finger=y, middle finger=z).
world_T_blender = np.array([
    [0, 0, -1, 0],
    [-1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
])

blender_T_world = np.linalg.inv(world_T_blender)


def disparity_to_depth(disparity, intrinsics, baseline=1.0):
    return (baseline * intrinsics["fx"]) / disparity


assert len(rgb_images) == len(
    disparity_images
), f"len mismatch, {len(rgb_images)} != {len(disparity_images)}"

depth_images = [
    disparity_to_depth(disparity, intrinsics) for disparity in disparity_images
]

# Create o3d visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()
# Draw world coordinate frame
world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
vis.add_geometry(world_frame)

for idx, (blender_T_cam_left, blender_T_cam_right, disparity, rgb,
          depth) in enumerate(
              zip(camera_data_left, camera_data_right, disparity_images,
                  rgb_images, depth_images)):

    zero_vector = np.array([0, 0, 0, 1])

    world_forward_vector = np.array([1, 0, 0, 1])
    world_left_vector = np.array([0, 1, 0, 1])
    world_up_vector = np.array([0, 0, 1, 1])

    world_left_cam_position = blender_T_cam_left @ blender_T_world @ zero_vector
    world_right_cam_position = blender_T_cam_right @ blender_T_world @ zero_vector

    world_left_cam_forward_world_vector = blender_T_cam_left @ blender_T_world @ world_forward_vector
    world_left_cam_up_world_vector = blender_T_cam_left @ blender_T_world @ world_up_vector
    world_left_cam_left_world_vector = blender_T_cam_left @ blender_T_world @ world_left_vector

    # Left camera position
    left_cam_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    left_cam_sphere.paint_uniform_color([0, 0, 0])
    left_cam_sphere.translate(world_left_cam_position[:3])

    # Left camera forward world
    left_cam_forward_world_sphere = o3d.geometry.TriangleMesh.create_sphere(
        radius=0.2)
    left_cam_forward_world_sphere.paint_uniform_color([1, 0, 0])
    left_cam_forward_world_sphere.translate(
        world_left_cam_forward_world_vector[:3])

    # Left camera up world
    left_cam_up_world_sphere = o3d.geometry.TriangleMesh.create_sphere(
        radius=0.2)
    left_cam_up_world_sphere.paint_uniform_color([0, 0, 1])
    left_cam_up_world_sphere.translate(world_left_cam_up_world_vector[:3])

    # Left camera left world
    left_cam_left_world_sphere = o3d.geometry.TriangleMesh.create_sphere(
        radius=0.2)
    left_cam_left_world_sphere.paint_uniform_color([0, 1, 0])
    left_cam_left_world_sphere.translate(world_left_cam_left_world_vector[:3])

    # Right camera position
    right_cam_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    right_cam_sphere.paint_uniform_color([0.5, 0.5, 0.5])
    right_cam_sphere.translate(world_right_cam_position[:3])

    # Draw line between camera positions
    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector(
        np.vstack((world_left_cam_position[:3], world_right_cam_position[:3])))
    line.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
    line.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0], [0, 0, 1]]))

    # Convert depth to o3d PointCloud
    # plt.imshow(depth)
    # plt.colorbar()
    # plt.show()
    o3d_depth_image = to_o3d_rgbd_image(rgb, depth)
    o3d_intrinsics = get_o3d_intrinsics(intrinsics)

    standard_frame_extrinsics = np.array([
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ])
    # Convert depth image to point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        o3d_depth_image, o3d_intrinsics, standard_frame_extrinsics)
    pcd = pcd.transform(blender_T_cam_left @ blender_T_world)

    # Add meshes to visualizer
    vis.add_geometry(left_cam_sphere)
    vis.add_geometry(left_cam_forward_world_sphere)
    vis.add_geometry(left_cam_up_world_sphere)
    vis.add_geometry(left_cam_left_world_sphere)

    vis.add_geometry(right_cam_sphere)
    vis.add_geometry(line)
    vis.add_geometry(pcd)

# Show visualization
vis.run()

# Make N subplots, one for each rgb image
fig, axes = plt.subplots(nrows=len(rgb_images), ncols=3)

# Show RGB images in column 0
for i, ax in enumerate(axes[:, 0]):
    ax.imshow(rgb_images[i])
    ax.set_title(f"rgb image {i}")
    ax.axis("off")

# Show disparity images in column 1
for i, ax in enumerate(axes[:, 1]):
    im = ax.imshow(disparity_images[i])
    ax.set_title(f"disparity image {i}")
    ax.axis("off")
    # Add colorbar to current axis
    fig.colorbar(im, ax=ax)

# Show depth images in column 2
for i, ax in enumerate(axes[:, 2]):
    depth_image = depth_images[i]
    im = ax.imshow(depth_image)
    ax.set_title(f"depth image {i}")
    ax.axis("off")
    # Add colorbar to current axis
    fig.colorbar(im, ax=ax)

plt.show()
