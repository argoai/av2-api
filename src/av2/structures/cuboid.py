# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Structure which represents ground truth annotations in R^3."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Final, List, Optional, Tuple

import numpy as np
import pandas as pd

from av2.geometry.camera.pinhole_camera import PinholeCamera
from av2.geometry.conversions import quat_to_mat
from av2.geometry.geometry import compute_interior_points_mask
from av2.geometry.se3 import SE3
from av2.rendering.color import BLUE_BGR, TRAFFIC_YELLOW1_BGR
from av2.rendering.vector import draw_line_frustum
from av2.utils.io import read_feather
from av2.utils.typing import NDArrayBool, NDArrayByte, NDArrayFloat, NDArrayInt, NDArrayObject

ORDERED_CUBOID_COL_NAMES: Final[List[str]] = [
    "tx_m",
    "ty_m",
    "tz_m",
    "length_m",
    "width_m",
    "height_m",
    "qw",
    "qx",
    "qy",
    "qz",
]


@dataclass
class Cuboid:
    """Models a cuboid annotated in a single lidar sweep.

    Reference: https://en.wikipedia.org/wiki/Cuboid

    Args:
        dst_SE3_object: Pose of the object in the destination reference frame. Translation component is in meters.
        length_m: Object extent along the x-axis in meters.
        width_m: Object extent along the y-axis in meters.
        height_m: Object extent along the z-axis in meters.
        category: Object category.
        timestamp_ns: Vehicle nanosecond timestamp. Corresponds 1-to-1 with a lidar sweep.
    """

    dst_SE3_object: SE3
    length_m: float
    width_m: float
    height_m: float
    timestamp_ns: Optional[int] = None
    category: Optional[Enum] = None

    @property
    def xyz_center_m(self) -> NDArrayFloat:
        """Cartesian coordinate center (x,y,z) in the destination reference frame."""
        return self.dst_SE3_object.translation

    @property
    def dims_lwh_m(self) -> NDArrayFloat:
        """Object extents (length,width,height) along the (x,y,z) axes respectively in meters."""
        dims_lwh: NDArrayFloat = np.stack([self.length_m, self.width_m, self.height_m])
        return dims_lwh

    @cached_property
    def vertices_m(self) -> NDArrayFloat:
        r"""Return the cuboid vertices in the destination reference frame.

            5------4
            |\\    |\\
            | \\   | \\
            6--\\--7  \\
            \\  \\  \\ \\
        l    \\  1-------0    h
         e    \\ ||   \\ ||   e
          n    \\||    \\||   i
           g    \\2------3    g
            t      width.     h
             h.               t.

        Returns:
            (8,3) array of cuboid vertices.
        """
        unit_vertices_obj_xyz_m: NDArrayFloat = np.array(
            [
                [+1, +1, +1],  # 0
                [+1, -1, +1],  # 1
                [+1, -1, -1],  # 2
                [+1, +1, -1],  # 3
                [-1, +1, +1],  # 4
                [-1, -1, +1],  # 5
                [-1, -1, -1],  # 6
                [-1, +1, -1],  # 7
            ],
        )
        dims_lwh_m = self.dims_lwh_m

        # Transform unit polygons.
        vertices_obj_xyz_m: NDArrayFloat = (dims_lwh_m / 2.0) * unit_vertices_obj_xyz_m
        vertices_dst_xyz_m = self.dst_SE3_object.transform_point_cloud(vertices_obj_xyz_m)

        # Finally, return the polygons.
        return vertices_dst_xyz_m

    def compute_interior_points(self, points_xyz_m: NDArrayFloat) -> Tuple[NDArrayFloat, NDArrayBool]:
        """Given a query point cloud, filter to points interior to the cuboid, and provide mask.

        Note: comparison is to cuboid vertices in the destination reference frame.

        Args:
            points_xyz_m: (N,3) Points to filter.

        Returns:
            The interior points and the boolean array indicating which points are interior.
        """
        vertices_dst_xyz_m = self.vertices_m
        is_interior = compute_interior_points_mask(points_xyz_m, vertices_dst_xyz_m)
        return points_xyz_m[is_interior], is_interior

    def transform(self, target_SE3_dst: SE3) -> Cuboid:
        """Apply an SE(3) transformation to the cuboid.

        Mathematically written as:
            cuboid_target = target_SE3_dst * cuboid_dst

        Args:
            target_SE3_dst: Transformation from the destination to the target reference frame.

        Returns:
            Transformed cuboid.
        """
        target_SE3_object = target_SE3_dst.compose(self.dst_SE3_object)
        return Cuboid(
            dst_SE3_object=target_SE3_object,
            length_m=self.length_m,
            width_m=self.width_m,
            height_m=self.height_m,
            category=self.category,
            timestamp_ns=self.timestamp_ns,
        )

    @classmethod
    def from_numpy(
        cls, params: NDArrayObject, category: Optional[Enum] = None, timestamp_ns: Optional[int] = None
    ) -> Cuboid:
        """Convert a set of cuboid parameters to a `Cuboid` object.

        NOTE: Category and timestamp may be optionally provided.

        Args:
            params: (N,10) Array of cuboid parameters corresponding to `ORDERED_CUBOID_COL_NAMES`.
            category: Category name of the cuboid.
            timestamp_ns: Sweep timestamp at which the cuboid was annotated or detected (in nanoseconds).

        Returns:
            Constructed cuboid.
        """
        translation = params[:3]
        length_m, width_m, height_m = params[3:6]
        quat_wxyz = params[6:10]

        rotation = quat_to_mat(quat_wxyz)
        ego_SE3_object = SE3(rotation=rotation, translation=translation)
        return cls(
            dst_SE3_object=ego_SE3_object,
            length_m=length_m,
            width_m=width_m,
            height_m=height_m,
            category=category,
            timestamp_ns=timestamp_ns,
        )


@dataclass
class CuboidList:
    """Models a list of cuboids annotated in a single lidar sweep.

    Args:
        cuboids: List of `Cuboid` objects.
    """

    cuboids: List[Cuboid]

    @property
    def xyz_center_m(self) -> NDArrayFloat:
        """Cartesian coordinate centers (x,y,z) in the destination reference frame."""
        center_xyz_m: NDArrayFloat = np.stack([cuboid.dst_SE3_object.translation for cuboid in self.cuboids])
        return center_xyz_m

    @property
    def dims_lwh_m(self) -> NDArrayFloat:
        """Object extents (length,width,height) along the (x,y,z) axes respectively in meters."""
        dims_lwh: NDArrayFloat = np.stack([cuboid.dims_lwh_m for cuboid in self.cuboids])
        return dims_lwh

    @cached_property
    def vertices_m(self) -> NDArrayFloat:
        r"""Return the cuboid vertices in the destination reference frame.
        
            5------4
            |\\    |\\
            | \\   | \\
            6--\\--7  \\
            \\  \\  \\ \\
        l    \\  1-------0    h
         e    \\ ||   \\ ||   e
          n    \\||    \\||   i
           g    \\2------3    g
            t      width.     h
             h.               t.

        Returns:
            (N,8,3) array of cuboid vertices.
        """
        vertices_m: NDArrayFloat = np.stack([cuboid.vertices_m for cuboid in self.cuboids])
        return vertices_m

    @property
    def categories(self) -> List[str]:
        """Return the object category names."""
        return [str(cuboid.category) for cuboid in self.cuboids]

    def __len__(self) -> int:
        """Return the number of cuboids."""
        return len(self.cuboids)

    def __getitem__(self, idx: int) -> Cuboid:
        """Return the cuboid at index `idx`.

        Args:
            idx: index of cuboid to return from the CuboidList.

        Returns:
            Cuboid object present at index `idx`.

        Raises:
            IndexError: if index is invalid (i.e. out of bounds).
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Attempted to access cuboid {idx}, but only indices [0,{len(self)-1}] are valid")
        return self.cuboids[idx]

    def transform(self, target_SE3_dst: SE3) -> CuboidList:
        """Apply an SE(3) transformation to each cuboid in the list.

        Mathematically written as:
            cuboid_target = target_SE3_dst * cuboid_dst

        Args:
            target_SE3_dst: Transformation from the destination to the target reference frame.

        Returns:
            Transformed cuboids.
        """
        transformed_cuboids: List[Cuboid] = []
        for cuboid in self.cuboids:
            target_SE3_object = target_SE3_dst.compose(cuboid.dst_SE3_object)
            transformed_cuboid = Cuboid(
                dst_SE3_object=target_SE3_object,
                length_m=cuboid.length_m,
                width_m=cuboid.width_m,
                height_m=cuboid.height_m,
                category=cuboid.category,
                timestamp_ns=cuboid.timestamp_ns,
            )
            transformed_cuboids.append(transformed_cuboid)
        return CuboidList(cuboids=transformed_cuboids)

    def project_to_cam(
        self,
        img: NDArrayByte,
        cam_model: PinholeCamera,
        city_SE3_ego_cam_t: Optional[SE3] = None,
        city_SE3_ego_lidar_t: Optional[SE3] = None,
    ) -> NDArrayByte:
        """Project the cuboids to the camera by clipping cuboid line segments against the camera view frustum.

        NOTE: The front face of each cuboid is drawn in yellow, and all other line segments are drawn in blue.

        Args:
            img: (H,W,3) BGR image.
            cam_model: Camera model class.
            city_SE3_ego_cam_t: City egovehicle pose at the camera timestamp.
            city_SE3_ego_lidar_t: City egovehicle pose at the lidar timestamp.

        Returns:
            (H,W,3) BGR image with projected cuboids overlaid.
        """
        # Return original image if no annotations exist.
        if len(self) == 0:
            return img

        #  Number of cuboids, number of vertices, number of dimensions in a vertex.
        cuboids_vertices_ego = self.vertices_m
        N, V, D = cuboids_vertices_ego.shape

        # Collapse first dimension to allow for vectorization.
        cuboids_vertices_ego = cuboids_vertices_ego.reshape(-1, D)
        if city_SE3_ego_cam_t is not None and city_SE3_ego_lidar_t is not None:
            _, cuboids_vertices_cam, _ = cam_model.project_ego_to_img_motion_compensated(
                cuboids_vertices_ego, city_SE3_ego_cam_t=city_SE3_ego_cam_t, city_SE3_ego_lidar_t=city_SE3_ego_lidar_t
            )
        else:
            _, cuboids_vertices_cam, _ = cam_model.project_ego_to_img(cuboids_vertices_ego)

        cuboids_vertices_cam = cuboids_vertices_cam[:, :3].reshape(N, V, D)  # Unravel collapsed dimension.

        # Compute depth of each cuboid center (mean of the cuboid's vertices).
        cuboid_centers = cuboids_vertices_cam.mean(axis=1)
        z_buffer = cuboid_centers[..., -1]

        # Sort by z-order to respect visibility in the scene.
        # i.e, closer objects cuboids should be drawn on top of farther objects.
        z_orders: NDArrayFloat = np.argsort(-z_buffer)

        cuboids_vertices_cam = cuboids_vertices_cam[z_orders]
        front_face_indices = [0, 1, 2, 3, 0]
        back_face_indices = [4, 5, 6, 7, 4]
        line_segment_indices_list = [[0, 4], [1, 5], [2, 6], [3, 7]]

        # Iterate over the vertices for each cuboid in the cuboid list.
        # Draw 6 cuboid faces in the inner loop.
        for i, cuboid_vertices_cam in enumerate(cuboids_vertices_cam):
            cuboid_front_face = cuboid_vertices_cam[front_face_indices]
            cuboid_back_face = cuboid_vertices_cam[back_face_indices]

            # Iterate over the edges of the faces.
            num_line_segments = len(cuboid_front_face) - 1
            for i in range(num_line_segments):
                draw_line_frustum(
                    img,
                    cuboid_front_face[i],
                    cuboid_front_face[i + 1],
                    cam_model=cam_model,
                    color=TRAFFIC_YELLOW1_BGR,
                )
                draw_line_frustum(
                    img,
                    cuboid_back_face[i],
                    cuboid_back_face[i + 1],
                    cam_model=cam_model,
                    color=BLUE_BGR,
                )

            # Draw the line segments that connect the front and back faces.
            for line_segment_indices in line_segment_indices_list:
                line_segment = cuboid_vertices_cam[line_segment_indices]
                draw_line_frustum(
                    img,
                    line_segment[0],
                    line_segment[1],
                    cam_model=cam_model,
                    color=BLUE_BGR,
                )
        return img

    @classmethod
    def from_feather(cls, annotations_feather_path: Path) -> CuboidList:
        """Read annotations from a feather file.

        Args:
            annotations_feather_path: Feather file path.

        Returns:
            Constructed cuboids.
        """
        data = read_feather(annotations_feather_path)

        rotation = quat_to_mat(data.loc[:, ["qw", "qx", "qy", "qz"]].to_numpy())
        translation_m = data.loc[:, ["tx_m", "ty_m", "tz_m"]].to_numpy()
        length_m = data.loc[:, "length_m"].to_numpy()
        width_m = data.loc[:, "width_m"].to_numpy()
        height_m = data.loc[:, "height_m"].to_numpy()
        category = data.loc[:, "category"].to_numpy()
        timestamp_ns = data.loc[:, "timestamp_ns"].to_numpy()
        N = len(data)

        cuboid_list: List[Cuboid] = []
        for i in range(N):
            ego_SE3_object = SE3(rotation=rotation[i], translation=translation_m[i])
            cuboid = Cuboid(
                dst_SE3_object=ego_SE3_object,
                length_m=length_m[i],
                width_m=width_m[i],
                height_m=height_m[i],
                category=category[i],
                timestamp_ns=timestamp_ns[i],
            )
            cuboid_list.append(cuboid)
        return cls(cuboids=cuboid_list)

    @classmethod
    def from_dataframe(cls, data: pd.DataFrame) -> CuboidList:
        """Read annotations from a feather file.

        Args:
            data: (N,12) Dataframe containing the cuboids and their respective parameters.

        Returns:
            Constructed cuboids.
        """
        cuboids_parameters: NDArrayFloat = data.loc[:, ORDERED_CUBOID_COL_NAMES].to_numpy()
        categories: NDArrayObject = data.loc[:, "category"].to_numpy()
        timestamps_ns: NDArrayInt = data.loc[:, "timestamp_ns"].to_numpy()

        cuboid_list = [
            Cuboid.from_numpy(params, category, timestamp_ns)
            for params, category, timestamp_ns in zip(cuboids_parameters, categories, timestamps_ns)
        ]
        return cls(cuboid_list)
