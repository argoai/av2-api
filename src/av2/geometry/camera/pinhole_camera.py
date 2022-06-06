# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Implements a pinhole camera interface."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Tuple, Union

import numpy as np

import av2.geometry.geometry as geometry_utils
import av2.utils.io as io_utils
from av2.geometry.se3 import SE3
from av2.utils.typing import NDArrayBool, NDArrayFloat, NDArrayInt


@dataclass(frozen=True)
class Intrinsics:
    """Models a camera intrinsic matrix.

    Args:
        fx_px: Horizontal focal length in pixels.
        fy_px: Vertical focal length in pixels.
        cx_px: Horizontal focal center in pixels.
        cy_px: Vertical focal center in pixels.
        width_px: Width of image in pixels.
        height_px: Height of image in pixels.
    """

    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    width_px: int
    height_px: int

    @cached_property
    def K(self) -> NDArrayFloat:
        """Camera intrinsic matrix."""
        K: NDArrayFloat = np.eye(3, dtype=float)
        K[0, 0] = self.fx_px
        K[1, 1] = self.fy_px
        K[0, 2] = self.cx_px
        K[1, 2] = self.cy_px
        return K


@dataclass(frozen=True)
class PinholeCamera:
    """Parameterizes a pinhole camera with zero skew.

    Args:
        ego_SE3_cam: pose of camera in the egovehicle frame (inverse of extrinsics matrix).
        intrinsics: `Intrinsics` object containing intrinsic parameters and image dimensions.
        cam_name: sensor name that camera parameters correspond to.
    """

    ego_SE3_cam: SE3
    intrinsics: Intrinsics
    cam_name: str

    @property
    def width_px(self) -> int:
        """Return the width of the image in pixels."""
        return self.intrinsics.width_px

    @property
    def height_px(self) -> int:
        """Return the height of the image in pixels."""
        return self.intrinsics.height_px

    @cached_property
    def extrinsics(self) -> NDArrayFloat:
        """Return the camera extrinsics."""
        return self.ego_SE3_cam.inverse().transform_matrix

    @classmethod
    def from_feather(cls, log_dir: Path, cam_name: str) -> PinholeCamera:
        """Create a pinhole camera model from a feather file.

        Note: Data is laid out with sensor names along row dimension, and columns are sensor attribute data.

        Args:
            log_dir: path to a log directory containing feather files w/ calibration info.
            cam_name: name of the camera.

        Returns:
            A new PinholeCamera object, containing camera extrinsics and intrinsics.
        """
        intrinsics_path = log_dir / "calibration" / "intrinsics.feather"
        intrinsics_df = io_utils.read_feather(intrinsics_path).set_index("sensor_name")
        params = intrinsics_df.loc[cam_name]
        intrinsics = Intrinsics(
            fx_px=params["fx_px"],
            fy_px=params["fy_px"],
            cx_px=params["cx_px"],
            cy_px=params["cy_px"],
            width_px=int(params["width_px"]),
            height_px=int(params["height_px"]),
        )
        sensor_name_to_pose = io_utils.read_ego_SE3_sensor(log_dir)
        return cls(
            ego_SE3_cam=sensor_name_to_pose[cam_name],
            intrinsics=intrinsics,
            cam_name=cam_name,
        )

    def cull_to_view_frustum(self, uv: NDArrayFloat, points_cam: NDArrayFloat) -> NDArrayBool:
        """Cull 3d points to camera view frustum.

        Given a set of coordinates in the image plane and corresponding points
        in the camera coordinate reference frame, determine those points
        that have a valid projection into the image. 3d points with valid
        projections have x coordinates in the range [0,width_px-1], y-coordinates
        in the range [0,height_px-1], and a positive z-coordinate (lying in
        front of the camera frustum).

        Ref: https://en.wikipedia.org/wiki/Hidden-surface_determination#Viewing-frustum_culling

        Args:
            uv: Numpy array of shape (N,2) representing image plane coordinates in [0,W-1] x [0,H-1]
                where (H,W) are the image height and width.
            points_cam: Numpy array of shape (N,3) representing corresponding 3d points in the camera coordinate frame.

        Returns:
            Numpy boolean array of shape (N,) indicating which points fall within the camera view frustum.
        """
        is_valid_x = np.logical_and(0 <= uv[:, 0], uv[:, 0] < self.width_px - 1)
        is_valid_y = np.logical_and(0 <= uv[:, 1], uv[:, 1] < self.height_px - 1)
        is_valid_z = points_cam[:, 2] > 0
        is_valid_points: NDArrayBool = np.logical_and.reduce([is_valid_x, is_valid_y, is_valid_z])
        return is_valid_points

    def project_ego_to_img(
        self, points_ego: NDArrayFloat, remove_nan: bool = False
    ) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayBool]:
        """Project a collection of 3d points (provided in the egovehicle frame) to the image plane.

        Args:
            points_ego: numpy array of shape (N,3) representing points in the egovehicle frame.
            remove_nan: whether to remove coordinates that project to invalid (NaN) values.

        Returns:
            uv: image plane coordinates, as Numpy array of shape (N,2).
            points_cam: camera frame coordinates as Numpy array of shape (N,3) representing
            is_valid_points: boolean indicator of valid cheirality and within image boundary, as
                boolean Numpy array of shape (N,).
        """
        # convert cartesian to homogeneous coordinates.
        points_ego_hom = geometry_utils.cart_to_hom(points_ego)
        points_cam: NDArrayFloat = self.extrinsics @ points_ego_hom.T

        # remove bottom row of all 1s.
        uv = self.intrinsics.K @ points_cam[:3, :]
        uv = uv.T
        points_cam = points_cam.T

        if remove_nan:
            uv, points_cam = remove_nan_values(uv, points_cam)

        uv = uv[:, :2] / uv[:, 2].reshape(-1, 1)
        is_valid_points = self.cull_to_view_frustum(uv, points_cam)

        return uv, points_cam, is_valid_points

    def project_cam_to_img(
        self, points_cam: NDArrayFloat, remove_nan: bool = False
    ) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayBool]:
        """Project a collection of 3d points in the camera reference frame to the image plane.

        Args:
            points_cam: numpy array of shape (N,3) representing points in the egovehicle frame.
            remove_nan: whether to remove coordinates that project to invalid (NaN) values.

        Returns:
            uv: image plane coordinates, as Numpy array of shape (N,2).
            points_cam: camera frame coordinates as Numpy array of shape (N,3) representing
            is_valid_points: boolean indicator of valid cheirality and within image boundary, as
                boolean Numpy array of shape (N,).
        """
        points_cam = points_cam.transpose()
        uv: NDArrayFloat = self.intrinsics.K @ points_cam
        uv = uv.transpose()
        points_cam = points_cam.transpose()

        if remove_nan:
            uv, points_cam = remove_nan_values(uv, points_cam)

        uv = uv[:, :2] / uv[:, 2].reshape(-1, 1)
        is_valid_points = self.cull_to_view_frustum(uv, points_cam)

        return uv, points_cam, is_valid_points

    def project_ego_to_img_motion_compensated(
        self,
        points_lidar_time: NDArrayFloat,
        city_SE3_ego_cam_t: SE3,
        city_SE3_ego_lidar_t: SE3,
    ) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayBool]:
        """Project points in the ego frame to the image with motion compensation.

        Because of the high frame rate, motion compensation's role between the
        sensors is not very significant, moving points only by millimeters
        to centimeters. If the vehicle is moving at 25 miles per hour, equivalent
        to 11 meters/sec, then in 17 milliseconds (the max time between a lidar sweep
        and camera image capture) we should be able to move up to 187 millimeters.

        This can be verified in practice as the mean_change:
            mean_change = np.amax(points_h_cam_time.T[:,:3] - points_h_lidar_time ,axis=0)

        Adjust LiDAR points for egovehicle motion. This function accepts the
        egovehicle's pose in the city map both at camera time and also at
        the LiDAR time.

        We perform the following transformation:

            pt_egovehicle_cam_t = egovehicle_cam_t_SE3_city * city_SE3_egovehicle_lidar_t * pt_egovehicle_lidar_t

        Note that both "cam_time_points_h" and "lidar_time_points_h" are 3D points in the
        vehicle coordinate frame, but captured at different times. These LiDAR points
        always live in the vehicle frame, but just in different timestamps. If we take
        a lidar point in the egovehicle frame, captured at lidar time, and bring it into
        the map at this lidar timestamp, then we know the transformation from map to
        egovehicle reference frame at the time when the camera image was captured.

        Thus, we move from egovehicle @ lidar time, to the map (which is time agnostic),
        then we move from map to egovehicle @ camera time. Now we suddenly have lidar points
        living in the egovehicle frame @ camera time.

        Args:
            points_lidar_time: Numpy array of shape (N,3)
            city_SE3_ego_cam_t: egovehicle pose when camera image was recorded.
            city_SE3_ego_lidar_t: egovehicle pose when LiDAR sweep was recorded.

        Returns:
            uv: image plane coordinates, as Numpy array of shape (N,2).
            points_cam: Numpy array of shape (N,3) representing coordinates of points within the camera frame.
            is_valid_points_cam: boolean indicator of valid cheirality and within image boundary, as
                boolean Numpy array of shape (N,).

        Raises:
            ValueError: If `city_SE3_egovehicle_cam_t` or `city_SE3_egovehicle_lidar_t` is `None`.
        """
        if city_SE3_ego_cam_t is None:
            raise ValueError("city_SE3_ego_cam_t cannot be `None`!")
        if city_SE3_ego_lidar_t is None:
            raise ValueError("city_SE3_ego_lidar_t cannot be `None`!")

        ego_cam_t_SE3_ego_lidar_t = city_SE3_ego_cam_t.inverse().compose(city_SE3_ego_lidar_t)
        points_cam_time = ego_cam_t_SE3_ego_lidar_t.transform_point_cloud(points_lidar_time)
        return self.project_ego_to_img(points_cam_time)

    @cached_property
    def right_clipping_plane(self) -> NDArrayFloat:
        """Form the right clipping plane for a camera view frustum.

        Returns:
            (4,) tuple of Hessian normal coefficients.
        """
        a, b, c, d = -self.intrinsics.fx_px, 0.0, self.width_px / 2.0, 0.0
        coeffs: NDArrayFloat = np.array([a, b, c, d]) / np.linalg.norm([a, b, c])
        return coeffs

    @cached_property
    def left_clipping_plane(self) -> NDArrayFloat:
        """Form the left clipping plane for a camera view frustum.

        Returns:
            (4,) tuple of Hessian normal coefficients.
        """
        a, b, c, d = self.intrinsics.fx_px, 0.0, self.width_px / 2.0, 0.0
        coeffs: NDArrayFloat = np.array([a, b, c, d]) / np.linalg.norm([a, b, c])
        return coeffs

    @cached_property
    def top_clipping_plane(self) -> NDArrayFloat:
        """Top clipping plane for a camera view frustum.

        Returns:
            (4,) tuple of Hessian normal coefficients.
        """
        a, b, c, d = 0.0, self.intrinsics.fx_px, self.height_px / 2.0, 0.0
        coeffs: NDArrayFloat = np.array([a, b, c, d]) / np.linalg.norm([a, b, c])
        return coeffs

    @cached_property
    def bottom_clipping_plane(self) -> NDArrayFloat:
        """Bottom clipping plane for a camera view frustum.

        Returns:
            (4,) tuple of Hessian normal coefficients.
        """
        a, b, c, d = 0.0, -self.intrinsics.fx_px, self.height_px / 2.0, 0.0
        coeffs: NDArrayFloat = np.array([a, b, c, d]) / np.linalg.norm([a, b, c])
        return coeffs

    def near_clipping_plane(self, near_clip_m: float) -> NDArrayFloat:
        """Near clipping plane for a camera view frustum.

        Args:
            near_clip_m: Near clipping plane distance in meters.

        Returns:
            (4,) tuple of Hessian normal coefficients.
        """
        a, b, c, d = 0.0, 0.0, 1.0, -near_clip_m
        coeffs: NDArrayFloat = np.array([a, b, c, d])
        return coeffs

    def frustum_planes(self, near_clip_dist: float = 0.5) -> NDArrayFloat:
        """Compute the planes enclosing the field of view (view frustum).

            Reference (1): https://en.wikipedia.org/wiki/Viewing_frustum
            Reference (2): https://en.wikipedia.org/wiki/Plane_(geometry)

            Solve for the coefficients of all frustum planes:
                ax + by + cz = d

        Args:
            near_clip_dist: Distance of the near clipping plane from the origin.

        Returns:
            (5, 4) matrix where each row corresponds to the coeffients of a plane.
        """
        left_plane = self.left_clipping_plane
        right_plane = self.right_clipping_plane

        top_plane = self.top_clipping_plane
        bottom_plane = self.bottom_clipping_plane

        near_plane = self.near_clipping_plane(near_clip_dist)
        planes: NDArrayFloat = np.stack([left_plane, right_plane, near_plane, bottom_plane, top_plane])
        return planes

    @cached_property
    def egovehicle_yaw_cam_rad(self) -> float:
        """Compute the camera's yaw, in the egovehicle frame.

        R takes the x axis to be a vector equivalent to the first column of R.
        Similarly, the y and z axes are transformed to be the second and third columns.

        Returns:
            Counter-clockwise angle from x=0 (in radians) of camera center ray, in the egovehicle frame.
        """
        egovehicle_SE3_camera = self.ego_SE3_cam
        # the third column of this rotation matrix, is the new basis vector for the z axis (pointing out of camera)
        # take its x and y components (the z component is near zero, as close to horizontal)
        new_z_axis = egovehicle_SE3_camera.rotation[:, 2]
        dx, dy, dz = new_z_axis
        egovehicle_yaw_cam_rad = np.arctan2(dy, dx)
        return float(egovehicle_yaw_cam_rad)

    @cached_property
    def fov_theta_rad(self) -> float:
        """Compute the field of view of a camera frustum to use for view frustum culling during rendering.

        Returns:
            Angular extent of camera's field of view (measured in radians).
        """
        fov_theta_rad = 2 * np.arctan(0.5 * self.width_px / self.intrinsics.fx_px)
        return float(fov_theta_rad)

    def compute_pixel_ray_directions(self, uv: Union[NDArrayFloat, NDArrayInt]) -> NDArrayFloat:
        """Given (u,v) coordinates and intrinsics, generate pixel rays in the camera coordinate frame.

        Assume +z points out of the camera, +y is downwards, and +x is across the imager.

        Args:
            uv: Numpy array of shape (N,2) with (u,v) coordinates

        Returns:
            Array of shape (N,3) with ray directions to each pixel, provided in the camera frame.

        Raises:
            ValueError: If input (u,v) coordinates are not (N,2) in shape.
            RuntimeError: If generated ray directions are not (N,3) in shape.
        """
        fx, fy = self.intrinsics.fx_px, self.intrinsics.fy_px
        img_h, img_w = self.height_px, self.width_px

        if not np.isclose(fx, fy, atol=1e-3):
            raise ValueError(f"Focal lengths in the x and y directions must match: {fx} != {fy}")

        if uv.shape[1] != 2:
            raise ValueError("Input (u,v) coordinates must be (N,2) in shape.")

        # Approximation for principal point
        px = img_w / 2
        py = img_h / 2

        u = uv[:, 0]
        v = uv[:, 1]
        num_rays = uv.shape[0]
        ray_dirs = np.zeros((num_rays, 3))
        # x center offset from center
        ray_dirs[:, 0] = u - px
        # y center offset from center
        ray_dirs[:, 1] = v - py
        ray_dirs[:, 2] = fx

        # elementwise multiplication of scalars requires last dim to match
        ray_dirs = ray_dirs / np.linalg.norm(ray_dirs, axis=1, keepdims=True)
        if ray_dirs.shape[1] != 3:
            raise RuntimeError("Ray directions must be (N,3)")
        return ray_dirs

    def scale(self, scale: float) -> PinholeCamera:
        """Scale the intrinsics and image size.

        Args:
            scale: Scaling factor.

        Returns:
            The scaled pinhole camera model.
        """
        intrinsics = Intrinsics(
            self.intrinsics.fx_px * scale,
            self.intrinsics.fy_px * scale,
            self.intrinsics.cx_px * scale,
            self.intrinsics.cy_px * scale,
            round(self.intrinsics.width_px * scale),
            round(self.intrinsics.height_px * scale),
        )
        return PinholeCamera(ego_SE3_cam=self.ego_SE3_cam, intrinsics=intrinsics, cam_name=self.cam_name)


def remove_nan_values(uv: NDArrayFloat, points_cam: NDArrayFloat) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """Remove NaN values from camera coordinates and image plane coordinates (accepts corrupt array).

    Args:
        uv: image plane coordinates, as Numpy array of shape (N,2).
        points_cam: Numpy array of shape (N,3) representing coordinates of points within the camera frame.

    Returns:
        uv_valid: subset of image plane coordinates, which contain no NaN coordinates.
        is_valid_points_cam: subset of 3d points within the camera frame, which contain no NaN coordinates.
    """
    is_u_valid = np.logical_not(np.isnan(uv[:, 0]))
    is_v_valid = np.logical_not(np.isnan(uv[:, 1]))
    is_uv_valid = np.logical_and(is_u_valid, is_v_valid)

    uv_valid = uv[is_uv_valid]
    is_valid_points_cam = points_cam[is_uv_valid]
    return uv_valid, is_valid_points_cam
