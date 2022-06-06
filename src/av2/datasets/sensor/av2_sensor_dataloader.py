# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>
"""Implements a dataloader for the Argoverse 2.0 Sensor and TbV Datasets."""

import logging
from pathlib import Path
from typing import Final, List, Optional, Tuple

import numpy as np
import pandas as pd

import av2.geometry.geometry as geometry_utils
import av2.utils.dense_grid_interpolation as dense_grid_interpolation
import av2.utils.io as io_utils
from av2.datasets.sensor.constants import RingCameras
from av2.geometry.camera.pinhole_camera import PinholeCamera
from av2.geometry.se3 import SE3
from av2.structures.cuboid import CuboidList
from av2.structures.sweep import Sweep
from av2.utils.synchronization_database import SynchronizationDB
from av2.utils.typing import NDArrayBool, NDArrayByte, NDArrayFloat

logger = logging.getLogger(__name__)

# number of nanoseconds in a single second.
NANOSECONDS_PER_SECOND: Final[int] = int(1e9)
MAX_MEASUREMENT_FREQUENCY_HZ: Final[int] = NANOSECONDS_PER_SECOND


class AV2SensorDataLoader:
    """Simple abstraction for retrieving log data, given a path to the dataset."""

    def __init__(self, data_dir: Path, labels_dir: Path) -> None:
        """Create the Sensor dataloader from a data directory and labels directory.

        Args:
            data_dir: Path to raw Argoverse 2.0 data
            labels_dir: Path to Argoverse 2.0 data labels (e.g. labels or estimated detections/tracks)

        Raises:
            ValueError: if input arguments are not Path objects.
        """
        if not isinstance(data_dir, Path) or not isinstance(labels_dir, Path):
            raise ValueError("Input arguments must be Path objects, representing paths to local directories")

        self._data_dir = data_dir
        self._labels_dir = labels_dir
        self._sdb = SynchronizationDB(str(data_dir))

    def get_log_pinhole_camera(self, log_id: str, cam_name: str) -> PinholeCamera:
        """Return a PinholeCamera parameterized by sensor pose in vehicle frame, intrinsics, and image dimensions."""
        log_dir = self._data_dir / log_id
        return PinholeCamera.from_feather(log_dir=log_dir, cam_name=cam_name)

    def get_city_SE3_ego(self, log_id: str, timestamp_ns: int) -> SE3:
        """Obtain the egovehicle's pose in the city reference frame.

        Note: the poses dataframe contains tables with columns {timestamp_ns, qw, qx, qy, qz, tx_m, ty_m, tz_m}.

        Args:
            log_id: unique ID of vehicle log.
            timestamp_ns: timestamp of sensor observation, in nanoseconds.

        Returns:
            SE(3) transformation to bring points in egovehicle frame into city frame.

        Raises:
            RuntimeError: If no recorded pose is available for the requested timestamp.
        """
        log_poses_df = io_utils.read_feather(self._data_dir / log_id / "city_SE3_egovehicle.feather")
        pose_df = log_poses_df.loc[log_poses_df["timestamp_ns"] == timestamp_ns]

        if len(pose_df) == 0:
            raise RuntimeError("Pose was not available for the requested timestamp.")

        city_SE3_ego = convert_pose_dataframe_to_SE3(pose_df)
        return city_SE3_ego

    def get_subsampled_ego_trajectory(self, log_id: str, sample_rate_hz: float = 1.0) -> NDArrayFloat:
        """Get the trajectory of the AV (egovehicle) at an approximate sampling rate (Hz).

        Note: the trajectory is NOT interpolated to give an *exact* sampling rate.

        Args:
            log_id: unique ID of vehicle log.
            sample_rate_hz: provide sample_rate_hz pose measurements per second. We approximate this
                by providing a new pose after a required interval has elapsed. Since pose measurements
                are provided at a high frequency, this is adequate for the purposes of visualization.

        Returns:
            array of shape (N,2) representing autonomous vehicle's (AV) trajectory.

        Raises:
            ValueError: If pose timestamps aren't in chronological order.
        """
        if sample_rate_hz > MAX_MEASUREMENT_FREQUENCY_HZ:
            logger.warning(
                "You requested at sampling rate of %d Hz, but the measurements are only at nanosecond precision, "
                " so falling back to %d Hz.",
                sample_rate_hz,
                MAX_MEASUREMENT_FREQUENCY_HZ,
            )

        log_poses_df = io_utils.read_feather(self._data_dir / log_id / "city_SE3_egovehicle.feather")

        # timestamp of the pose measurement.
        timestamp_ns = list(log_poses_df.timestamp_ns)
        tx_m = list(log_poses_df.tx_m)
        ty_m = list(log_poses_df.ty_m)

        if not np.array_equal(np.argsort(timestamp_ns), np.arange(len(timestamp_ns))):
            raise ValueError("Pose timestamps are not sorted chronologically, invalid.")

        # e.g. for 2 Hz, get a sampling rate of 500 ms, then convert to nanoseconds.
        interval_threshold_s = 1 / sample_rate_hz
        # addition must happen in integer domain, not float domain, to prevent overflow
        interval_threshold_ns = int(interval_threshold_s * NANOSECONDS_PER_SECOND)

        next_timestamp = timestamp_ns[0] + interval_threshold_ns
        traj = [(tx_m[0], ty_m[0])]

        for timestamp_, tx_, ty_ in zip(timestamp_ns[1:], tx_m[1:], ty_m[1:]):
            if timestamp_ < next_timestamp:
                # still within last interval, need to exit this interval before sampling new measurement.
                continue
            traj += [(tx_, ty_)]
            next_timestamp = timestamp_ + interval_threshold_ns

        traj_npy: NDArrayFloat = np.array(traj, dtype=float)
        return traj_npy

    def get_log_map_dirpath(self, log_id: str) -> Path:
        """Fetch the path to the directory containing map files for a single vehicle log."""
        return self._data_dir / log_id / "map"

    def get_city_name(self, log_id: str) -> str:
        """Return the name of the city where the log of interest was captured.

        Vector map filenames contain the city name, and have a name in the following format:
            `log_map_archive_453e5558-6363-38e3-bf9b-42b5ba0a6f1d____PAO_city_71741.json`

        Args:
            log_id: unique ID of vehicle log.

        Returns:
            Name of the city where the log of interest was captured.

        Raises:
            RuntimeError: If no vector map file is found for the query log ID.
        """
        vector_map_fpaths = list(self._data_dir.glob(f"{log_id}/map/log_map_archive*"))
        if len(vector_map_fpaths) == 0:
            raise RuntimeError(f"Vector map file is missing for {log_id}.")
        vector_map_fpath = vector_map_fpaths[0]
        log_city_name = vector_map_fpath.name.split("____")[1].split("_")[0]
        return log_city_name

    def get_log_ids(self) -> List[str]:
        """Return a list of all vehicle log IDs available at the provided dataroot."""
        return sorted([d.name for d in self._data_dir.glob("*") if d.is_dir()])

    def get_closest_img_fpath(self, log_id: str, cam_name: str, lidar_timestamp_ns: int) -> Optional[Path]:
        """Return the filepath corresponding to the closest image from the lidar timestamp.

        Args:
            log_id: unique ID of vehicle log.
            cam_name: name of camera.
            lidar_timestamp_ns: timestamp of LiDAR sweep capture, in nanoseconds

        Returns:
            im_fpath: path to image if one is found within the expected time interval, or else None.
        """
        cam_timestamp_ns = self._sdb.get_closest_cam_channel_timestamp(lidar_timestamp_ns, cam_name, log_id)
        if cam_timestamp_ns is None:
            return None
        img_fpath = self._data_dir / log_id / "sensors" / "cameras" / cam_name / f"{cam_timestamp_ns}.jpg"
        return img_fpath

    def get_closest_lidar_fpath(self, log_id: str, cam_timestamp_ns: int) -> Optional[Path]:
        """Get file path for LiDAR sweep accumulated to a timestamp closest to a camera timestamp.

        Args:
            log_id: unique ID of vehicle log.
            cam_timestamp_ns: integer timestamp of image capture, in nanoseconds

        Returns:
            lidar_fpath: path to sweep .feather file if one is found within the expected time interval, or else None.
        """
        lidar_timestamp_ns = self._sdb.get_closest_lidar_timestamp(cam_timestamp_ns, log_id)
        if lidar_timestamp_ns is None:
            return None
        lidar_fname = f"{lidar_timestamp_ns}.feather"
        lidar_fpath = self._data_dir / log_id / "sensors" / "lidar" / lidar_fname
        return lidar_fpath

    def get_lidar_fpath_at_lidar_timestamp(self, log_id: str, lidar_timestamp_ns: int) -> Optional[Path]:
        """Return the file path for the LiDAR sweep accumulated to the query timestamp, if it exists.

        Args:
            log_id: unique ID of vehicle log.
            lidar_timestamp_ns: timestamp of LiDAR sweep capture, in nanoseconds

        Returns:
            Path to sweep .feather file if one exists at the requested timestamp, or else None.
        """
        lidar_fname = f"{lidar_timestamp_ns}.feather"
        lidar_fpath = self._data_dir / log_id / "sensors" / "lidar" / lidar_fname
        if not lidar_fpath.exists():
            return None

        return lidar_fpath

    def get_lidar_fpath(self, log_id: str, lidar_timestamp_ns: int) -> Path:
        """Get file path for LiDAR sweep accumulated to the query reference timestamp.

        Args:
            log_id: unique ID of vehicle log.
            lidar_timestamp_ns: query reference timestamp, in nanoseconds.

        Returns:
            Path to .feather file, containing sweep information.
        """
        lidar_fname = f"{lidar_timestamp_ns}.feather"
        lidar_fpath = Path(self._data_dir) / log_id / "sensors" / "lidar" / lidar_fname
        return lidar_fpath

    def get_ordered_log_lidar_timestamps(self, log_id: str) -> List[int]:
        """Return chronologically-ordered timestamps corresponding to each LiDAR sweep in a log.

        Args:
            log_id: unique ID of vehicle log.

        Returns:
            lidar_timestamps_ns: ordered timestamps, provided in nanoseconds.
        """
        ordered_lidar_fpaths: List[Path] = self.get_ordered_log_lidar_fpaths(log_id=log_id)
        lidar_timestamps_ns = [int(fp.stem) for fp in ordered_lidar_fpaths]
        return lidar_timestamps_ns

    def get_ordered_log_lidar_fpaths(self, log_id: str) -> List[Path]:
        """Get a list of all file paths for LiDAR sweeps in a single log (ordered chronologically).

        Args:
            log_id: unique ID of vehicle log.

        Returns:
            lidar_fpaths: List of paths to chronologically ordered LiDAR feather files in this log.
                File paths are strings are of the same length ending with a nanosecond timestamp, thus
                sorted() will place them in numerical order.
        """
        lidar_fpaths = sorted(self._data_dir.glob(f"{log_id}/sensors/lidar/*.feather"))
        return lidar_fpaths

    def get_ordered_log_cam_fpaths(self, log_id: str, cam_name: str) -> List[Path]:
        """Get a list of all file paths for one particular camera in a single log (ordered chronologically).

        Args:
            log_id: unique ID of vehicle log.
            cam_name: camera name.

        Returns:
            List of paths, representing paths to ordered JPEG files in this log, for a specific camera.
        """
        cam_img_fpaths = sorted(self._data_dir.glob(f"{log_id}/sensors/cameras/{cam_name}/*.jpg"))
        return cam_img_fpaths

    def get_labels_at_lidar_timestamp(self, log_id: str, lidar_timestamp_ns: int) -> CuboidList:
        """Load the sweep annotations at the provided timestamp.

        Args:
            log_id: Log unique id.
            lidar_timestamp_ns: Nanosecond timestamp.

        Returns:
            Cuboid list of annotations.
        """
        annotations_feather_path = self._data_dir / log_id / "annotations.feather"

        # Load annotations from disk.
        # NOTE: This file contains annotations for the ENTIRE sequence.
        # The sweep annotations are selected below.
        cuboid_list = CuboidList.from_feather(annotations_feather_path)
        cuboids = list(filter(lambda x: x.timestamp_ns == lidar_timestamp_ns, cuboid_list.cuboids))
        return CuboidList(cuboids=cuboids)

    def project_ego_to_img_motion_compensated(
        self,
        points_lidar_time: NDArrayFloat,
        cam_name: str,
        cam_timestamp_ns: int,
        lidar_timestamp_ns: int,
        log_id: str,
    ) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayBool]:
        """Project points in the ego frame to the image with motion compensation.

        Args:
            points_lidar_time: Numpy array of shape (N,3) representing points in the egovehicle frame.
            cam_name: name of camera.
            cam_timestamp_ns: timestamp (in nanoseconds) when camera image was recorded.
            lidar_timestamp_ns: timestamp (in nanoseconds) when LiDAR sweep was recorded.
            log_id: unique ID of vehicle log.

        Returns:
            uv: image plane coordinates, as Numpy array of shape (N,2).
            points_cam: Numpy array of shape (N,3) representing coordinates of points within the camera frame.
            is_valid_points: boolean indicator of valid cheirality and within image boundary, as
                boolean Numpy array of shape (N,).
        """
        pinhole_camera = self.get_log_pinhole_camera(log_id=log_id, cam_name=cam_name)

        # get transformation to bring point in egovehicle frame to city frame,
        # at the time when camera image was recorded.
        city_SE3_ego_cam_t = self.get_city_SE3_ego(log_id=log_id, timestamp_ns=cam_timestamp_ns)

        # get transformation to bring point in egovehicle frame to city frame,
        # at the time when the LiDAR sweep was recorded.
        city_SE3_ego_lidar_t = self.get_city_SE3_ego(log_id=log_id, timestamp_ns=lidar_timestamp_ns)

        return pinhole_camera.project_ego_to_img_motion_compensated(
            points_lidar_time=points_lidar_time,
            city_SE3_ego_cam_t=city_SE3_ego_cam_t,
            city_SE3_ego_lidar_t=city_SE3_ego_lidar_t,
        )

    def get_colored_sweep(self, log_id: str, lidar_timestamp_ns: int) -> NDArrayByte:
        """Given a LiDAR sweep, use its corresponding RGB imagery to color the sweep points.

        If a sweep points is not captured by any ring camera, it will default to black color.

        Args:
            log_id: unique ID of vehicle log.
            lidar_timestamp_ns: timestamp (in nanoseconds) when LiDAR sweep was recorded.

        Returns:
            Array of shape (N,3) representing RGB colors per sweep point.

        Raises:
            ValueError: If requested timestamp has no corresponding LiDAR sweep.
        """
        lidar_fpath = self.get_lidar_fpath_at_lidar_timestamp(log_id=log_id, lidar_timestamp_ns=lidar_timestamp_ns)
        if lidar_fpath is None:
            raise ValueError("Requested colored sweep at a timestamp that has no corresponding LiDAR sweep.")

        sweep = Sweep.from_feather(lidar_fpath)
        n_sweep_pts = len(sweep)
        # defaults to black RGB (0,0,0)
        sweep_rgb: NDArrayByte = np.zeros((n_sweep_pts, 3), dtype=np.uint8)

        # color as much of the sweep that we can, as we loop through each camera.
        for cam_enum in list(RingCameras):
            cam_name = cam_enum.value
            img_fpath = self.get_closest_img_fpath(
                log_id=log_id, cam_name=cam_name, lidar_timestamp_ns=lidar_timestamp_ns
            )
            if img_fpath is None:
                continue
            cam_timestamp_ns = int(img_fpath.stem)

            uv, points_cam, is_valid = self.project_ego_to_img_motion_compensated(
                points_lidar_time=sweep.xyz,
                cam_name=cam_name,
                cam_timestamp_ns=cam_timestamp_ns,
                lidar_timestamp_ns=lidar_timestamp_ns,
                log_id=log_id,
            )
            uv_valid = np.round(uv[is_valid]).astype(np.int64)
            u = uv_valid[:, 0]
            v = uv_valid[:, 1]
            img = io_utils.read_img(img_fpath, channel_order="RGB")
            sweep_rgb[is_valid] = img[v, u]

        return sweep_rgb

    def get_depth_map_from_lidar(
        self,
        lidar_points: NDArrayFloat,
        cam_name: str,
        log_id: str,
        cam_timestamp_ns: int,
        lidar_timestamp_ns: int,
        interp_depth_map: bool = True,
    ) -> Optional[NDArrayFloat]:
        """Create a sparse or dense depth map, with height & width equivalent to the corresponding camera image.

        Args:
            lidar_points: array of shape (K,3)
            cam_name: name of camera from which to simulate a per-pixel depth map.
            log_id: unique identifier of log/scenario.
            cam_timestamp_ns: timestamp when image was captured, measured in nanoseconds.
            lidar_timestamp_ns: timestamp when LiDAR was captured, measured in nanoseconds.
            interp_depth_map: whether to densely interpolate the depth map from sparse LiDAR returns.

        Returns:
            depth_map: array of shape (height_px, width_px) representing a depth map.

        Raises:
            RuntimeError: If `u` or `v` are outside of the camera dimensions.
        """
        pinhole_camera = self.get_log_pinhole_camera(log_id=log_id, cam_name=cam_name)
        height_px, width_px = pinhole_camera.height_px, pinhole_camera.width_px

        # motion compensate always
        uv, points_cam, is_valid_points = self.project_ego_to_img_motion_compensated(
            points_lidar_time=lidar_points,
            cam_name=cam_name,
            cam_timestamp_ns=cam_timestamp_ns,
            lidar_timestamp_ns=lidar_timestamp_ns,
            log_id=log_id,
        )
        if uv is None or points_cam is None:
            # poses were missing for either the camera or lidar timestamp
            return None
        if is_valid_points is None or is_valid_points.sum() == 0:
            return None

        u = np.round(uv[:, 0][is_valid_points]).astype(np.int32)
        v = np.round(uv[:, 1][is_valid_points]).astype(np.int32)
        z = points_cam[:, 2][is_valid_points]

        depth_map: NDArrayFloat = np.zeros((height_px, width_px), dtype=np.float32)

        # form depth map from LiDAR
        if interp_depth_map:
            if u.max() > pinhole_camera.width_px or v.max() > pinhole_camera.height_px:
                raise RuntimeError("Regular grid interpolation will fail due to out-of-bound inputs.")

            depth_map = dense_grid_interpolation.interp_dense_grid_from_sparse(
                grid_img=depth_map,
                points=np.hstack([u.reshape(-1, 1), v.reshape(-1, 1)]),
                values=z,
                grid_h=height_px,
                grid_w=width_px,
                interp_method="linear",
            ).astype(float)
        else:
            depth_map[v, u] = z

        return depth_map


def convert_pose_dataframe_to_SE3(pose_df: pd.DataFrame) -> SE3:
    """Convert a dataframe with parameterization of a single pose, to an SE(3) object.

    Args:
        pose_df: parameterization of a single pose.

    Returns:
        SE(3) object representing the egovehicle's pose in the city frame.
    """
    qw, qx, qy, qz = pose_df[["qw", "qx", "qy", "qz"]].to_numpy().squeeze()
    tx_m, ty_m, tz_m = pose_df[["tx_m", "ty_m", "tz_m"]].to_numpy().squeeze()
    city_q_ego: NDArrayFloat = np.array([qw, qx, qy, qz])
    city_t_ego: NDArrayFloat = np.array([tx_m, ty_m, tz_m])
    city_R_ego = geometry_utils.quat_to_mat(quat_wxyz=city_q_ego)
    city_SE3_ego = SE3(rotation=city_R_ego, translation=city_t_ego)
    return city_SE3_ego
