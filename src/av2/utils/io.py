# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Helper functions for deserializing AV2 data."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from pyarrow import feather

import av2.geometry.geometry as geometry_utils
from av2.geometry.se3 import SE3
from av2.utils.typing import NDArrayByte, NDArrayFloat

# Mapping from egovehicle time in nanoseconds to egovehicle pose.
TimestampedCitySE3EgoPoses = Dict[int, SE3]

# Mapping from sensor name to sensor pose.
SensorPosesMapping = Dict[str, SE3]


def read_feather(path: Path, columns: Optional[Tuple[str, ...]] = None) -> pd.DataFrame:
    """Read Apache Feather data from a .feather file.

    AV2 uses .feather to serialize much of its data. This function handles the deserialization
    process and returns a `pandas` DataFrame with rows corresponding to the records and the
    columns corresponding to the record attributes.

    Args:
        path: Source data file (e.g., 'lidar.feather', 'calibration.feather', etc.)
        columns: Tuple of columns to load for the given record. Defaults to None.

    Returns:
        (N,len(columns)) Apache Feather data represented as a `pandas` DataFrame.
    """
    data: pd.DataFrame = feather.read_feather(path, columns=columns)
    return data


def read_lidar_sweep(fpath: Path, attrib_spec: str = "xyz") -> NDArrayFloat:
    """Load a point cloud file from a filepath.

    Args:
        fpath: path to a .feather file
        attrib_spec: string of C characters, each char representing a desired point attribute
            x -> point x-coord
            y -> point y-coord
            z -> point z-coord

        The following attributes are not loaded:
            intensity -> point intensity
            laser_number -> laser number of laser from which point was returned
            offset_ns -> nanosecond timestamp offset per point, from sweep timestamp.

    Returns:
        Array of shape (N, C). If attrib_str is invalid, `None` will be returned

    Raises:
        ValueError: If any element of `attrib_spec` is not in (x, y, z, intensity, laser_number, offset_ns).
    """
    possible_attributes = ["x", "y", "z"]
    if not all([a in possible_attributes for a in attrib_spec]):
        raise ValueError("Attributes must be in (x, y, z, intensity, laser_number, offset_ns).")

    sweep_df = read_feather(fpath)

    # return only the requested point attributes
    sweep: NDArrayFloat = sweep_df[list(attrib_spec)].to_numpy().astype(np.float64)
    return sweep


def read_ego_SE3_sensor(log_dir: Path) -> SensorPosesMapping:
    """Read the sensor poses for the given log.

    The sensor pose defines an SE3 transformation from the sensor reference frame to the egovehicle reference frame.
    Mathematically we define this transformation as: $$ego_SE3_sensor$$.

    In other words, when this transformation is applied to a set of points in the sensor reference frame, they
    will be transformed to the egovehicle reference frame.

    Example (1).
        points_ego = ego_SE3_sensor(points_sensor) apply the SE3 transformation to points in the sensor reference frame.

    Example (2).
        sensor_SE3_ego = ego_SE3_sensor^{-1} take the inverse of the SE3 transformation.
        points_sensor = sensor_SE3_ego(points_ego) apply the SE3 transformation to points in the ego reference frame.

    Extrinsics:
        sensor_name: Name of the sensor.
        qw: scalar component of a quaternion.
        qx: X-axis coefficient of a quaternion.
        qy: Y-axis coefficient of a quaternion.
        qz: Z-axis coefficient of a quaternion.
        tx_m: X-axis translation component.
        ty_m: Y-axis translation component.
        tz_m: Z-axis translation component.

    Args:
        log_dir: Path to the log directory.

    Returns:
        Mapping from sensor name to sensor pose.
    """
    ego_SE3_sensor_path = Path(log_dir, "calibration", "egovehicle_SE3_sensor.feather")
    ego_SE3_sensor = read_feather(ego_SE3_sensor_path)
    rotations = geometry_utils.quat_to_mat(ego_SE3_sensor.loc[:, ["qw", "qx", "qy", "qz"]].to_numpy())
    translations = ego_SE3_sensor.loc[:, ["tx_m", "ty_m", "tz_m"]].to_numpy()
    sensor_names = ego_SE3_sensor.loc[:, "sensor_name"].to_numpy()

    sensor_name_to_pose: SensorPosesMapping = {
        name: SE3(rotation=rotations[i], translation=translations[i]) for i, name in enumerate(sensor_names)
    }
    return sensor_name_to_pose


def read_city_SE3_ego(log_dir: Path) -> TimestampedCitySE3EgoPoses:
    """Read the egovehicle poses in the city reference frame.

    The egovehicle city pose defines an SE3 transformation from the egovehicle reference frame to the city ref. frame.
    Mathematically we define this transformation as: $$city_SE3_ego$$.

    In other words, when this transformation is applied to a set of points in the egovehicle reference frame, they
    will be transformed to the city reference frame.

    Example (1).
        points_city = city_SE3_ego(points_ego) applying the SE3 transformation to points in the egovehicle ref. frame.

    Example (2).
        ego_SE3_city = city_SE3_ego^{-1} take the inverse of the SE3 transformation.
        points_ego = ego_SE3_city(points_city) applying the SE3 transformation to points in the city ref. frame.

    Extrinsics:
        timestamp_ns: Egovehicle nanosecond timestamp.
        qw: scalar component of a quaternion.
        qx: X-axis coefficient of a quaternion.
        qy: Y-axis coefficient of a quaternion.
        qz: Z-axis coefficient of a quaternion.
        tx_m: X-axis translation component.
        ty_m: Y-axis translation component.
        tz_m: Z-axis translation component.

    Args:
        log_dir: Path to the log directory.

    Returns:
        Mapping from egovehicle time (in nanoseconds) to egovehicle pose in the city reference frame.
    """
    city_SE3_ego_path = Path(log_dir, "city_SE3_egovehicle.feather")
    city_SE3_ego = read_feather(city_SE3_ego_path)

    quat_wxyz = city_SE3_ego.loc[:, ["qw", "qx", "qy", "qz"]].to_numpy()
    translation_xyz_m = city_SE3_ego.loc[:, ["tx_m", "ty_m", "tz_m"]].to_numpy()
    timestamps_ns = city_SE3_ego["timestamp_ns"].to_numpy()

    rotation = geometry_utils.quat_to_mat(quat_wxyz)
    timestamp_city_SE3_ego_dict: TimestampedCitySE3EgoPoses = {
        ts: SE3(rotation=rotation[i], translation=translation_xyz_m[i]) for i, ts in enumerate(timestamps_ns)
    }
    return timestamp_city_SE3_ego_dict


def read_img(img_path: Path, channel_order: str = "RGB") -> NDArrayByte:
    """Return a RGB or BGR image array, given an image file path.

    Args:
        img_path: Source path to load the image.
        channel_order: color channel ordering for 3-channel images.

    Returns:
        (H,W,3) RGB or BGR image.

    Raises:
        ValueError: If `channel_order` isn't 'RGB' or 'BGR'.
    """
    if channel_order not in ["RGB", "BGR"]:
        raise ValueError("Unsupported channel order (must be BGR or RGB).")

    img_bgr: NDArrayByte = cv2.imread(str(img_path))
    if channel_order == "BGR":
        return img_bgr

    img_rgb: NDArrayByte = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def write_img(img_path: Path, img: NDArrayByte, channel_order: str = "RGB") -> None:
    """Save image to disk.

    Args:
        img_path: Destination path to write the image.
        img: (H,W,3) image.
        channel_order: color channel ordering for 3-channel images.

    Raises:
        ValueError: If `channel_order` isn't 'RGB' or 'BGR'.
    """
    if channel_order not in ["RGB", "BGR"]:
        raise ValueError("Unsupported channel order (must be BGR or RGB).")

    if channel_order == "RGB":
        img: NDArrayByte = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # type: ignore

    cv2.imwrite(str(img_path), img)


def read_json_file(fpath: Path) -> Dict[str, Any]:
    """Load dictionary from JSON file.

    Args:
        fpath: Path to JSON file.

    Returns:
        Deserialized Python dictionary.
    """
    with open(fpath, "rb") as f:
        data: Dict[str, Any] = json.load(f)
        return data


def save_json_dict(
    json_fpath: Path,
    dictionary: Union[Dict[Any, Any], List[Any]],
) -> None:
    """Save a Python dictionary to a JSON file.

    Args:
        json_fpath: Path to file to create.
        dictionary: Python dictionary to be serialized.
    """
    with open(json_fpath, "w") as f:
        json.dump(dictionary, f)


def read_all_annotations(dataset_dir: Path, split: str) -> pd.DataFrame:
    """Read all annotations for a particular split.

    Args:
        dataset_dir: Root directory to the dataset.
        split: Name of the dataset split.

    Returns:
        Dataframe which contains all of the ground truth annotations from the split.
    """
    split_dir = dataset_dir / split
    annotations_path_list = split_dir.glob("*/annotations.feather")

    annotations_list: List[pd.DataFrame] = []
    for annotations_path in annotations_path_list:
        annotations = read_feather(annotations_path)
        annotations_list.append(annotations)
    return pd.concat(annotations_list).reset_index(drop=True)
