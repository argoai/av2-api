# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Example script for loading data from the AV2 sensor dataset."""

from pathlib import Path
from typing import Final, List, Tuple, Union

import click
import numpy as np
from rich.progress import track

from av2.datasets.sensor.constants import RingCameras, StereoCameras
from av2.datasets.sensor.sensor_dataloader import SensorDataloader
from av2.rendering.color import ColorFormats, create_range_map
from av2.rendering.rasterize import draw_points_xy_in_img
from av2.rendering.video import tile_cameras, write_video
from av2.structures.ndgrid import BEVGrid
from av2.utils.typing import NDArrayByte, NDArrayInt

# Bird's-eye view parameters.
MIN_RANGE_M: Tuple[float, float] = (-102.4, -77.5)
MAX_RANGE_M: Tuple[float, float] = (+102.4, +77.5)
RESOLUTION_M_PER_CELL: Tuple[float, float] = (+0.1, +0.1)

# Model an xy grid in the Bird's-eye view.
BEV_GRID: Final[BEVGrid] = BEVGrid(
    min_range_m=MIN_RANGE_M, max_range_m=MAX_RANGE_M, resolution_m_per_cell=RESOLUTION_M_PER_CELL
)


def generate_sensor_dataset_visualizations(
    dataset_dir: Path,
    with_annotations: bool,
    cam_names: Tuple[Union[RingCameras, StereoCameras], ...],
) -> None:
    """Create a video of a point cloud in the ego-view. Annotations may be overlaid.

    Args:
        dataset_dir: Path to the dataset directory.
        with_annotations: Boolean flag to enable loading of annotations.
        cam_names: Set of camera names to render.
    """
    dataset = SensorDataloader(
        dataset_dir,
        with_annotations=with_annotations,
        with_cache=True,
        cam_names=cam_names,
    )

    tiled_cams_list: List[NDArrayByte] = []
    for _, datum in enumerate(track(dataset, "Creating sensor tutorial videos ...")):
        sweep = datum.sweep
        annotations = datum.annotations

        timestamp_city_SE3_ego_dict = datum.timestamp_city_SE3_ego_dict
        synchronized_imagery = datum.synchronized_imagery
        if synchronized_imagery is not None:
            cam_name_to_img = {}
            for cam_name, cam in synchronized_imagery.items():
                if (
                    cam.timestamp_ns in timestamp_city_SE3_ego_dict
                    and sweep.timestamp_ns in timestamp_city_SE3_ego_dict
                ):
                    city_SE3_ego_cam_t = timestamp_city_SE3_ego_dict[cam.timestamp_ns]
                    city_SE3_ego_lidar_t = timestamp_city_SE3_ego_dict[sweep.timestamp_ns]

                    uv, points_cam, is_valid_points = cam.camera_model.project_ego_to_img_motion_compensated(
                        sweep.xyz,
                        city_SE3_ego_cam_t=city_SE3_ego_cam_t,
                        city_SE3_ego_lidar_t=city_SE3_ego_lidar_t,
                    )

                    uv_int: NDArrayInt = np.round(uv[is_valid_points]).astype(int)
                    colors = create_range_map(points_cam[is_valid_points, :3])
                    img = draw_points_xy_in_img(
                        cam.img, uv_int, colors=colors, alpha=0.85, diameter=5, sigma=1.0, with_anti_alias=True
                    )
                    if annotations is not None:
                        img = annotations.project_to_cam(
                            img, cam.camera_model, city_SE3_ego_cam_t, city_SE3_ego_lidar_t
                        )
                    cam_name_to_img[cam_name] = img
            if len(cam_name_to_img) < len(cam_names):
                continue
            tiled_img = tile_cameras(cam_name_to_img, bev_img=None)
            tiled_cams_list.append(tiled_img)

        if datum.sweep_number == datum.num_sweeps_in_log - 1:
            video: NDArrayByte = np.stack(tiled_cams_list)
            dst_path = Path("videos") / f"{datum.log_id}.mp4"
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            write_video(video, dst_path, crf=30, color_format=ColorFormats.BGR)
            tiled_cams_list = []


@click.command(help="Generate visualizations from the Argoverse 2 Sensor Dataset.")
@click.option(
    "-d",
    "--dataset-dir",
    required=True,
    help="Path to local directory where the Argoverse 2 Sensor Dataset is stored.",
    type=click.Path(exists=True),
)
@click.option(
    "-a",
    "--with-annotations",
    default=True,
    help="Boolean flag to return annotations from the dataloader.",
    type=bool,
)
@click.option(
    "-c",
    "--cam_names",
    default=tuple(x.value for x in RingCameras),
    help="List of cameras to load for each lidar sweep.",
    multiple=True,
    type=str,
)
def run_generate_sensor_dataset_visualizations(
    dataset_dir: str, with_annotations: bool, cam_names: Tuple[str, ...]
) -> None:
    """Click entry point for Argoverse Sensor Dataset visualization.

    Args:
        dataset_dir: Dataset directory.
        with_annotations: Boolean flag to return annotations.
        cam_names: Tuple of camera names to load.

    Raises:
        ValueError: If no valid camera names are provided.
    """
    valid_ring_cams = set([x.value for x in RingCameras])
    valid_stereo_cams = set([x.value for x in StereoCameras])

    cam_enums: List[Union[RingCameras, StereoCameras]] = []
    for cam_name in cam_names:
        if cam_name in valid_ring_cams:
            cam_enums.append(RingCameras(cam_name))
        elif cam_name in valid_stereo_cams:
            cam_enums.append(StereoCameras(cam_name))
        else:
            raise ValueError("Must provide _valid_ camera names!")

    generate_sensor_dataset_visualizations(
        Path(dataset_dir),
        with_annotations,
        tuple(cam_enums),
    )


if __name__ == "__main__":
    run_generate_sensor_dataset_visualizations()
