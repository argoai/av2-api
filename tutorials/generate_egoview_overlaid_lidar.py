# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Lidar egoview visualization."""

import logging
import os
import sys
from pathlib import Path
from typing import Final

import click
import cv2
import numpy as np

import av2.rendering.color as color_utils
import av2.rendering.rasterize as raster_rendering_utils
import av2.rendering.video as video_utils
import av2.utils.io as io_utils
import av2.utils.raster as raster_utils
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.datasets.sensor.constants import RingCameras
from av2.map.map_api import ArgoverseStaticMap
from av2.rendering.color import GREEN_HEX, RED_HEX
from av2.utils.typing import NDArrayByte, NDArrayFloat, NDArrayInt

logger = logging.getLogger(__name__)


NUM_RANGE_BINS: Final[int] = 50
RING_CAMERA_FPS: Final[int] = 20


def generate_egoview_overlaid_lidar(
    data_root: Path, output_dir: Path, log_id: str, render_ground_pts_only: bool, dump_single_frames: bool
) -> None:
    """Render LiDAR points from a particular camera's viewpoint (color by ground surface, and apply ROI filtering).

    Args:
        data_root: path to directory where the logs live on disk.
        output_dir: path to directory where renderings will be saved.
        log_id: unique ID for AV2 scenario/log.
        render_ground_pts_only: whether to only render LiDAR points located close to the ground surface.
        dump_single_frames: Whether to save to disk individual RGB frames of the rendering, in addition to generating
            the mp4 file.

    Raises:
        RuntimeError: If vehicle log data is not present at `data_root` for `log_id`.
    """
    loader = AV2SensorDataLoader(data_dir=data_root, labels_dir=data_root)

    log_map_dirpath = data_root / log_id / "map"
    avm = ArgoverseStaticMap.from_map_dir(log_map_dirpath, build_raster=True)

    # repeat red to green colormap every 50 m.
    colors_arr_rgb = color_utils.create_colormap(color_list=[RED_HEX, GREEN_HEX], n_colors=NUM_RANGE_BINS)
    colors_arr_rgb = (colors_arr_rgb * 255).astype(np.uint8)
    colors_arr_bgr: NDArrayByte = np.fliplr(colors_arr_rgb)

    for _, cam_name in enumerate(list(RingCameras)):
        cam_im_fpaths = loader.get_ordered_log_cam_fpaths(log_id, cam_name)
        num_cam_imgs = len(cam_im_fpaths)

        video_list = []
        for i, im_fpath in enumerate(cam_im_fpaths):
            if i % 50 == 0:
                logging.info(f"\tOn file {i}/{num_cam_imgs} of camera {cam_name} of {log_id}")

            cam_timestamp_ns = int(im_fpath.stem)
            city_SE3_ego = loader.get_city_SE3_ego(log_id, cam_timestamp_ns)
            if city_SE3_ego is None:
                logger.exception("missing LiDAR pose")
                continue

            # load feather file path, e.g. '315978406032859416.feather"
            lidar_fpath = loader.get_closest_lidar_fpath(log_id, cam_timestamp_ns)
            if lidar_fpath is None:
                logger.info("No LiDAR sweep found within the synchronization interval for %s, so skipping...", cam_name)
                continue

            img_bgr = io_utils.read_img(im_fpath, channel_order="BGR")

            lidar_points_ego = io_utils.read_lidar_sweep(lidar_fpath, attrib_spec="xyz")
            lidar_timestamp_ns = int(lidar_fpath.stem)

            # put into city coords, then prune away ground and non-RoI points
            lidar_points_city = city_SE3_ego.transform_point_cloud(lidar_points_ego)
            lidar_points_city = avm.remove_non_drivable_area_points(lidar_points_city)
            is_ground_logicals = avm.get_ground_points_boolean(lidar_points_city)
            lidar_points_city = lidar_points_city[is_ground_logicals if render_ground_pts_only else ~is_ground_logicals]
            lidar_points_ego = city_SE3_ego.inverse().transform_point_cloud(lidar_points_city)

            # motion compensate always
            uv, points_cam, is_valid_points = loader.project_ego_to_img_motion_compensated(
                points_lidar_time=lidar_points_ego,
                cam_name=cam_name,
                cam_timestamp_ns=cam_timestamp_ns,
                lidar_timestamp_ns=lidar_timestamp_ns,
                log_id=log_id,
            )

            if is_valid_points is None or uv is None or points_cam is None:
                continue

            if is_valid_points.sum() == 0:
                continue

            uv_int: NDArrayInt = np.round(uv[is_valid_points]).astype(np.int32)
            points_cam = points_cam[is_valid_points]
            pt_ranges: NDArrayFloat = np.linalg.norm(points_cam[:, :3], axis=1)
            color_bins: NDArrayInt = np.round(pt_ranges).astype(np.int32)
            # account for moving past 100 meters, loop around again
            color_bins = color_bins % NUM_RANGE_BINS
            uv_colors_bgr = colors_arr_bgr[color_bins]

            img_empty = np.full_like(img_bgr, fill_value=255)
            img_empty = raster_rendering_utils.draw_points_xy_in_img(img_empty, uv_int, uv_colors_bgr, diameter=10)
            blended_bgr = raster_utils.blend_images(img_bgr, img_empty)
            frame_rgb = blended_bgr[:, :, ::-1]

            if dump_single_frames:
                save_dir = output_dir / log_id / cam_name
                os.makedirs(save_dir, exist_ok=True)
                cv2.imwrite(str(save_dir / f"{cam_name}_{lidar_timestamp_ns}.jpg"), blended_bgr)

            video_list.append(frame_rgb)

        if len(video_list) == 0:
            raise RuntimeError("No video frames were found; log data was not found on disk.")

        video: NDArrayByte = np.stack(video_list).astype(np.uint8)
        video_output_dir = output_dir / "videos"
        video_utils.write_video(video=video, dst=video_output_dir / f"{log_id}_{cam_name}.mp4", fps=RING_CAMERA_FPS)


@click.command(help="Generate LiDAR + map visualizations from the Argoverse 2 Sensor Dataset.")
@click.option(
    "-d",
    "--data-root",
    required=True,
    help="Path to local directory where the Argoverse 2 Sensor Dataset logs are stored.",
    type=click.Path(exists=True),
)
@click.option(
    "-o",
    "--output-dir",
    required=True,
    help="Path to local directory where renderings will be saved.",
    type=str,
)
@click.option(
    "-l",
    "--log-id",
    default="00a6ffc1-6ce9-3bc3-a060-6006e9893a1a",
    help="unique log identifier.",
    type=str,
)
@click.option(
    "-g",
    "--render-ground-pts-only",
    default=True,
    help="Boolean argument whether to only render LiDAR points located close to the ground surface.",
    type=bool,
)
@click.option(
    "-s",
    "--dump-single-frames",
    default=False,
    help="Whether to save to disk individual RGB frames of the rendering, in addition to generating the mp4 file"
    "(defaults to False). Note: can quickly generate 100s of MBs, for 200 KB frames.",
    type=bool,
)
def run_generate_egoview_overlaid_lidar(
    data_root: str, output_dir: str, log_id: str, render_ground_pts_only: bool, dump_single_frames: bool
) -> None:
    """Click entry point for visualizing LiDAR returns rendered on top of sensor imagery."""
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    data_root_path = Path(data_root)
    output_dir_path = Path(output_dir)

    logger.info(
        "data_root: %s, output_dir: %s, log_id: %s, render_ground_pts_only: %s, dump_single_frames: %s",
        data_root_path,
        output_dir_path,
        log_id,
        render_ground_pts_only,
        dump_single_frames,
    )
    generate_egoview_overlaid_lidar(
        data_root=data_root_path,
        output_dir=output_dir_path,
        log_id=log_id,
        render_ground_pts_only=render_ground_pts_only,
        dump_single_frames=dump_single_frames,
    )


if __name__ == "__main__":
    run_generate_egoview_overlaid_lidar()
