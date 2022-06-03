# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Argoverse 2 Lidar Dataset tutorial.

Generates videos visualizing the lidar dataset.
"""

from __future__ import annotations

import copy
import logging
import sys
from pathlib import Path
from typing import Final, List

import click
import cv2
import numpy as np
from rich.progress import track

import av2.rendering.video as video_utils
import av2.utils.io as io_utils
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.geometry.se3 import SE3
from av2.map.map_api import ArgoverseStaticMap
from av2.rendering.color import GREEN_BGR, PURPLE_BGR, RED_BGR
from av2.rendering.map_bev import BirdsEyeViewMapRenderer
from av2.structures.ndgrid import BEVGrid
from av2.structures.sweep import Sweep
from av2.utils.typing import NDArrayByte, NDArrayFloat

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)


LIDAR_HZ: Final[int] = 10
CROSSWALK_EPS: Final[float] = 1e-5


def accumulate_all_frames(loader: AV2SensorDataLoader, log_id: str) -> Sweep:
    """Aggregate sweeps representing (x,y,z) in city coordinate system and intensity.

    Args:
        loader: The AV2 Lidar Dataset dataloader.
        log_id: Unique ID for AV2 scenario/log.

    Returns:
        Sweep representing aggregated lidar returns, placed in the city frame.

    Raises:
        RuntimeError: If no lidar sweeps are found.
    """
    lidar_timestamps_ns = loader.get_ordered_log_lidar_timestamps(log_id=log_id)

    if len(lidar_timestamps_ns) == 0:
        raise RuntimeError("No lidar sweeps were found.")

    city_SE3_ego = loader.get_city_SE3_ego(log_id=log_id, timestamp_ns=lidar_timestamps_ns[0])
    feather_fpath = loader.get_lidar_fpath(log_id, lidar_timestamps_ns[0])
    sweep_ego_base = Sweep.from_feather(feather_fpath)
    agg_sweep_city = city_SE3_ego.transform_sweep_from(sweep_ego_base)

    for lidar_timestamp_ns in track(lidar_timestamps_ns[1:], description="Aggregating sweeps ..."):
        city_SE3_ego = loader.get_city_SE3_ego(log_id=log_id, timestamp_ns=lidar_timestamp_ns)
        feather_fpath = loader.get_lidar_fpath(log_id, lidar_timestamp_ns)
        sweep_ego = Sweep.from_feather(feather_fpath)
        sweep_city = city_SE3_ego.transform_sweep_from(sweep_ego)
        agg_sweep_city.stack(sweep_city)

    logger.info("Aggregated %d points.", len(agg_sweep_city))
    return agg_sweep_city


def render_map_bev(
    img_bgr: NDArrayByte, bev_map_renderer: BirdsEyeViewMapRenderer, line_width_px: int = 2
) -> NDArrayByte:
    """Render pedestrian crossings and lane segments in the ego-view.

    Pedestrian crossings (crosswalks) will be rendered in purple, lane markings will be colored according to their
    marking color, or otherwise red, if markings are implicit, and drivable area in green.

    Args:
        img_bgr: Array of shape (H,W,3) representing BGR canvas to rasterize map elements onto.
        bev_map_renderer: Rendering engine for map elements in the bird's-eye-view (BEV).
        line_width_px: Thickness (in pixels) to use for rendering each polyline.

    Returns:
        Array of shape (H,W,3) and type uint8 representing a BGR image.
    """
    # Overlay drivable area.
    for da_polygon_city in bev_map_renderer.avm.vector_drivable_areas.values():
        da_polygon_city_xyz: NDArrayFloat = da_polygon_city.xyz

        bev_map_renderer.render_polyline_bev(
            da_polygon_city_xyz,
            img_bgr,
            bound_color=GREEN_BGR,
            thickness_px=line_width_px,
        )

        img_bgr = bev_map_renderer.render_filled_polyline_bev(
            da_polygon_city_xyz,
            img_bgr,
            bound_color=GREEN_BGR,
            alpha=0.7,
        )

    # Overlay lane segments.
    for ls in bev_map_renderer.avm.get_scenario_lane_segments():
        img_bgr = bev_map_renderer.render_lane_boundary_bev(img_bgr, ls, "right", line_width_px)
        img_bgr = bev_map_renderer.render_lane_boundary_bev(img_bgr, ls, "left", line_width_px)

    # Overlay pedestrian crossings.
    for pc in bev_map_renderer.avm.get_scenario_ped_crossings():
        # Render ped crossings (pc's).
        crosswalk_polygon = pc.polygon
        # Prevent duplicate first and last coords.
        crosswalk_polygon[:-1] += CROSSWALK_EPS

        bev_map_renderer.render_polyline_bev(
            crosswalk_polygon,
            img_bgr,
            bound_color=PURPLE_BGR,
            thickness_px=line_width_px,
        )

        img_bgr = bev_map_renderer.render_filled_polyline_bev(
            crosswalk_polygon,
            img_bgr,
            bound_color=PURPLE_BGR,
            alpha=0.4,
        )

    return img_bgr


def render_log_lidar_dataset_video(
    data_root: Path,
    output_dir: Path,
    log_id: str,
    render_north_as_up: bool,
    use_single_sweep: bool,
    dump_single_frames: bool,
    bev_grid: BEVGrid,
    overlay_map: bool,
) -> None:
    """Render lidar intensity and map in the BEV for each frame (sweep) of an AV2 log.

    Args:
        data_root: Path to local directory where the Argoverse 2 Lidar Dataset logs are stored.
        output_dir: Path to local directory where renderings will be saved.
        log_id: Unique ID for AV2 scenario/log.
        render_north_as_up: Whether to render north as up in the rendering.
        use_single_sweep: Whether to use a single sweep vs. aggregating points from all sweeps.
        dump_single_frames: Whether to save to disk individual RGB frames of the rendering,
            in addition to generating the mp4 file.
        bev_grid: Bird's-eye-view (BEV) rendering parameters.
        overlay_map: Whether to overlay the local vector map on the lidar sweep's BEV.
    """
    save_dir = output_dir / log_id
    save_dir.mkdir(parents=True, exist_ok=True)

    loader = AV2SensorDataLoader(data_dir=data_root, labels_dir=data_root)

    if overlay_map:
        log_map_dirpath = data_root / log_id / "map"
        avm = ArgoverseStaticMap.from_map_dir(log_map_dirpath, build_raster=False)

    if not use_single_sweep:
        agg_sweep_city = accumulate_all_frames(loader=loader, log_id=log_id)
        agg_sweep_city.equalize_intensity_distribution()

    video_list: List[NDArrayByte] = []

    lidar_timestamps_ns = loader.get_ordered_log_lidar_timestamps(log_id=log_id)
    for lidar_timestamp_ns in track(lidar_timestamps_ns, description="Rendering frames..."):
        city_SE3_ego = loader.get_city_SE3_ego(log_id=log_id, timestamp_ns=lidar_timestamp_ns)
        ego_SE3_city = city_SE3_ego.inverse()

        # Whether or not to accumulate multiple sweeps.
        if use_single_sweep:
            feather_fpath = loader.get_lidar_fpath(log_id, lidar_timestamp_ns)
            sweep_ego = Sweep.from_feather(feather_fpath)
            sweep_ego.equalize_intensity_distribution()
        else:
            # Accumulate sweeps.
            sweep_ego = ego_SE3_city.transform_sweep_from(copy.deepcopy(agg_sweep_city))

        # Allow option to keep North as +y in image frame.
        if render_north_as_up:
            city_SO3_ego = SE3(rotation=city_SE3_ego.rotation, translation=np.zeros(3))
            # Centered at egovehicle, but now with the city's orientation.
            sweep_ego = city_SO3_ego.transform_sweep_from(sweep_ego)

        frame_rgb = bev_grid.points_to_bev_img(sweep_ego.xyz, color=sweep_ego.intensity, diameter=1)

        # Convert RGB to BGR color for OpenCV processing.
        frame_bgr: NDArrayByte = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Draw filled circle to represent the ego vehicle.
        h, w, _ = frame_bgr.shape
        cv2.circle(frame_bgr, (h // 2, w // 2), radius=10, color=RED_BGR, thickness=-1)

        # Whether to overlay the local vector map on the lidar sweep's BEV.
        if overlay_map:
            bev_map_renderer = BirdsEyeViewMapRenderer(
                avm=avm,
                city_SE3_ego=city_SE3_ego,
                bev_grid=bev_grid,
                render_north_as_up=render_north_as_up,
            )
            frame_bgr = render_map_bev(img_bgr=frame_bgr, bev_map_renderer=bev_map_renderer)

        # Flip image along y axis, for positive y-axis mirroring.
        frame_bgr_flipup: NDArrayByte = np.flipud(frame_bgr)  # type: ignore

        # Convert BGR to RGB color.
        frame_rgb_final: NDArrayByte = cv2.cvtColor(frame_bgr_flipup, cv2.COLOR_BGR2RGB)

        if dump_single_frames:
            img_fpath = save_dir / f"lidar_bev_{lidar_timestamp_ns}.png"
            io_utils.write_img(img_fpath, frame_rgb_final)

        video_list.append(frame_rgb_final)

    video: NDArrayByte = np.stack(video_list).astype(np.uint8)
    video_output_dir = output_dir / "videos"
    video_utils.write_video(
        video=video,
        dst=Path(video_output_dir) / f"{log_id}_lidar_dataset.mp4",
        fps=LIDAR_HZ,
        preset="medium",
    )


@click.command(help="Generate videos of logs from the Argoverse 2 Lidar Dataset.")
@click.option(
    "-d",
    "--data_root",
    required=True,
    help="Path to local directory where the Argoverse 2 Lidar Dataset logs are stored.",
    type=click.Path(exists=True),
)
@click.option(
    "-l",
    "--log_id",
    default="NdfecXiYwbBSGfmUiPE9s1T0KWc7odmO",
    help="Unique log identifier.",
    type=str,
)
@click.option(
    "-o",
    "--output_dir",
    required=True,
    help="Path to local directory where renderings will be saved.",
    type=str,
)
@click.option(
    "--render_north_as_up",
    default=True,
    help="Boolean flag whether to render north as up in the rendering"
    " (as opposed to using the +y axis of the egovehicle frame -- left of the AV -- as up).",
    type=bool,
)
@click.option(
    "--use_single_sweep",
    default=False,
    help="Whether to use a single sweep vs. aggregating points from all sweeps using ego-motion compensation.",
    type=bool,
)
@click.option(
    "--resolution_m_per_px",
    default=0.1,
    help="Resolution of rendering, expressed in meters/pixel.",
    type=float,
)
@click.option(
    "--range_m",
    default=50,
    help="Maximum spatial range to include in rendering (distance to 3D points by infinity norm).",
    type=float,
)
@click.option(
    "--dump_single_frames",
    default=False,
    help="Whether to save to disk individual RGB frames of the rendering, in addition to generating the mp4 file."
    " Note: it can quickly generate 100s of MBs, for 200 KB frames.",
    type=bool,
)
@click.option(
    "--overlay_map",
    default=True,
    help="Whether to overlay the local vector map on the lidar sweep's bird's eye view.",
    type=bool,
)
def run_render_log_lidar_dataset_video(
    data_root: str,
    output_dir: str,
    log_id: str,
    render_north_as_up: bool,
    use_single_sweep: bool,
    resolution_m_per_px: float,
    range_m: float,
    dump_single_frames: bool,
    overlay_map: bool,
) -> None:
    """Click entry point for lidar dataset log video generation."""
    logging.info(
        "data_root: %s, output_dir: %s, log_id: %s, render_north_as_up: %s, use_single_sweep: %s"
        "resolution_m_per_px: %f, range_m: %f, dump_single_frames: %s",
        data_root,
        output_dir,
        log_id,
        render_north_as_up,
        use_single_sweep,
        resolution_m_per_px,
        range_m,
        dump_single_frames,
        overlay_map,
    )

    bev_grid = BEVGrid(
        min_range_m=(-range_m, -range_m),
        max_range_m=(+range_m, +range_m),
        resolution_m_per_cell=(resolution_m_per_px, resolution_m_per_px),
    )

    render_log_lidar_dataset_video(
        data_root=Path(data_root),
        output_dir=Path(output_dir),
        log_id=log_id,
        render_north_as_up=render_north_as_up,
        use_single_sweep=use_single_sweep,
        dump_single_frames=dump_single_frames,
        bev_grid=bev_grid,
        overlay_map=overlay_map,
    )


if __name__ == "__main__":
    run_render_log_lidar_dataset_video()
