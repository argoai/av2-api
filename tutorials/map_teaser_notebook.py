# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Quick tutorial on AV2 map functionality, with visualizations of various map elements."""

import logging
import sys
from pathlib import Path
from typing import Final, Sequence, Tuple, Union

import click
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable

import av2.geometry.polyline_utils as polyline_utils
import av2.rendering.vector as vector_plotting_utils
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.map.lane_segment import LaneSegment
from av2.map.map_api import ArgoverseStaticMap
from av2.utils.typing import NDArrayFloat

logger = logging.getLogger(__name__)


SAVE_DIR = Path(__file__).resolve().parent.parent / "argoverse2_map_figures"

# scaled to [0,1] for matplotlib.
PURPLE_RGB: Final[Tuple[int, int, int]] = (201, 71, 245)
PURPLE_RGB_MPL: Final[Tuple[float, float, float]] = (PURPLE_RGB[0] / 255, PURPLE_RGB[1] / 255, PURPLE_RGB[2] / 255)

DARK_GRAY_RGB: Final[Tuple[int, int, int]] = (40, 39, 38)
DARK_GRAY_RGB_MPL: Final[Tuple[float, float, float]] = (
    DARK_GRAY_RGB[0] / 255,
    DARK_GRAY_RGB[1] / 255,
    DARK_GRAY_RGB[2] / 255,
)

OVERLAID_MAPS_ALPHA: Final[float] = 0.1


def single_log_teaser(data_root: Path, log_id: str, save_figures: bool) -> None:
    """For a single log, render all local lane segments in green, and pedestrian crossings in purple, in a bird's eye view.

    Args:
        data_root: path to where the AV2 logs live.
        log_id: unique ID for the AV2 vehicle log.
        save_figures: whether to save each generated figure to disk.
    """
    log_map_dirpath = data_root / log_id / "map"
    avm = ArgoverseStaticMap.from_map_dir(log_map_dirpath, build_raster=False)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()

    for _, ls in avm.vector_lane_segments.items():
        vector_plotting_utils.draw_polygon_mpl(ax, ls.polygon_boundary, color="g", linewidth=0.5)
        vector_plotting_utils.plot_polygon_patch_mpl(ls.polygon_boundary, ax, color="g", alpha=0.2)

    # plot all pedestrian crossings
    for _, pc in avm.vector_pedestrian_crossings.items():
        vector_plotting_utils.draw_polygon_mpl(ax, pc.polygon, color="m", linewidth=0.5)
        vector_plotting_utils.plot_polygon_patch_mpl(pc.polygon, ax, color="m", alpha=0.2)

    plt.axis("equal")
    plt.tight_layout()
    if save_figures:
        plt.savefig(SAVE_DIR / f"lane_graph_ped_crossings_{log_id}_bev.jpg", dpi=500)
    plt.show()

    # Plot drivable areas in light gray on a new plot.
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    for da in list(avm.vector_drivable_areas.values()):
        vector_plotting_utils.draw_polygon_mpl(ax, da.xyz, color="gray", linewidth=0.5)
        vector_plotting_utils.plot_polygon_patch_mpl(da.xyz, ax, color="gray", alpha=0.2)

    plt.axis("equal")
    plt.tight_layout()
    if save_figures:
        plt.savefig(SAVE_DIR / f"drivable_areas_{log_id}_bev.jpg", dpi=500)
    plt.show()
    plt.close("all")


def visualize_raster_layers(data_root: Path, log_id: str, save_figures: bool) -> None:
    """Visualize the ground surface height/elevation map, w/ a colorbar indicating the value range.

    Also, visualize side-by-side plots of the 3 raster arrays -- ground height, drivable area, ROI.

    Args:
        data_root: path to where the AV2 logs live.
        log_id: unique ID for the AV2 vehicle log.
        save_figures: whether to save each generated figure to disk.

    Raises:
        ValueError: If `self.raster_ground_height_layer`,`self.raster_drivable_area_layer`,
            or `self.raster_roi_layer` is `None`.
    """
    log_map_dirpath = data_root / log_id / "map"
    avm = ArgoverseStaticMap.from_map_dir(log_map_dirpath, build_raster=True)

    if avm.raster_ground_height_layer is None:
        raise ValueError("Raster ground height is not loaded!")
    if avm.raster_drivable_area_layer is None:
        raise ValueError("Raster drivable area is not loaded!")
    if avm.raster_roi_layer is None:
        raise ValueError("Raster ROI is not loaded!")

    height_array = avm.raster_ground_height_layer.array
    ax = plt.subplot()
    plt.title("Ground surface height (@ 30 centimeter resolution).")
    img = plt.imshow(np.flipud(height_array))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img, cax=cax)
    if save_figures:
        plt.savefig(SAVE_DIR / f"ground_surface_height_{log_id}_bev.jpg", dpi=500)
    plt.show()
    plt.close("all")

    fig = plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(np.flipud(height_array))
    plt.title("Ground Surface Height")

    plt.subplot(1, 3, 2)
    da_array = avm.raster_drivable_area_layer.array
    plt.imshow(np.flipud(da_array))
    plt.title("Drivable Area (rasterized \nfrom vector polygons)")

    plt.subplot(1, 3, 3)
    roi_array = avm.raster_roi_layer.array
    plt.imshow(np.flipud(roi_array))
    plt.title("Region of Interest (ROI)")

    fig.tight_layout()
    if save_figures:
        plt.savefig(SAVE_DIR / f"all_raster_layers_{log_id}_bev.jpg", dpi=500)
    plt.show()
    plt.close("all")


def overlaid_maps_all_logs_teaser(data_root: Path) -> None:
    """Render local maps rendered in the egovehicle frame from all logs on top of one another.

    The egovehicle points towards the +x axis in each.

    Args:
        data_root: path to where the AV2 logs live.
    """
    loader = AV2SensorDataLoader(data_dir=data_root, labels_dir=data_root)
    log_ids = loader.get_log_ids()

    fig = plt.figure(1, figsize=(10, 10), dpi=90)
    ax = fig.add_subplot(111)

    for i, log_id in enumerate(log_ids):
        log_map_dirpath = data_root / log_id / "map"

        avm = ArgoverseStaticMap.from_map_dir(log_map_dirpath, build_raster=False)
        lane_segments = avm.get_scenario_lane_segments()

        lidar_fpaths = loader.get_ordered_log_lidar_fpaths(log_id=log_id)
        timestamp0 = int(lidar_fpaths[0].stem)

        city_SE3_egot0 = loader.get_city_SE3_ego(log_id, timestamp0)

        color = DARK_GRAY_RGB_MPL

        linestyle: Union[str, Tuple[int, Tuple[int, int]]] = ""

        for ls in lane_segments:
            pts_city = ls.polygon_boundary
            pts_ego = city_SE3_egot0.inverse().transform_point_cloud(pts_city)

            for bound_type, bound_city in zip(
                [ls.left_mark_type, ls.right_mark_type], [ls.left_lane_boundary, ls.right_lane_boundary]
            ):
                if "YELLOW" in bound_type:
                    mark_color = "y"
                elif "WHITE" in bound_type:
                    mark_color = "w"
                elif "BLUE" in bound_type:
                    mark_color = "b"
                else:
                    mark_color = "grey"  # mark color could be "None"

                LOOSELY_DASHED = (0, (5, 10))

                if "DASHED" in bound_type:
                    linestyle = LOOSELY_DASHED
                else:
                    linestyle = "solid"

                bound_ego = city_SE3_egot0.inverse().transform_point_cloud(bound_city.xyz)
                ax.plot(
                    bound_ego[:, 0],
                    bound_ego[:, 1],
                    mark_color,
                    alpha=OVERLAID_MAPS_ALPHA,
                    linestyle=linestyle,
                    zorder=i,
                )

            vector_plotting_utils.plot_polygon_patch_mpl(
                polygon_pts=pts_ego, ax=ax, color=color, alpha=OVERLAID_MAPS_ALPHA, zorder=i
            )

    plt.axis("equal")
    plt.xlim(-30, 30)
    plt.ylim(-30, 30)
    plt.show()
    plt.close("all")


def plot_lane_segments(
    ax: Axes, lane_segments: Sequence[LaneSegment], lane_color: Tuple[float, float, float] = DARK_GRAY_RGB_MPL
) -> None:
    """Plot lane segments onto a Matplotlib canvas, according to their lane marking boundary type/color.

    Note: we use an approximation for SOLID_DASHED and other mixed pattern/color marking types.

    Args:
        ax: matplotlib figure to use as drawing canvas.
        lane_segments: lane segment objects. The lane markings along their boundaries will be rendered.
        lane_color: Color of the lane.
    """
    for ls in lane_segments:
        pts_city = ls.polygon_boundary
        ALPHA = 1.0  # 0.1
        vector_plotting_utils.plot_polygon_patch_mpl(
            polygon_pts=pts_city, ax=ax, color=lane_color, alpha=ALPHA, zorder=1
        )

        mark_color: str = ""
        linestyle: Union[str, Tuple[int, Tuple[int, int]]] = ""
        for bound_type, bound_city in zip(
            [ls.left_mark_type, ls.right_mark_type], [ls.left_lane_boundary, ls.right_lane_boundary]
        ):
            if "YELLOW" in bound_type:
                mark_color = "y"
            elif "WHITE" in bound_type:
                mark_color = "w"
            elif "BLUE" in bound_type:
                mark_color = "b"
            else:
                mark_color = "grey"

            LOOSELY_DASHED = (0, (5, 10))

            if "DASHED" in bound_type:
                linestyle = LOOSELY_DASHED
            else:
                linestyle = "solid"

            if "DOUBLE" in bound_type:
                left, right = polyline_utils.get_double_polylines(
                    polyline=bound_city.xyz[:, :2], width_scaling_factor=0.1
                )
                ax.plot(left[:, 0], left[:, 1], color=mark_color, alpha=ALPHA, linestyle=linestyle, zorder=2)
                ax.plot(right[:, 0], right[:, 1], color=mark_color, alpha=ALPHA, linestyle=linestyle, zorder=2)
            else:
                ax.plot(
                    bound_city.xyz[:, 0],
                    bound_city.xyz[:, 1],
                    mark_color,
                    alpha=ALPHA,
                    linestyle=linestyle,
                    zorder=2,
                )


def visualize_ego_pose_and_lane_markings(data_root: Path, log_id: str, save_figures: bool) -> None:
    """Visualize both ego-vehicle poses and the per-log local vector map.

    Crosswalks are plotted in purple. Lane segments plotted in dark gray. Ego-pose in red.

    Args:
        data_root: path to where the AV2 logs live.
        log_id: unique ID for the AV2 vehicle log.
        save_figures: whether to save each generated figure to disk.
    """
    loader = AV2SensorDataLoader(data_dir=data_root, labels_dir=data_root)

    fig = plt.figure(1, figsize=(10, 10))
    ax = fig.add_subplot(111)

    log_map_dirpath = data_root / log_id / "map"
    avm = ArgoverseStaticMap.from_map_dir(log_map_dirpath, build_raster=False)

    # retain every pose first.
    traj_ns = loader.get_subsampled_ego_trajectory(log_id, sample_rate_hz=1e9)
    # now, sample @ 1 Hz
    traj_1hz = loader.get_subsampled_ego_trajectory(log_id, sample_rate_hz=1.0)
    med: NDArrayFloat = np.median(traj_ns, axis=0)
    med_x, med_y = med

    # Derive plot area from trajectory (with radius defined in infinity norm).
    # A larger distance traveled during trajectory means we should have a larger viewing window size.
    view_radius_m: float = float(np.linalg.norm(traj_ns[-1] - traj_ns[0])) + 20
    xlims = [med_x - view_radius_m, med_x + view_radius_m]
    ylims = [med_y - view_radius_m, med_y + view_radius_m]

    crosswalk_color = PURPLE_RGB_MPL
    CROSSWALK_ALPHA = 0.6
    for pc in avm.get_scenario_ped_crossings():
        vector_plotting_utils.plot_polygon_patch_mpl(
            polygon_pts=pc.polygon[:, :2],
            ax=ax,
            color=crosswalk_color,
            alpha=CROSSWALK_ALPHA,
            zorder=3,
        )

    plot_lane_segments(ax=ax, lane_segments=avm.get_scenario_lane_segments())

    # Plot nearly continuous line for ego-pose, and show the AV's pose @ 1 Hz w/ red unfilled circles.
    ax.plot(traj_ns[:, 0], traj_ns[:, 1], color="r", zorder=4, label="Ego-vehicle pose")
    ax.scatter(x=traj_1hz[:, 0], y=traj_1hz[:, 1], s=100, marker="o", facecolors="none", edgecolors="r", zorder=4)

    plt.axis("equal")
    plt.xlim(*xlims)
    plt.ylim(*ylims)
    plt.title(f"Log {log_id}")
    plt.axis("off")
    plt.legend()
    plt.tight_layout()
    if save_figures:
        plt.savefig(SAVE_DIR / f"egovehicle_pose_on_map_{log_id}_bev.jpg", dpi=500)
    plt.show()
    plt.close("all")


@click.command(help="Run map tutorial to visualize maps for the Argoverse 2 Datasets.")
@click.option(
    "-d",
    "--data-root",
    required=True,
    help="Path to local directory where the Argoverse 2 Sensor Dataset logs are stored.",
    type=click.Path(exists=True),
)
@click.option(
    "-l",
    "--log-id",
    default="00a6ffc1-6ce9-3bc3-a060-6006e9893a1a",
    help="unique log identifier.",
    type=str,
)
@click.option(
    "--save-figures",
    default=False,
    help="whether to save each generated figure to disk.",
    type=bool,
)
def run_map_tutorial(data_root: str, log_id: str, save_figures: bool) -> None:
    """Click entry point for map tutorial."""
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    data_root_path = Path(data_root)

    SAVE_DIR.mkdir(exist_ok=True, parents=True)

    logger.info("data_root: %s, log_id: %s", data_root_path, log_id)
    single_log_teaser(data_root=data_root_path, log_id=log_id, save_figures=save_figures)
    visualize_raster_layers(data_root=data_root_path, log_id=log_id, save_figures=save_figures)
    visualize_ego_pose_and_lane_markings(data_root=data_root_path, log_id=log_id, save_figures=save_figures)
    overlaid_maps_all_logs_teaser(data_root=data_root_path)


if __name__ == "__main__":
    run_map_tutorial()
