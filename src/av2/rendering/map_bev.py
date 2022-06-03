# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Render map in bird's-eye-view (BEV).

Red represents an implicit lane marking (with no actual paint, merely inferred by road users).
"""

import copy
from dataclasses import dataclass
from typing import Final, Tuple, Union

import cv2
import numpy as np

import av2.geometry.interpolate as interp_utils
import av2.geometry.polyline_utils as polyline_utils
import av2.utils.raster as raster_utils
from av2.geometry.se3 import SE3
from av2.map.lane_segment import LaneMarkType, LaneSegment
from av2.map.map_api import ArgoverseStaticMap
from av2.rendering.color import HANDICAP_BLUE_BGR, LIGHT_RED_BGR, TRAFFIC_YELLOW1_BGR, WHITE_BGR
from av2.structures.ndgrid import BEVGrid
from av2.utils.typing import NDArrayByte, NDArrayFloat, NDArrayInt

# Apply uniform dashes. in reality, there is often a roughly 2:1 ratio between empty space and dashes.
DASH_INTERVAL_M: Final[float] = 2.0  # Every 1 meter.

# Number of sampled waypoints per polyline.
N_INTERP_PTS: Final[int] = 1000


@dataclass(frozen=True)
class BirdsEyeViewMapRenderer:
    """Rendering engine for map entities in the bird's eye view (BEV).

    Args:
        avm: Map, with vector and raster elements, for querying ground height at arbitrary locations.
        city_SE3_ego: AV pose at a single timestamp.
        bev_grid: Parameters for BEV rendering.
        render_north_as_up: Whether to render north as up in (as opposed to using the +y axis of the egovehicle
            frame -- left of the AV -- as up).
    """

    avm: ArgoverseStaticMap
    city_SE3_ego: SE3
    bev_grid: BEVGrid
    render_north_as_up: bool

    @property
    def ego_SE3_city(self) -> SE3:
        """Retrieve transformation that brings map elements in city frame into the egovehicle frame."""
        return self.city_SE3_ego.inverse()

    @property
    def city_SO3_ego(self) -> SE3:
        """Retrieve the egovehicle pose in the city reference frame."""
        return SE3(rotation=self.city_SE3_ego.rotation, translation=np.zeros(3))

    def render_lane_boundary_bev(
        self,
        img_bgr: NDArrayByte,
        lane_segment: LaneSegment,
        side: str,
        line_width_px: float,
    ) -> NDArrayByte:
        """Draw left or right lane boundary (only one is rendered here).

        Double lines are to be understood from the inside out.  e.g., DASHED_SOLID means that the dashed line
        is adjacent to the lane carrying the property and the solid line is adjacent to the neighbor lane.

        Args:
            img_bgr: Array of shape (H,W,3) representing BGR image (canvas), to render on.
            lane_segment: Vector lane segment object.
            side: Lane side to render. should be "left" or "right"
            line_width_px: Thickness (in pixels) to use when rendering the polyline.

        Returns:
            Array of shape (H,W,3) representing BGR image (canvas), with lane boundary rendered on it.

        Raises:
            ValueError: If `mark_type` is unknown.
        """
        if side == "right":
            polyline = lane_segment.right_lane_boundary.xyz
            mark_type = lane_segment.right_mark_type
        else:
            polyline = lane_segment.left_lane_boundary.xyz
            mark_type = lane_segment.left_mark_type

        # Interpolation needs to happen before rounded to integer coordinates.
        polyline_city_frame = interp_utils.interp_arc(t=N_INTERP_PTS, points=polyline)

        if "WHITE" in mark_type:
            bound_color = WHITE_BGR
        elif "YELLOW" in mark_type:
            bound_color = TRAFFIC_YELLOW1_BGR
        elif "BLUE" in mark_type:
            bound_color = HANDICAP_BLUE_BGR
        else:
            bound_color = LIGHT_RED_BGR

        if ("DOUBLE" in mark_type) or ("SOLID_DASH" in mark_type) or ("DASH_SOLID" in mark_type):
            left, right = polyline_utils.get_double_polylines(polyline_city_frame, width_scaling_factor=0.1)

        if mark_type in [
            LaneMarkType.SOLID_WHITE,
            LaneMarkType.SOLID_YELLOW,
            LaneMarkType.SOLID_BLUE,
            LaneMarkType.NONE,
        ]:
            self.render_polyline_bev(polyline_city_frame, img_bgr, bound_color, thickness_px=line_width_px)

        elif mark_type in [LaneMarkType.DOUBLE_DASH_YELLOW, LaneMarkType.DOUBLE_DASH_WHITE]:
            self.draw_dashed_polyline_bev(
                left, img_bgr, bound_color, thickness_px=line_width_px, dash_interval_m=DASH_INTERVAL_M
            )
            self.draw_dashed_polyline_bev(
                right, img_bgr, bound_color, thickness_px=line_width_px, dash_interval_m=DASH_INTERVAL_M
            )

        elif mark_type in [LaneMarkType.DOUBLE_SOLID_YELLOW, LaneMarkType.DOUBLE_SOLID_WHITE]:
            self.render_polyline_bev(left, img_bgr, bound_color, thickness_px=line_width_px)
            self.render_polyline_bev(right, img_bgr, bound_color, thickness_px=line_width_px)

        elif mark_type in [LaneMarkType.DASHED_WHITE, LaneMarkType.DASHED_YELLOW]:
            self.draw_dashed_polyline_bev(
                polyline_city_frame,
                img_bgr,
                bound_color,
                thickness_px=line_width_px,
                dash_interval_m=DASH_INTERVAL_M,
            )

        elif (mark_type in [LaneMarkType.SOLID_DASH_WHITE, LaneMarkType.SOLID_DASH_YELLOW] and side == "right") or (
            mark_type == LaneMarkType.DASH_SOLID_YELLOW and side == "left"
        ):
            self.render_polyline_bev(left, img_bgr, bound_color, thickness_px=line_width_px)
            self.draw_dashed_polyline_bev(
                right, img_bgr, bound_color, thickness_px=line_width_px, dash_interval_m=DASH_INTERVAL_M
            )

        elif (mark_type in [LaneMarkType.SOLID_DASH_WHITE, LaneMarkType.SOLID_DASH_YELLOW] and side == "left") or (
            mark_type == LaneMarkType.DASH_SOLID_YELLOW and side == "right"
        ):
            self.draw_dashed_polyline_bev(
                left, img_bgr, bound_color, thickness_px=line_width_px, dash_interval_m=DASH_INTERVAL_M
            )
            self.render_polyline_bev(right, img_bgr, bound_color, thickness_px=line_width_px)

        else:
            raise ValueError(f"Unknown marking type {mark_type}.")

        return img_bgr

    def draw_dashed_polyline_bev(
        self,
        polyline: NDArrayFloat,
        img_bgr: NDArrayByte,
        bound_color: Tuple[int, int, int],
        thickness_px: float,
        dash_interval_m: float,
        dash_frequency: int = 2,
    ) -> None:
        """Draw a dashed polyline in the bird's-eye-view (BEV).

        Generate 1 dash every N meters, with equal dash-non-dash spacing.
        Ignoring residual at ends, since assume lanes quite long.

        Args:
            polyline: Array of shape (K, 2) representing the coordinates of each line segment.
            img_bgr: Array of shape (M, N, 3), representing a 3-channel BGR image, passed by reference.
            bound_color: Tuple of shape (3,) with a BGR format color.
            thickness_px: Thickness (in pixels) to use when rendering the polyline.
            dash_interval_m: Length of one dash, in meters.
            dash_frequency: For each dash_interval_m, we will discretize the length into n sections.
                1 of n sections will contain a marked dash, and the other (n-1) spaces will be empty (non-marked).
        """
        interp_polyline, num_waypts = polyline_utils.interp_polyline_by_fixed_waypt_interval(polyline, dash_interval_m)
        for i in range(num_waypts - 1):

            # Every other segment is a gap
            # (first the next dash interval is a line, and then the dash interval is empty, ...).
            if (i % dash_frequency) != 0:
                continue

            dashed_segment_arr = interp_polyline[i : i + 2]
            self.render_polyline_bev(
                polyline_city_frame=dashed_segment_arr,
                img_bgr=img_bgr,
                bound_color=bound_color,
                thickness_px=thickness_px,
            )

    def render_polyline_bev(
        self,
        polyline_city_frame: NDArrayFloat,
        img_bgr: NDArrayByte,
        bound_color: Tuple[int, int, int],
        thickness_px: float,
    ) -> None:
        """Rasterize a polygon onto an image canvas in the bird's-eye-view (BEV).

        Args:
            polyline_city_frame: Array of shape (N,3) representing a polyline in city coordinates
                (e.g., crosswalk boundary or lane segment boundary).
            img_bgr: Array of shape (H,W,3) representing a BGR image to write to (the canvas).
            bound_color: Tuple of BGR intensities to use as color for rendering the lane boundary (i.e. polyline).
            thickness_px: Thickness (in pixels) to use when rendering the polyline.
        """
        polyline_ego_frame = self.ego_SE3_city.transform_from(copy.deepcopy(polyline_city_frame))

        # Allow option to keep North as +y in image frame.
        if self.render_north_as_up:
            polyline_ego_frame = self.city_SO3_ego.transform_from(copy.deepcopy(polyline_ego_frame))

        polyline_img_frame = self.bev_grid.transform_to_grid_coordinates(polyline_ego_frame[:, :2])

        draw_polyline_segments_cv2(
            copy.deepcopy(polyline_img_frame),
            image=img_bgr,
            color=bound_color,
            thickness_px=thickness_px,
        )

    def render_filled_polyline_bev(
        self,
        polyline_city_frame: NDArrayFloat,
        img_bgr: NDArrayByte,
        bound_color: Tuple[int, int, int],
        alpha: float = 0.5,
    ) -> NDArrayByte:
        """Rasterize a filled polygon onto an image canvas in the bird's-eye-view (BEV).

        Args:
            polyline_city_frame: Array of shape (N,3) representing a polyline in city coordinates.
                (e.g., crosswalk boundary or lane segment boundary).
            img_bgr: Array of shape (H,W,3) representing a BGR image to write to (the canvas).
            bound_color: Tuple of BGR intensities to use as color for rendering the lane boundary (i.e. polyline).
            alpha: Alpha blending coefficient.

        Returns:
            Array of shape (H,W,3) representing a 3-channel BGR image.
        """
        polyline_ego_frame = self.ego_SE3_city.transform_from(copy.deepcopy(polyline_city_frame))

        # Allow option to keep North as +y in image frame.
        if self.render_north_as_up:
            polyline_ego_frame = self.city_SO3_ego.transform_from(copy.deepcopy(polyline_ego_frame))

        polyline_img_frame = self.bev_grid.transform_to_grid_coordinates(polyline_ego_frame[:, :2])

        img_bgr_overlay = img_bgr.copy()
        polyline_img_frame = polyline_img_frame.reshape(1, -1, 2)
        polyline_img_frame = np.round(polyline_img_frame).astype(int)  # type: ignore
        cv2.fillPoly(img=img_bgr, pts=polyline_img_frame, color=bound_color)

        img_bgr = raster_utils.blend_images(img_bgr_overlay, img_bgr, alpha=alpha)
        return img_bgr


def draw_polyline_segments_cv2(
    line_segments_arr: Union[NDArrayFloat, NDArrayInt],
    image: NDArrayByte,
    color: Tuple[int, int, int],
    thickness_px: float = 1,
) -> None:
    """Draw a polyline onto an image using given line segments.

    Args:
        line_segments_arr: Array of shape (K, 2) representing the coordinates of each line segment.
            Vertices may be out of bounds (outside the image borders) and the line segment will be interpolated.
        image: Array of shape (H, W, 3), representing a 3-channel BGR image.
        color: Tuple of shape (3,) with a BGR format color.
        thickness_px: Thickness (in pixels) to use when rendering the polyline.
    """
    line_segments_arr_int: NDArrayInt = np.round(line_segments_arr).astype(int)  # type: ignore
    for i in range(len(line_segments_arr_int) - 1):

        x1 = line_segments_arr_int[i][0]
        y1 = line_segments_arr_int[i][1]
        x2 = line_segments_arr_int[i + 1][0]
        y2 = line_segments_arr_int[i + 1][1]

        # Use anti-aliasing (AA) for curves.
        image = cv2.line(image, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=thickness_px, lineType=cv2.LINE_AA)
