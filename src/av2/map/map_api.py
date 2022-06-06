# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""API for loading and Argoverse 2.0 maps.

These include left and right lane boundaries, instead of only lane centerlines,
as was the case in Argoverse 1.0 and 1.1.

Separate map data (files) is provided for each log/scenario. This local map data represents
map entities that fall within some distance according to l-infinity norm from the trajectory
of the egovehicle (AV).
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Final, List, Optional, Tuple, Union

import numpy as np

import av2.geometry.interpolate as interp_utils
import av2.utils.dilation_utils as dilation_utils
import av2.utils.io as io_utils
import av2.utils.raster as raster_utils
from av2.geometry.sim2 import Sim2
from av2.map.drivable_area import DrivableArea
from av2.map.lane_segment import LaneSegment
from av2.map.pedestrian_crossing import PedestrianCrossing
from av2.utils.typing import NDArrayBool, NDArrayByte, NDArrayFloat, NDArrayInt

# 1 meter resolution is insufficient for the online-generated drivable area and ROI raster grids
# these grids can be generated at an arbitrary resolution, from vector (polygon) objects.
ONLINE_RASTER_RESOLUTION_M: Final[float] = 0.1  # 10 cm resolution
ONLINE_RASTER_RESOLUTION_SCALE: Final[float] = 1 / ONLINE_RASTER_RESOLUTION_M

GROUND_HEIGHT_THRESHOLD_M: Final[float] = 0.3  # 30 centimeters
ROI_ISOCONTOUR_M: Final[float] = 5.0  # in meters
ROI_ISOCONTOUR_GRID: Final[float] = ROI_ISOCONTOUR_M * ONLINE_RASTER_RESOLUTION_SCALE

WPT_INFINITY_NORM_INTERP_NUM: Final[int] = 50


logger = logging.getLogger(__name__)


class RasterLayerType(str, Enum):
    """Raster layer types."""

    ROI = "ROI"
    DRIVABLE_AREA = "DRIVABLE_AREA"
    GROUND_HEIGHT = "GROUND_HEIGHT"


@dataclass(frozen=True)
class RasterMapLayer:
    """Data sampled at points along a regular grid, and a mapping from city coordinates to grid array coordinates."""

    array: Union[NDArrayByte, NDArrayFloat]
    array_Sim2_city: Sim2

    def get_raster_values_at_coords(
        self, points_xyz: NDArrayFloat, fill_value: Union[float, int]
    ) -> Union[NDArrayFloat, NDArrayInt]:
        """Index into a raster grid and extract values corresponding to city coordinates.

        Note: a conversion is required between city coordinates and raster grid coordinates, via Sim(2).

        Args:
            points_xyz: array of shape (N,2) or (N,3) representing coordinates in the city coordinate frame.
            fill_value: float representing default "raster" return value for out-of-bounds queries.

        Returns:
            raster_values: array of shape (N,) representing raster values at the N query coordinates.
        """
        # Note: we do NOT round here, because we need to enforce scaled discretization.
        city_coords = points_xyz[:, :2]

        npyimage_coords = self.array_Sim2_city.transform_point_cloud(city_coords)
        npyimage_coords = npyimage_coords.astype(np.int64)

        # out of bounds values will default to the fill value, and will not be indexed into the array.
        # index in at (x,y) locations, which are (y,x) in the image
        raster_values = np.full((npyimage_coords.shape[0]), fill_value)
        # generate boolean array indicating whether the value at each index represents a valid coordinate.
        ind_valid_pts = (
            (npyimage_coords[:, 1] >= 0)
            * (npyimage_coords[:, 1] < self.array.shape[0])
            * (npyimage_coords[:, 0] >= 0)
            * (npyimage_coords[:, 0] < self.array.shape[1])
        )
        raster_values[ind_valid_pts] = self.array[npyimage_coords[ind_valid_pts, 1], npyimage_coords[ind_valid_pts, 0]]
        return raster_values


@dataclass(frozen=True)
class GroundHeightLayer(RasterMapLayer):
    """Rasterized ground height map layer.

    Stores the "ground_height_matrix" and also the array_Sim2_city: Sim(2) that produces takes point in city
    coordinates to numpy image/matrix coordinates, e.g. p_npyimage = array_Transformation_city * p_city
    """

    @classmethod
    def from_file(cls, log_map_dirpath: Path) -> GroundHeightLayer:
        """Load ground height values (w/ values at 30 cm resolution) from .npy file, and associated Sim(2) mapping.

        Note: ground height values are stored on disk as a float16 2d-array, but cast to float32 once loaded for
        compatibility with matplotlib.

        Args:
            log_map_dirpath: path to directory which contains map files associated with one specific log/scenario.

        Returns:
            The ground height map layer.

        Raises:
            RuntimeError: If raster ground height layer file is missing or Sim(2) mapping from city to image coordinates
                is missing.
        """
        ground_height_npy_fpaths = sorted(log_map_dirpath.glob("*_ground_height_surface____*.npy"))
        if not len(ground_height_npy_fpaths) == 1:
            raise RuntimeError("Raster ground height layer file is missing")

        Sim2_json_fpaths = sorted(log_map_dirpath.glob("*___img_Sim2_city.json"))
        if not len(Sim2_json_fpaths) == 1:
            raise RuntimeError("Sim(2) mapping from city to image coordinates is missing")

        # load the file with rasterized values
        ground_height_array: NDArrayFloat = np.load(ground_height_npy_fpaths[0])

        array_Sim2_city = Sim2.from_json(Sim2_json_fpaths[0])

        return cls(array=ground_height_array.astype(float), array_Sim2_city=array_Sim2_city)

    def get_ground_points_boolean(self, points_xyz: NDArrayFloat) -> NDArrayBool:
        """Check whether each 3d point is likely to be from the ground surface.

        Args:
            points_xyz: Numpy array of shape (N,3) representing 3d coordinates of N query locations.

        Returns:
            Numpy array of shape (N,) where ith entry is True if the 3d point (e.g. a LiDAR return) is likely
                located on the ground surface.

        Raises:
            ValueError: If `points_xyz` aren't 3d.
        """
        if points_xyz.shape[1] != 3:
            raise ValueError("3-dimensional points must be provided to classify them as `ground` with the map.")

        ground_height_values = self.get_ground_height_at_xy(points_xyz)
        z = points_xyz[:, 2]
        near_ground: NDArrayBool = np.absolute(z - ground_height_values) <= GROUND_HEIGHT_THRESHOLD_M
        underground: NDArrayBool = z < ground_height_values
        is_ground_boolean_arr: NDArrayBool = near_ground | underground
        return is_ground_boolean_arr

    def get_rasterized_ground_height(self) -> Tuple[NDArrayFloat, Sim2]:
        """Get ground height matrix along with Sim(2) that maps matrix coordinates to city coordinates.

        Returns:
            ground_height_matrix:
            array_Sim2_city: Sim(2) that produces takes point in city coordinates to image coordinates, e.g.
                    p_image = image_Transformation_city * p_city
        """
        ground_height_matrix: NDArrayFloat = self.array.astype(float)
        return ground_height_matrix, self.array_Sim2_city

    def get_ground_height_at_xy(self, points_xyz: NDArrayFloat) -> NDArrayFloat:
        """Get ground height for each of the xy locations for all points {(x,y,z)} in a point cloud.

        Args:
            points_xyz: Numpy array of shape (K,2) or (K,3)

        Returns:
            Numpy array of shape (K,)
        """
        ground_height_values: NDArrayFloat = self.get_raster_values_at_coords(points_xyz, fill_value=np.nan).astype(
            float
        )
        return ground_height_values


@dataclass(frozen=True)
class DrivableAreaMapLayer(RasterMapLayer):
    """Rasterized drivable area map layer.

    This provides the "drivable area" as a binary segmentation mask in the bird's eye view.
    """

    @classmethod
    def from_vector_data(cls, drivable_areas: List[DrivableArea]) -> DrivableAreaMapLayer:
        """Return a drivable area map from vector data.

        NOTE: This function provides "drivable area" as a binary segmentation mask in the bird's eye view.

        Args:
            drivable_areas: List of drivable areas.

        Returns:
            Driveable area map layer.
        """
        # We compute scene boundaries on the fly, based on the vertices of all drivable area polygons.
        # These scene boundaries are used to define the raster grid extents.
        x_min, y_min, x_max, y_max = compute_data_bounds(drivable_areas)

        # The resolution of the rasterization will affect image dimensions.
        array_s_city = ONLINE_RASTER_RESOLUTION_SCALE
        img_h = int((y_max - y_min + 1) * array_s_city)
        img_w = int((x_max - x_min + 1) * array_s_city)

        # scale determines the resolution of the raster DA layer.
        array_Sim2_city = Sim2(R=np.eye(2), t=np.array([-x_min, -y_min]), s=array_s_city)

        # convert vertices for each polygon from a 3d array in city coordinates, to a 2d array
        # in image/array coordinates.
        da_polygons_img = []
        for da_polygon_city in drivable_areas:
            da_polygon_img = array_Sim2_city.transform_from(da_polygon_city.xyz[:, :2])
            da_polygon_img = np.round(da_polygon_img).astype(np.int32)
            da_polygons_img.append(da_polygon_img)

        da_mask = raster_utils.get_mask_from_polygons(da_polygons_img, img_h, img_w)

        return cls(array=da_mask, array_Sim2_city=array_Sim2_city)


@dataclass(frozen=True)
class RoiMapLayer(RasterMapLayer):
    """Rasterized Region of Interest (RoI) map layer.

    This layer provides the "region of interest" as a binary segmentation mask in the bird's eye view.
    """

    @classmethod
    def from_drivable_area_layer(cls, drivable_area_layer: DrivableAreaMapLayer) -> RoiMapLayer:
        """Rasterize and return 3d vector drivable area as a 2d array, and dilate it by 5 meters, to return a ROI mask.

        Args:
            drivable_area_layer: Drivable map layer.

        Returns:
            ROI Layer, containing a (M,N) matrix representing a binary segmentation for the region of interest,
                and `array_Sim2_city`, Similarity(2) transformation that transforms point in the city coordinates to
                2d array coordinates:
                    p_array  = array_Sim2_city * p_city
        """
        # initialize ROI as zero-level isocontour of drivable area, and the dilate to 5-meter isocontour
        roi_mat_init: NDArrayByte = copy.deepcopy(drivable_area_layer.array).astype(np.uint8)
        roi_mask = dilation_utils.dilate_by_l2(roi_mat_init, dilation_thresh=ROI_ISOCONTOUR_GRID)

        return cls(array=roi_mask, array_Sim2_city=drivable_area_layer.array_Sim2_city)


def compute_data_bounds(drivable_areas: List[DrivableArea]) -> Tuple[int, int, int, int]:
    """Find the minimum and maximum coordinates along the x and y axes for a set of drivable areas.

    Args:
        drivable_areas: list of drivable area objects, defined in the city coordinate frame.

    Returns:
        xmin: float representing minimum x-coordinate of any vertex of any provided drivable area.
        ymin: float representing minimum y-coordinate, as above.
        xmax: float representing maximum x-coordinate, as above.
        ymax: float representing maximum y-coordinate, as above.
    """
    xmin = math.floor(min([da.xyz[:, 0].min() for da in drivable_areas]))
    ymin = math.floor(min([da.xyz[:, 1].min() for da in drivable_areas]))
    xmax = math.ceil(max([da.xyz[:, 0].max() for da in drivable_areas]))
    ymax = math.ceil(max([da.xyz[:, 1].max() for da in drivable_areas]))

    return xmin, ymin, xmax, ymax


@dataclass
class ArgoverseStaticMap:
    """API to interact with a local map for a single log (within a single city).

    Nodes in the lane graph are lane segments. Edges in the lane graph provided the lane segment connectivity, via
    left and right neighbors and successors.

    Lane segments are parameterized by 3d waypoints representing their left and right boundaries.
        Note: predecessors are implicit and available by reversing the directed graph dictated by successors.

    Args:
        log_id: unique identifier for log/scenario.
        vector_drivable_areas: drivable area polygons. Each polygon is represented by a Nx3 array of its vertices.
            Note: the first and last polygon vertex are identical (i.e. the first index is repeated).
        vector_lane_segments: lane segments that are local to this log/scenario. Consists of a mapping from
            lane segment ID to vector lane segment object, parameterized in 3d.
        vector_pedestrian_crossings: all pedestrian crossings (i.e. crosswalks) that are local to this log/scenario.
            Note: the lookup index is simply a list, rather than a dictionary-based mapping, since pedestrian crossings
            are not part of a larger graph.
        raster_drivable_area_layer: 2d raster representation of drivable area segmentation.
        raster_roi_layer: 2d raster representation of region of interest segmentation.
        raster_ground_height_layer: not provided for Motion Forecasting-specific scenarios/logs.
    """

    # handle out-of-bounds lane segment ids with ValueError

    log_id: str
    vector_drivable_areas: Dict[int, DrivableArea]
    vector_lane_segments: Dict[int, LaneSegment]
    vector_pedestrian_crossings: Dict[int, PedestrianCrossing]
    raster_drivable_area_layer: Optional[DrivableAreaMapLayer]
    raster_roi_layer: Optional[RoiMapLayer]
    raster_ground_height_layer: Optional[GroundHeightLayer]

    @classmethod
    def from_json(cls, static_map_path: Path) -> ArgoverseStaticMap:
        """Instantiate an Argoverse static map object (without raster data) from a JSON file containing map data.

        Args:
            static_map_path: Path to the JSON file containing map data. The file name must match
                the following pattern: "log_map_archive_{log_id}.json".

        Returns:
            An Argoverse HD map.
        """
        log_id = static_map_path.stem.split("log_map_archive_")[1]
        vector_data = io_utils.read_json_file(static_map_path)

        vector_drivable_areas = {da["id"]: DrivableArea.from_dict(da) for da in vector_data["drivable_areas"].values()}
        vector_lane_segments = {ls["id"]: LaneSegment.from_dict(ls) for ls in vector_data["lane_segments"].values()}

        if "pedestrian_crossings" not in vector_data:
            logger.error("Missing Pedestrian crossings!")
            vector_pedestrian_crossings = {}
        else:
            vector_pedestrian_crossings = {
                pc["id"]: PedestrianCrossing.from_dict(pc) for pc in vector_data["pedestrian_crossings"].values()
            }

        return cls(
            log_id=log_id,
            vector_drivable_areas=vector_drivable_areas,
            vector_lane_segments=vector_lane_segments,
            vector_pedestrian_crossings=vector_pedestrian_crossings,
            raster_drivable_area_layer=None,
            raster_roi_layer=None,
            raster_ground_height_layer=None,
        )

    @classmethod
    def from_map_dir(cls, log_map_dirpath: Path, build_raster: bool = False) -> ArgoverseStaticMap:
        """Instantiate an Argoverse map object from data stored within a map data directory.

        Note: the ground height surface file and associated coordinate mapping is not provided for the
        2.0 Motion Forecasting dataset, so `build_raster` defaults to False. If raster functionality is
        desired, users should pass `build_raster` to True (e.g. for the Sensor Datasets and Map Change Datasets).

        Args:
            log_map_dirpath: Path to directory containing scenario-specific map data,
                JSON file must follow this schema: "log_map_archive_{log_id}.json".
            build_raster: Whether to rasterize drivable areas, compute region of interest BEV binary segmentation,
                and to load raster ground height from disk (when available).

        Returns:
            The HD map.

        Raises:
            RuntimeError: If the vector map data JSON file is missing.
        """
        # Load vector map data from JSON file
        vector_data_fnames = sorted(log_map_dirpath.glob("log_map_archive_*.json"))
        if not len(vector_data_fnames) == 1:
            raise RuntimeError(f"JSON file containing vector map data is missing (searched in {log_map_dirpath})")
        vector_data_fname = vector_data_fnames[0]

        vector_data_json_path = log_map_dirpath / vector_data_fname
        static_map = cls.from_json(vector_data_json_path)
        static_map.log_id = log_map_dirpath.parent.stem

        # Avoid file I/O and polygon rasterization when not needed
        if build_raster:
            drivable_areas: List[DrivableArea] = list(static_map.vector_drivable_areas.values())
            static_map.raster_drivable_area_layer = DrivableAreaMapLayer.from_vector_data(drivable_areas=drivable_areas)
            static_map.raster_roi_layer = RoiMapLayer.from_drivable_area_layer(static_map.raster_drivable_area_layer)
            static_map.raster_ground_height_layer = GroundHeightLayer.from_file(log_map_dirpath)

        return static_map

    def get_scenario_vector_drivable_areas(self) -> List[DrivableArea]:
        """Fetch a list of polygons, whose union represents the drivable area for the log/scenario.

        NOTE: this function provides drivable areas in vector, not raster, format).

        Returns:
            List of drivable area polygons.
        """
        return list(self.vector_drivable_areas.values())

    def get_lane_segment_successor_ids(self, lane_segment_id: int) -> Optional[List[int]]:
        """Get lane id for the lane sucessor of the specified lane_segment_id.

        Args:
            lane_segment_id: unique identifier for a lane segment within a log scenario map (within a single city).

        Returns:
            successor_ids: list of integers, representing lane segment IDs of successors. If there are no
                successor lane segments, then the list will be empty.
        """
        successor_ids = self.vector_lane_segments[lane_segment_id].successors
        return successor_ids

    def get_lane_segment_left_neighbor_id(self, lane_segment_id: int) -> Optional[int]:
        """Get id of lane segment that is the left neighbor (if any exists) to the query lane segment id.

        Args:
            lane_segment_id: unique identifier for a lane segment within a log scenario map (within a single city).

        Returns:
            integer representing id of left neighbor to the query lane segment id, or None if no such neighbor exists.
        """
        return self.vector_lane_segments[lane_segment_id].left_neighbor_id

    def get_lane_segment_right_neighbor_id(self, lane_segment_id: int) -> Optional[int]:
        """Get id of lane segment that is the right neighbor (if any exists) to the query lane segment id.

        Args:
            lane_segment_id: unique identifier for a lane segment within a log scenario map (within a single city).

        Returns:
            integer representing id of right neighbor to the query lane segment id, or None if no such neighbor exists.
        """
        return self.vector_lane_segments[lane_segment_id].right_neighbor_id

    def get_scenario_lane_segment_ids(self) -> List[int]:
        """Get ids of all lane segments that are local to this log/scenario (according to l-infinity norm).

        Returns:
            list containing ids of local lane segments
        """
        return list(self.vector_lane_segments.keys())

    def get_lane_segment_centerline(self, lane_segment_id: int) -> NDArrayFloat:
        """Infer a 3D centerline for any particular lane segment by forming a ladder of left and right waypoints.

        Args:
            lane_segment_id: unique identifier for a lane segment within a log scenario map (within a single city).

        Returns:
            Numpy array of shape (N,3).
        """
        left_ln_bound = self.vector_lane_segments[lane_segment_id].left_lane_boundary.xyz
        right_ln_bound = self.vector_lane_segments[lane_segment_id].right_lane_boundary.xyz

        lane_centerline, _ = interp_utils.compute_midpoint_line(
            left_ln_boundary=left_ln_bound,
            right_ln_boundary=right_ln_bound,
            num_interp_pts=interp_utils.NUM_CENTERLINE_INTERP_PTS,
        )
        return lane_centerline

    def get_lane_segment_polygon(self, lane_segment_id: int) -> NDArrayFloat:
        """Return an array contained coordinates of vertices that represent the polygon's boundary.

        Args:
            lane_segment_id: unique identifier for a lane segment within a log scenario map (within a single city).

        Returns:
            Array of polygon boundary (K,3), with identical and last boundary points
        """
        return self.vector_lane_segments[lane_segment_id].polygon_boundary

    def lane_is_in_intersection(self, lane_segment_id: int) -> bool:
        """Check if the specified lane_segment_id falls within an intersection.

        Args:
            lane_segment_id: unique identifier for a lane segment within a log scenario map (within a single city).

        Returns:
            boolean indicating if the lane segment falls within an intersection
        """
        return self.vector_lane_segments[lane_segment_id].is_intersection

    def get_scenario_ped_crossings(self) -> List[PedestrianCrossing]:
        """Return a list of all pedestrian crossing objects that are local to this log/scenario (by l-infinity norm).

        Returns:
            lpcs: local pedestrian crossings
        """
        return list(self.vector_pedestrian_crossings.values())

    def get_nearby_ped_crossings(self, query_center: NDArrayFloat, search_radius_m: float) -> List[PedestrianCrossing]:
        """Return nearby pedestrian crossings.

        Returns pedestrian crossings for which any waypoint of their boundary falls within `search_radius_m` meters
        of query center, by l-infinity norm.

        Search radius defined in l-infinity norm (could also provide an l2 norm variant).

        Args:
            query_center: Numpy array of shape (2,) representing 2d query center.
            search_radius_m: distance threshold in meters (by infinity norm) to use for search.

        Raises:
            NotImplementedError: Always (not implemented!).
        """
        raise NotImplementedError("This method isn't currently supported.")

    def get_scenario_lane_segments(self) -> List[LaneSegment]:
        """Return a list of all lane segments objects that are local to this log/scenario.

        Returns:
            vls_list: lane segments local to this scenario (any waypoint within 100m by L2 distance)
        """
        return list(self.vector_lane_segments.values())

    def get_nearby_lane_segments(self, query_center: NDArrayFloat, search_radius_m: float) -> List[LaneSegment]:
        """Return the nearby lane segments.

        Return lane segments for which any waypoint of their lane boundaries falls
            within search_radius meters of query center, by l-infinity norm.

        Args:
            query_center: Numpy array of shape (2,) representing 2d query center.
            search_radius_m: distance threshold in meters (by infinity norm) to use for search.

        Returns:
            ls_list: lane segments that fall within the requested search radius.
        """
        scenario_lane_segments = self.get_scenario_lane_segments()
        return [
            ls for ls in scenario_lane_segments if ls.is_within_l_infinity_norm_radius(query_center, search_radius_m)
        ]

    def remove_ground_surface(self, points_xyz: NDArrayFloat) -> NDArrayFloat:
        """Get a collection of 3d points, snap them to the grid, perform the O(1) raster map queries.

        If our z-height is within THRESHOLD of that grid's z-height, then we keep it; otherwise, discard it.

        Args:
            points_xyz: Numpy array of shape (N,3) representing 3d coordinates of N query locations.

        Returns:
            subset of original point cloud, with ground points removed
        """
        is_ground_boolean_arr = self.get_ground_points_boolean(points_xyz)
        filtered_points_xyz: NDArrayFloat = points_xyz[~is_ground_boolean_arr]
        return filtered_points_xyz

    def get_ground_points_boolean(self, points_xyz: NDArrayFloat) -> NDArrayBool:
        """Check whether each 3d point is likely to be from the ground surface.

        Args:
            points_xyz: Numpy array of shape (N,3) representing 3d coordinates of N query locations.

        Returns:
            Numpy array of shape (N,) where ith entry is True if the 3d point
                (e.g. a LiDAR return) is likely located on the ground surface.

        Raises:
            ValueError: If `self.raster_ground_height_layer` is `None`.
        """
        if self.raster_ground_height_layer is None:
            raise ValueError("Raster ground height is not loaded!")

        return self.raster_ground_height_layer.get_ground_points_boolean(points_xyz)

    def remove_non_drivable_area_points(self, points_xyz: NDArrayFloat) -> NDArrayFloat:
        """Decimate the point cloud to the drivable area only.

        Get a 3d point, snap it to the grid, perform the O(1) raster map query.

        Args:
            points_xyz: Numpy array of shape (N,3) representing 3d coordinates of N query locations.

        Returns:
            subset of original point cloud, returning only those points lying within the drivable area.
        """
        is_da_boolean_arr = self.get_raster_layer_points_boolean(points_xyz, layer_name=RasterLayerType.DRIVABLE_AREA)
        filtered_points_xyz: NDArrayFloat = points_xyz[is_da_boolean_arr]
        return filtered_points_xyz

    def remove_non_roi_points(self, points_xyz: NDArrayFloat) -> NDArrayFloat:
        """Decimate the point cloud to the Region of Interest (ROI) area only.

        Get a 3d point, snap it to the grid, perform the O(1) raster map query.

        Args:
            points_xyz: Numpy array of shape (N,3) representing 3d coordinates of N query locations.

        Returns:
            subset of original point cloud, returning only those points lying within the ROI.
        """
        is_da_boolean_arr = self.get_raster_layer_points_boolean(points_xyz, layer_name=RasterLayerType.ROI)
        filtered_points_xyz: NDArrayFloat = points_xyz[is_da_boolean_arr]
        return filtered_points_xyz

    def get_rasterized_drivable_area(self) -> Tuple[NDArrayByte, Sim2]:
        """Get the drivable area along with Sim(2) that maps matrix coordinates to city coordinates.

        Returns:
            da_matrix: Numpy array of shape (M,N) representing binary values for drivable area,
                or None if `build_raster=False`.
            array_Sim2_city: Sim(2) that produces takes point in city coordinates to Numpy array coordinates, e.g.
                    p_array = array_Transformation_city * p_city

        Raises:
            ValueError: If `self.raster_drivable_area_layer` is `None`.
        """
        if self.raster_drivable_area_layer is None:
            raise ValueError("Raster drivable area is not loaded!")

        raster_drivable_area_layer: NDArrayByte = self.raster_drivable_area_layer.array.astype(np.uint8)
        return raster_drivable_area_layer, self.raster_drivable_area_layer.array_Sim2_city

    def get_rasterized_roi(self) -> Tuple[NDArrayByte, Sim2]:
        """Get the drivable area along with Sim(2) that maps matrix coordinates to city coordinates.

        Returns:
            da_matrix: Numpy array of shape (M,N) representing binary values for drivable area.
            array_Sim2_city: Sim(2) that produces takes point in city coordinates to numpy image, e.g.
                    p_npyimage = npyimage_Transformation_city * p_city

        Raises:
            ValueError: If `self.raster_roi_layer` is `None`.
        """
        if self.raster_roi_layer is None:
            raise ValueError("Raster ROI is not loaded!")

        raster_roi_layer: NDArrayByte = self.raster_roi_layer.array.astype(np.uint8)
        return raster_roi_layer, self.raster_roi_layer.array_Sim2_city

    def get_raster_layer_points_boolean(self, points_xyz: NDArrayFloat, layer_name: RasterLayerType) -> NDArrayBool:
        """Query the binary segmentation layers (drivable area and ROI) at specific coordinates, to check values.

        Args:
            points_xyz: Numpy array of shape (N,3) representing 3d coordinates of N query locations.
            layer_name: enum indicating layer name, for either region-of-interest or drivable area.

        Returns:
            Numpy array of shape (N,) where i'th entry is True if binary segmentation is
                equal to 1 at the i'th point coordinate (i.e. is within the ROI, or within the drivable area,
                depending upon `layer_name` argument).

        Raises:
            ValueError: If `self.raster_roi_layer`, `self.raster_drivable_area_layer` is `None`. Additionally,
                if `layer_name` is not `roi` or `driveable_area`.
        """
        if layer_name == RasterLayerType.ROI:
            if self.raster_roi_layer is None:
                raise ValueError("Raster ROI is not loaded!")
            layer_values = self.raster_roi_layer.get_raster_values_at_coords(points_xyz, fill_value=0)
        elif layer_name == RasterLayerType.DRIVABLE_AREA:
            if self.raster_drivable_area_layer is None:
                raise ValueError("Raster drivable area is not loaded!")
            layer_values = self.raster_drivable_area_layer.get_raster_values_at_coords(points_xyz, fill_value=0)
        else:
            raise ValueError("layer_name should be either `roi` or `drivable_area`.")

        is_layer_boolean_arr: NDArrayBool = layer_values == 1.0
        return is_layer_boolean_arr

    def append_height_to_2d_city_pt_cloud(self, points_xy: NDArrayFloat) -> NDArrayFloat:
        """Accept 2d point cloud in xy plane and returns a 3d point cloud (xyz) by querying map for ground height.

        Args:
            points_xy: Numpy array of shape (N,2) representing 2d coordinates of N query locations.

        Returns:
            Numpy array of shape (N,3) representing 3d coordinates on the ground surface at N (x,y) query locations.

        Raises:
            ValueError: If `self.raster_ground_height_layer` is `None` or input is not a set of 2d coordinates.
        """
        if self.raster_ground_height_layer is None:
            raise ValueError("Raster ground height is not loaded!")

        if points_xy.shape[1] != 2:
            raise ValueError("Input query points must have shape (N,2")

        points_z = self.raster_ground_height_layer.get_ground_height_at_xy(points_xy)
        points_xyz: NDArrayFloat = np.hstack([points_xy, points_z[:, np.newaxis]])
        return points_xyz
