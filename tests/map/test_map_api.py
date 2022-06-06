# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""
Tests for the Argoverse 2.0 map API.

Uses a simplified map with 2 pedestrian crossings, and 3 lane segments.
"""

from pathlib import Path

import numpy as np
import pytest

from av2.map.drivable_area import DrivableArea
from av2.map.lane_segment import LaneSegment
from av2.map.map_api import ArgoverseStaticMap
from av2.map.map_primitives import Point, Polyline
from av2.map.pedestrian_crossing import PedestrianCrossing
from av2.utils.typing import NDArrayBool, NDArrayFloat


@pytest.fixture()
def dummy_static_map(test_data_root_dir: Path) -> ArgoverseStaticMap:
    """Set up test by instantiating static map object from dummy test data.

    Args:
        test_data_root_dir: Path to the root dir for test data (provided via fixture).

    Returns:
        Static map instantiated from dummy test data.
    """
    log_map_dirpath = (
        test_data_root_dir / "static_maps" / "dummy_log_map_gs1B8ZCv7DMi8cMt5aN5rSYjQidJXvGP__2020-07-21-Z1F0076"
    )

    return ArgoverseStaticMap.from_map_dir(log_map_dirpath, build_raster=True)


@pytest.fixture(scope="module")
def full_static_map(test_data_root_dir: Path) -> ArgoverseStaticMap:
    """Set up test by instantiating static map object from full test data.

    Args:
        test_data_root_dir: Path to the root dir for test data (provided via fixture).

    Returns:
        Static map instantiated from full test data.
    """
    log_map_dirpath = (
        test_data_root_dir / "static_maps" / "full_log_map_gs1B8ZCv7DMi8cMt5aN5rSYjQidJXvGP__2020-07-21-Z1F0076"
    )
    return ArgoverseStaticMap.from_map_dir(log_map_dirpath, build_raster=True)


class TestPolyline:
    """Class for unit testing `PolyLine`."""

    def test_from_list(self) -> None:
        """Ensure object is generated correctly from a list of dictionaries."""
        points_dict_list = [{"x": 874.01, "y": -105.15, "z": -19.58}, {"x": 890.58, "y": -104.26, "z": -19.58}]
        polyline = Polyline.from_json_data(points_dict_list)

        assert isinstance(polyline, Polyline)

        assert len(polyline.waypoints) == 2
        assert polyline.waypoints[0] == Point(874.01, -105.15, -19.58)
        assert polyline.waypoints[1] == Point(890.58, -104.26, -19.58)

    def test_from_array(self) -> None:
        """Ensure object is generated correctly from a Numpy array of shape (N,3)."""
        # fmt: off
        array: NDArrayFloat = np.array([
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.],
            [9., 10., 11.]
        ])
        # fmt: on
        polyline = Polyline.from_array(array)
        assert isinstance(polyline, Polyline)
        assert len(polyline) == 4
        assert polyline.waypoints[0].x == 1
        assert polyline.waypoints[0].y == 2
        assert polyline.waypoints[0].z == 3


class TestPedestrianCrossing:
    """Class for unit testing `PedestrianCrossing`."""

    def test_from_dict(self) -> None:
        """Ensure object is generated correctly from a dictionary."""
        json_data = {
            "id": 6310421,
            "edge1": [{"x": 899.17, "y": -91.52, "z": -19.58}, {"x": 915.68, "y": -93.93, "z": -19.53}],
            "edge2": [{"x": 899.44, "y": -95.37, "z": -19.48}, {"x": 918.25, "y": -98.05, "z": -19.4}],
        }
        pedestrian_crossing = PedestrianCrossing.from_dict(json_data)

        isinstance(pedestrian_crossing, PedestrianCrossing)


class TestArgoverseStaticMap:
    """Unit test for the Argoverse 2.0 per-log map."""

    def test_get_lane_segment_successor_ids(self, dummy_static_map: ArgoverseStaticMap) -> None:
        """Ensure lane segment successors are fetched properly."""
        lane_segment_id = 93269421
        successor_ids = dummy_static_map.get_lane_segment_successor_ids(lane_segment_id)
        expected_successor_ids = [93269500]
        assert successor_ids == expected_successor_ids

        lane_segment_id = 93269500
        successor_ids = dummy_static_map.get_lane_segment_successor_ids(lane_segment_id)
        expected_successor_ids = [93269554]
        assert successor_ids == expected_successor_ids

        lane_segment_id = 93269520
        successor_ids = dummy_static_map.get_lane_segment_successor_ids(lane_segment_id)
        expected_successor_ids = [93269526]
        assert successor_ids == expected_successor_ids

    def test_lane_is_in_intersection(self, dummy_static_map: ArgoverseStaticMap) -> None:
        """Ensure the attribute describing if a lane segment is located with an intersection is fetched properly."""
        lane_segment_id = 93269421
        in_intersection = dummy_static_map.lane_is_in_intersection(lane_segment_id)
        assert isinstance(in_intersection, bool)
        assert not in_intersection

        lane_segment_id = 93269500
        in_intersection = dummy_static_map.lane_is_in_intersection(lane_segment_id)
        assert isinstance(in_intersection, bool)
        assert in_intersection

        lane_segment_id = 93269520
        in_intersection = dummy_static_map.lane_is_in_intersection(lane_segment_id)
        assert isinstance(in_intersection, bool)
        assert not in_intersection

    def test_get_lane_segment_left_neighbor_id(self, dummy_static_map: ArgoverseStaticMap) -> None:
        """Test getting a lane segment id from the left neighbor."""
        # Ensure id of lane segment (if any) that is the left neighbor to the query lane segment can be fetched properly
        lane_segment_id = 93269421
        l_neighbor_id = dummy_static_map.get_lane_segment_left_neighbor_id(lane_segment_id)
        assert l_neighbor_id is None

        lane_segment_id = 93269500
        l_neighbor_id = dummy_static_map.get_lane_segment_left_neighbor_id(lane_segment_id)
        assert l_neighbor_id is None

        lane_segment_id = 93269520
        l_neighbor_id = dummy_static_map.get_lane_segment_left_neighbor_id(lane_segment_id)
        assert l_neighbor_id == 93269421

    def test_get_lane_segment_right_neighbor_id(self, dummy_static_map: ArgoverseStaticMap) -> None:
        """Test getting a lane segment id from the right neighbor."""
        # Ensure id of lane segment (if any) that is the right neighbor to the query lane segment can be fetched
        lane_segment_id = 93269421
        r_neighbor_id = dummy_static_map.get_lane_segment_right_neighbor_id(lane_segment_id)
        assert r_neighbor_id == 93269520

        lane_segment_id = 93269500
        r_neighbor_id = dummy_static_map.get_lane_segment_right_neighbor_id(lane_segment_id)
        assert r_neighbor_id == 93269526

        lane_segment_id = 93269520
        r_neighbor_id = dummy_static_map.get_lane_segment_right_neighbor_id(lane_segment_id)
        assert r_neighbor_id == 93269458

    def test_get_scenario_lane_segment_ids(self, dummy_static_map: ArgoverseStaticMap) -> None:
        """Ensure ids of all lane segments in the local map can be fetched properly."""
        lane_segment_ids = dummy_static_map.get_scenario_lane_segment_ids()

        expected_lane_segment_ids = [93269421, 93269500, 93269520]
        assert lane_segment_ids == expected_lane_segment_ids

    def test_get_lane_segment_polygon(self, dummy_static_map: ArgoverseStaticMap) -> None:
        """Ensure lane segment polygons are fetched properly."""
        lane_segment_id = 93269421

        ls_polygon = dummy_static_map.get_lane_segment_polygon(lane_segment_id)
        assert isinstance(ls_polygon, np.ndarray)

        expected_ls_polygon: NDArrayFloat = np.array(
            [
                [874.01, -105.15, -19.58],
                [890.58, -104.26, -19.58],
                [890.29, -100.56, -19.66],
                [880.31, -101.44, -19.7],
                [873.97, -101.75, -19.7],
                [874.01, -105.15, -19.58],
            ]
        )
        np.testing.assert_allclose(ls_polygon, expected_ls_polygon)

    def test_get_lane_segment_centerline(self, dummy_static_map: ArgoverseStaticMap) -> None:
        """Ensure lane segment centerlines can be inferred and fetched properly."""
        lane_segment_id = 93269421

        centerline = dummy_static_map.get_lane_segment_centerline(lane_segment_id)
        assert isinstance(centerline, np.ndarray)

        expected_centerline: NDArrayFloat = np.array(
            [
                [873.99, -103.45, -19.64],
                [875.81871374, -103.35615034, -19.64],
                [877.64742747, -103.26230069, -19.64],
                [879.47614121, -103.16845103, -19.64],
                [881.30361375, -103.0565384, -19.63815074],
                [883.129891, -102.92723072, -19.63452059],
                [884.95616825, -102.79792304, -19.63089044],
                [886.7824455, -102.66861536, -19.62726029],
                [888.60872275, -102.53930768, -19.62363015],
                [890.435, -102.41, -19.62],
            ]
        )
        np.testing.assert_allclose(centerline, expected_centerline)

    def test_get_scenario_lane_segments(self, dummy_static_map: ArgoverseStaticMap) -> None:
        """Ensure that all LaneSegment objects in the local map can be returned as a list."""
        vector_lane_segments = dummy_static_map.get_scenario_lane_segments()
        assert isinstance(vector_lane_segments, list)
        assert all([isinstance(vls, LaneSegment) for vls in vector_lane_segments])
        assert len(vector_lane_segments) == 3

    def test_get_scenario_ped_crossings(self, dummy_static_map: ArgoverseStaticMap) -> None:
        """Ensure that all PedCrossing objects in the local map can be returned as a list."""
        ped_crossings = dummy_static_map.get_scenario_ped_crossings()
        assert isinstance(ped_crossings, list)
        assert all([isinstance(pc, PedestrianCrossing) for pc in ped_crossings])

        # fmt: off
        expected_ped_crossings = [
            PedestrianCrossing(
                id=6310407,
                edge1=Polyline.from_array(np.array(
                    [
                        [ 892.17,  -99.44,  -19.59],
                        [ 893.47, -115.4 ,  -19.45]
                    ]
                )), edge2=Polyline.from_array(np.array(
                    [
                        [ 896.06,  -98.95,  -19.52],
                        [ 897.43, -116.58,  -19.42]
                    ]
                ))
            ), PedestrianCrossing(
                id=6310421,
                edge1=Polyline.from_array(np.array(
                    [
                        [899.17, -91.52, -19.58],
                        [915.68, -93.93, -19.53]
                    ]
                )),
                edge2=Polyline.from_array(np.array(
                    [
                        [899.44, -95.37, -19.48],
                        [918.25, -98.05, -19.4]
                    ]
                )),
            )
        ]
        # fmt: on
        assert len(ped_crossings) == len(expected_ped_crossings)
        assert all([pc == expected_pc for pc, expected_pc in zip(ped_crossings, expected_ped_crossings)])

    def test_get_scenario_vector_drivable_areas(self, dummy_static_map: ArgoverseStaticMap) -> None:
        """Ensure that drivable areas are loaded and formatted correctly."""
        vector_das = dummy_static_map.get_scenario_vector_drivable_areas()
        assert isinstance(vector_das, list)
        assert len(vector_das) == 1
        assert isinstance(vector_das[0], DrivableArea)

        # examine just one sample
        vector_da = vector_das[0]
        assert vector_da.xyz.shape == (172, 3)

        # compare first and last vertex, for equality
        np.testing.assert_allclose(vector_da.xyz[0], vector_da.xyz[171])

        # fmt: off
        # compare first 4 vertices
        expected_first4_vertices: NDArrayFloat = np.array(
            [[905.09, -148.95, -19.19],
             [904.85, -141.95, -19.25],
             [904.64, -137.25, -19.28],
             [904.37, -132.55, -19.32]])
        # fmt: on
        np.testing.assert_allclose(vector_da.xyz[:4], expected_first4_vertices)

    def test_get_ground_height_at_xy(self, dummy_static_map: ArgoverseStaticMap) -> None:
        """Ensure that ground height at (x,y) locations can be retrieved properly."""
        point_cloud: NDArrayFloat = np.array(
            [
                [770.6398, -105.8351, -19.4105],  # ego-vehicle pose at one timestamp
                [943.5386, -49.6295, -19.3291],  # ego-vehicle pose at one timestamp
                [918.0960, 82.5588, -20.5742],  # ego-vehicle pose at one timestamp
                [9999999, 999999, 0],  # obviously out of bounds value for city coordinate system
                [-999999, -999999, 0],  # obviously out of bounds value for city coordinate system
            ]
        )

        assert dummy_static_map.raster_ground_height_layer is not None

        ground_height_z = dummy_static_map.raster_ground_height_layer.get_ground_height_at_xy(point_cloud)

        assert ground_height_z.shape[0] == point_cloud.shape[0]
        assert ground_height_z.dtype == np.dtype(np.float64)

        # last 2 indices should be filled with dummy values (NaN) because obviously out of bounds.
        assert np.all(np.isnan(ground_height_z[-2:]))

        # based on grid resolution, ground should be within 7 centimeters of 30cm under back axle.
        expected_ground = point_cloud[:3, 2] - 0.30
        assert np.allclose(np.absolute(expected_ground - ground_height_z[:3]), 0, atol=0.07)

    def test_get_ground_points_boolean(self, dummy_static_map: ArgoverseStaticMap) -> None:
        """Ensure that points close to the ground surface are correctly classified as `ground` category."""
        point_cloud: NDArrayFloat = np.array(
            [
                [770.6398, -105.8351, -19.4105],  # ego-vehicle pose at one timestamp
                [943.5386, -49.6295, -19.3291],  # ego-vehicle pose at one timestamp
                [918.0960, 82.5588, -20.5742],  # ego-vehicle pose at one timestamp
                [9999999, 999999, 0],  # obviously out of bounds value for city coordinate system
                [-999999, -999999, 0],  # obviously out of bounds value for city coordinate system
            ]
        )

        # first 3 points correspond to city_SE3_ego, i.e. height of rear axle in city frame
        # ~30 cm below the axle should be the ground surface.
        point_cloud -= 0.30

        assert dummy_static_map.raster_ground_height_layer is not None

        is_ground_pt = dummy_static_map.raster_ground_height_layer.get_ground_points_boolean(point_cloud)
        expected_is_ground_pt: NDArrayBool = np.array([True, True, True, False, False])
        assert is_ground_pt.dtype == np.dtype(bool)
        assert np.array_equal(is_ground_pt, expected_is_ground_pt)


def test_load_motion_forecasting_map(test_data_root_dir: Path) -> None:
    """Try to load a real map from the motion forecasting dataset."""
    mf_scenario_id = "0a1e6f0a-1817-4a98-b02e-db8c9327d151"
    mf_scenario_map_path = (
        test_data_root_dir / "forecasting_scenarios" / mf_scenario_id / f"log_map_archive_{mf_scenario_id}.json"
    )

    mf_map = ArgoverseStaticMap.from_json(mf_scenario_map_path)
    assert mf_map.log_id == mf_scenario_id
