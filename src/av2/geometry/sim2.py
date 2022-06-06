# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Similarity(2) lie group for rigid transformations and scaling on points.

References:
    http://ethaneade.com/lie_groups.pdf
    https://github.com/borglab/gtsam/blob/develop/gtsam/geometry/Similarity3.h
"""

from __future__ import annotations

import json
import math
import numbers
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import av2.utils.io as io_utils
from av2.utils.helpers import assert_np_array_shape
from av2.utils.typing import NDArrayFloat


@dataclass(frozen=True)
class Sim2:
    """Similarity(2) lie group object.

    Args:
        R: array of shape (2x2) representing 2d rotation matrix
        t: array of shape (2,) representing 2d translation
        s: scaling factor
    """

    R: NDArrayFloat
    t: NDArrayFloat
    s: float

    def __post_init__(self) -> None:
        """Check validity of rotation, translation, and scale arguments.

        Raises:
            ValueError: If `R` is not shape (3,3), `t` is not shape (2,),
                 or `s` is not a numeric type.
            ZeroDivisionError: If `s` is close or equal to zero.
        """
        assert_np_array_shape(self.R, (2, 2))
        assert_np_array_shape(self.t, (2,))

        if not isinstance(self.s, numbers.Number):
            raise ValueError("Scale `s` must be a numeric type!")
        if math.isclose(self.s, 0.0):
            raise ZeroDivisionError("3x3 matrix formation would require division by zero")

    @property
    def theta_deg(self) -> float:
        """Recover the rotation angle `theta` (in degrees) from the 2d rotation matrix.

        Note: the first column of the rotation matrix R provides sine and cosine of theta,
            since R is encoded as [c,-s]
                                  [s, c]

        We use the following identity: tan(theta) = s/c = (opp/hyp) / (adj/hyp) = opp/adj

        Returns:
            Rotation angle from the 2d rotation matrix.
        """
        c, s = self.R[0, 0], self.R[1, 0]
        theta_rad = np.arctan2(s, c)
        return float(np.rad2deg(theta_rad))

    def __repr__(self) -> str:
        """Return a human-readable string representation of the class."""
        trans = np.round(self.t, 2)
        return f"Angle (deg.): {self.theta_deg:.1f}, Trans.: {trans}, Scale: {self.s:.1f}"

    def __eq__(self, other: object) -> bool:
        """Check for equality with other Sim(2) object."""
        if not isinstance(other, Sim2):
            return False

        if not np.isclose(self.scale, other.scale):
            return False

        if not np.allclose(self.rotation, other.rotation):
            return False

        if not np.allclose(self.translation, other.translation):
            return False

        return True

    @property
    def rotation(self) -> NDArrayFloat:
        """Return the 2x2 rotation matrix."""
        return self.R

    @property
    def translation(self) -> NDArrayFloat:
        """Return the (2,) translation vector."""
        return self.t

    @property
    def scale(self) -> float:
        """Return the scale."""
        return self.s

    @property
    def matrix(self) -> NDArrayFloat:
        """Calculate 3*3 matrix group equivalent."""
        T = np.zeros((3, 3))
        T[:2, :2] = self.R
        T[:2, 2] = self.t
        T[2, 2] = 1 / self.s
        return T

    def compose(self, S: Sim2) -> Sim2:
        """Composition with another Sim2.

        This can be understood via block matrix multiplication, if self is parameterized as (R1,t1,s1)
        and if `S` is parameterized as (R2,t2,s2):

        [R1  t1]   [R2  t2]   [R1 @ R2   R1@t2 + t1/s2]
        [0 1/s1] @ [0 1/s2] = [ 0          1/(s1*s2)  ]

        Args:
            S: Similarity(2) transformation.

        Returns:
            Composed Similarity(2) transformation.
        """
        # fmt: off
        return Sim2(
            R=self.R @ S.R,
            t=self.R @ S.t + ((1.0 / S.s) * self.t), 
            s=self.s * S.s
        )
        # fmt: on

    def inverse(self) -> Sim2:
        """Return the inverse."""
        Rt = self.R.T
        sRt: NDArrayFloat = -Rt @ (self.s * self.t)
        return Sim2(Rt, sRt, 1.0 / self.s)

    def transform_from(self, points_xy: NDArrayFloat) -> NDArrayFloat:
        """Transform point cloud from reference frame a to b.

        If the points are in frame A, and our Sim(3) transform is defines as bSa, then we get points
        back in frame B:
            p_b = bSa * p_a
        Action on a point p is s*(R*p+t).

        Args:
            points_xy: Nx2 array representing 2d points in frame A.

        Returns:
            Nx2 array representing 2d points in frame B.

        Raises:
            ValueError: if `points_xy` isn't in R^2.
        """
        if not points_xy.ndim == 2:
            raise ValueError("Input points are not 2-dimensional.")
        assert_np_array_shape(points_xy, (None, 2))
        # (2,2) x (2,N) + (2,1) = (2,N) -> transpose
        transformed_point_cloud: NDArrayFloat = (points_xy @ self.R.T) + self.t

        # now scale points
        return transformed_point_cloud * self.s

    def transform_point_cloud(self, points_xy: NDArrayFloat) -> NDArrayFloat:
        """Alias for `transform_from()`, for synchrony w/ API provided by SE(2) and SE(3) classes."""
        return self.transform_from(points_xy)

    def save_as_json(self, save_fpath: Path) -> None:
        """Save the Sim(2) object to a JSON representation on disk.

        Args:
            save_fpath: path to where json file should be saved
        """
        dict_for_serialization = {
            "R": self.rotation.flatten().tolist(),
            "t": self.translation.flatten().tolist(),
            "s": self.scale,
        }
        io_utils.save_json_dict(save_fpath, dict_for_serialization)

    @classmethod
    def from_json(cls, json_fpath: Path) -> Sim2:
        """Generate class inst. from a JSON file containing Sim(2) parameters as flattened matrices (row-major)."""
        with open(json_fpath, "r") as f:
            json_data = json.load(f)

        R: NDArrayFloat = np.array(json_data["R"]).reshape(2, 2)
        t: NDArrayFloat = np.array(json_data["t"]).reshape(2)
        s = float(json_data["s"])
        return cls(R, t, s)

    @classmethod
    def from_matrix(cls, T: NDArrayFloat) -> Sim2:
        """Generate class instance from a 3x3 Numpy matrix."""
        if np.isclose(T[2, 2], 0.0):
            raise ZeroDivisionError("Sim(2) scale calculation would lead to division by zero.")

        R = T[:2, :2]
        t = T[:2, 2]
        s = 1 / T[2, 2]
        return cls(R, t, s)
