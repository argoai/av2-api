# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""SE(3) lie group for rigid transformations on points."""

from __future__ import annotations

import typing
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from av2.utils.typing import NDArrayFloat

if typing.TYPE_CHECKING:
    from av2.structures.cuboid import Cuboid
    from av2.structures.sweep import Sweep


@dataclass
class SE3:
    """SE(3) lie group object.

    References:
        http://ethaneade.com/lie_groups.pdf
    Args:
        rotation: Array of shape (3, 3)
        translation: Array of shape (3,)
    """

    rotation: NDArrayFloat
    translation: NDArrayFloat

    def __post_init__(self) -> None:
        """Check validity of rotation and translation arguments.

        Raises:
            ValueError: If rotation is not shape (3,3) or translation is not shape (3,).
        """
        if self.rotation.shape != (3, 3):
            raise ValueError("Rotation matrix must be shape (3,3)!")
        if self.translation.shape != (3,):
            raise ValueError("Translation vector must be shape (3,)")

    @cached_property
    def transform_matrix(self) -> NDArrayFloat:
        """4x4 homogeneous transformation matrix."""
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = self.rotation
        transform_matrix[:3, 3] = self.translation
        return transform_matrix

    def transform_from(self, point_cloud: NDArrayFloat) -> NDArrayFloat:
        """Apply the SE(3) transformation to this point cloud.

        Args:
            point_cloud: Array of shape (N, 3). If the transform represents dst_SE3_src,
                then point_cloud should consist of points in frame `src`.

        Returns:
            Array of shape (N, 3) representing the transformed point cloud, i.e., points in frame `dst`.
        """
        return point_cloud @ self.rotation.T + self.translation

    def transform_point_cloud(self, point_cloud: NDArrayFloat) -> NDArrayFloat:
        """Apply the SE(3) transformation to this point cloud.

        Args:
            point_cloud: Array of shape (N, 3). If the transform represents dst_SE3_src,
                then point_cloud should consist of points in frame `src`.

        Returns:
            Array of shape (N, 3) representing the transformed point cloud, i.e. points in frame `dst`.
        """
        return self.transform_from(point_cloud)

    def transform_sweep_from(self, sweep_src: Sweep) -> Sweep:
        """Apply the SE(3) transformation to a lidar sweep object.

        This operation will transform 3D points in the sweep to a new reference frame.
        Example usage:
            sweep_dst = dst_SE3_src.transform_sweep_from(sweep_src)

        Args:
            sweep_src: Lidar sweep in the `src` reference frame.

        Returns:
            lidar sweep, with points provided in the `dst` frame.
        """
        return sweep_src.transform_from(self)

    def transform_cuboid_from(self, cuboid_src: Cuboid) -> Cuboid:
        """Apply the SE(3) transformation to the vertices of a cuboid.

        Args:
            cuboid_src: 3D bounding box in the `src` reference frame.

        Returns:
            3D bounding box, with vertices now provided in the `dst` frame.
        """
        return cuboid_src.transform(self)

    def inverse(self) -> SE3:
        """Return the inverse of the current SE(3) transformation.

        For example, if the current object represents target_SE3_src, we will return instead src_SE3_target.

        Returns:
            Instance of SE3 class, representing inverse of SE3 transformation target_SE3_src.
        """
        return SE3(rotation=self.rotation.T, translation=self.rotation.T.dot(-self.translation))

    def compose(self, right_SE3: SE3) -> SE3:
        """Compose (right multiply) this class' transformation matrix T with another SE(3) instance.

        Algebraic representation: chained_se3 = T * right_SE3

        Args:
            right_SE3: Another instance of SE3 class.

        Returns:
            New instance of SE3 class.
        """
        chained_transform_matrix: NDArrayFloat = self.transform_matrix @ right_SE3.transform_matrix
        chained_SE3 = SE3(
            rotation=chained_transform_matrix[:3, :3],
            translation=chained_transform_matrix[:3, 3],
        )
        return chained_SE3
