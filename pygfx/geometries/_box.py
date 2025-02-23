import numpy as np

from ..resources import Buffer
from ._base import Geometry
from ._plane import generate_plane
from .utils import merge, transform


class BoxGeometry(Geometry):
    """A geometry defining a cuboid."""

    def __init__(
        self,
        width=1,
        height=1,
        depth=1,
        width_segments=1,
        height_segments=1,
        depth_segments=1,
    ):
        super().__init__()

        # y z
        # |/
        #  -- x

        cube_dim = np.array([width, height, depth], dtype=np.float32)
        cube_seg = np.array(
            [width_segments, height_segments, depth_segments], dtype=np.uint32
        )
        cube_normal_up = np.array(
            [
                [[1, 0, 0], [0, 1, 0]],  # right
                [[-1, 0, 0], [0, 1, 0]],  # left
                [[0, 1, 0], [1, 0, 0]],  # top
                [[0, -1, 0], [-1, 0, 0]],  # bottom
                [[0, 0, 1], [0, 1, 0]],  # front (this matches the default plane)
                [[0, 0, -1], [0, 1, 0]],  # back
            ],
            dtype=np.float32,
        )

        plane_csys = np.array(
            [
                *cube_normal_up[4],
                np.cross(*cube_normal_up[4]),
            ]
        )

        planes = []
        for normal, up in cube_normal_up:
            plane_idx = np.flatnonzero(normal == 0)
            (
                plane_positions,
                plane_normals,
                plane_texcoords,
                plane_index,
            ) = generate_plane(*cube_dim[plane_idx], *cube_seg[plane_idx])

            sign_idx = np.flatnonzero(normal != 0)[0]
            matrix = np.identity(4, dtype=np.float32)
            matrix[:-1, -1] = (cube_dim[sign_idx] / 2) * normal
            matrix[:-1, :-1] = np.dot(
                np.array(
                    [
                        normal,
                        up,
                        np.cross(normal, up),
                    ]
                ).T,
                plane_csys,
            )

            plane_positions = transform(plane_positions, matrix)
            plane_normals = transform(plane_normals, matrix, directions=True)

            planes.append(
                (plane_positions, plane_normals, plane_texcoords, plane_index)
            )

        positions, normals, texcoords, index = merge(planes)
        self.positions = Buffer(positions)
        self.normals = Buffer(normals)
        self.texcoords = Buffer(texcoords)
        self.index = Buffer(index)
