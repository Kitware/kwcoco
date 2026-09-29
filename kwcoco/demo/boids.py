from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import kwarray
import numpy as np
import ubelt as ub
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from kwarray.util_random import RNGInput


class Boids(ub.NiceRepr):
    """
    Efficient numpy based backend for generating boid positions.

    BOID = bird-oid object

    References:
        https://www.youtube.com/watch?v=mhjuuHl6qHM
        https://medium.com/better-programming/boids-simulating-birds-flock-behavior-in-python-9fff99375118
        https://en.wikipedia.org/wiki/Boids

    Example:
        >>> from kwcoco.demo.boids import *  # NOQA
        >>> num_frames = 10
        >>> num_objects = 3
        >>> rng = None
        >>> self = Boids(num=num_objects, rng=rng).initialize()
        >>> paths = self.paths(num_frames)
        >>> #
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> plt = kwplot.autoplt()
        >>> from mpl_toolkits.mplot3d import Axes3D  # NOQA
        >>> ax = plt.axes(projection='3d')
        >>> ax.cla()
        >>> #
        >>> for path in paths:
        >>>     time = np.arange(len(path))
        >>>     ax.plot(time, path.T[0] * 1, path.T[1] * 1, ',-')
        >>> ax.set_xlim(0, num_frames)
        >>> ax.set_ylim(-.01, 1.01)
        >>> ax.set_zlim(-.01, 1.01)
        >>> ax.set_xlabel('time')
        >>> ax.set_ylabel('u-pos')
        >>> ax.set_zlabel('v-pos')
        >>> kwplot.show_if_requested()

        import xdev
        _ = xdev.profile_now(self.compute_forces)()
        _ = xdev.profile_now(self.update_neighbors)()

    Ignore:
        self = Boids(num=5, rng=0).initialize()
        self.pos

        fig = kwplot.figure(fnum=10, do_clf=True)
        ax = fig.gca()

        verts = np.array([[0, 0], [1, 0], [0.5, 2]])
        com = verts.mean(axis=0)
        verts = (verts - com) * 0.02

        import kwimage
        poly = kwimage.Polygon(exterior=verts)

        def rotate(poly, theta):
            sin_ = np.sin(theta)
            cos_ = np.cos(theta)
            rot_ = np.array(((cos_, -sin_),
                             (sin_,  cos_),))
            return poly.warp(rot_)

        for _ in xdev.InteractiveIter(list(range(10000))):
            self.step()
            ax.cla()
            import math
            for rx in range(len(self.pos)):
                x, y = self.pos[rx]
                dx, dy = (self.vel[rx] / np.linalg.norm(self.vel[rx], axis=0))

                theta = (np.arctan2(dy, dx) - math.tau / 4)
                boid_poly = rotate(poly, theta).translate(self.pos[rx])
                color = 'red' if rx == 0 else 'blue'
                boid_poly.draw(ax=ax, color=color)

                tip = boid_poly.data['exterior'].data[2]
                tx, ty = tip

                s = 100.0
                vel = self.vel[rx]
                acc = self.acc[rx]
                com = self.acc[rx]

                spsteer = self.sep_steering[rx]
                cmsteer = self.com_steering[rx]
                alsteer = self.align_steering[rx]
                avsteer = self.avoid_steering[rx]

                # plt.arrow(tip[0], tip[1], s * vel[0], s * vel[1], color='green')
                plt.arrow(tip[0], tip[1], s * acc[0], s * acc[1], color='purple')

                plt.arrow(tip[0], tip[1], s * spsteer[0], s * spsteer[1], color='dodgerblue')
                plt.arrow(tip[0], tip[1], s * cmsteer[0], s * cmsteer[1], color='orange')
                plt.arrow(tip[0], tip[1], s * alsteer[0], s * alsteer[1], color='pink')
                plt.arrow(tip[0], tip[1], s * avsteer[0], s * avsteer[1], color='black')

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            xdev.InteractiveIter.draw()
        rx = 0

    Example:
        >>> # Test determenism
        >>> from kwcoco.demo.boids import *  # NOQA
        >>> num_frames = 2
        >>> num_objects = 1
        >>> rng = 4532
        >>> self = Boids(num=num_objects, rng=rng).initialize()
        >>> #print(ub.hash_data(self.pos))
        >>> #print(ub.hash_data(self.vel))
        >>> #print(ub.hash_data(self.acc))
        >>> tocheck = []
        >>> for i in range(100):
        >>>     self = Boids(num=num_objects, rng=rng).initialize()
        >>>     self.step()
        >>>     self.step()
        >>>     self.step()
        >>>     tocheck.append(self.pos.copy())
        >>> assert ub.allsame(list(map(ub.hash_data, tocheck)))
    """

    def __init__(
        self, num: int, dims: int = 2, rng: RNGInput = None, **kwargs
    ) -> None:
        if dims != 2:
            raise NotImplementedError('Boids currently supports only dims=2')
        self.rng = kwarray.ensure_rng(rng, api='numpy')
        self.num = num
        self.dims = dims

        # self.config = {
        #     'perception_thresh': 0.08,
        #     'max_speed': 0.005,
        #     'max_force': 0.0003,
        # }
        self.config = {
            'perception_thresh': 0.05,
            'separation_thresh': 0.030,
            'max_speed': 0.003,
            'max_force': 0.00012,
            'damping': 0.99,
            'alignment_strength': 0.20,
            'cohesion_strength': 0.20,
            'separation_strength': 0.55,
            'random_strength': 0.12,
            'avoidance_strength': 0.75,
            'edge_thresh': 0.12,
        }
        self.config.update(ub.dict_isect(kwargs, self.config))

        # Populated by initialize().
        self.pos: np.ndarray
        self.vel: np.ndarray
        self.acc: np.ndarray
        self.neighbor_pairs: np.ndarray = np.empty((0, 2), dtype=np.int64)
        self.neighbor_vec: np.ndarray = np.empty((0, 2), dtype=float)
        self.neighbor_dist2: np.ndarray = np.empty(0, dtype=float)

    def __nice__(self) -> str:
        return '{}'.format(self.num)

    def initialize(self) -> Boids:
        # Generate random starting positions, velocities, and accelerations
        self.pos = self.rng.rand(self.num, self.dims)
        self.vel = self.rng.randn(self.num, self.dims) * self.config['max_speed']
        self.acc = self.rng.randn(self.num, self.dims) * self.config['max_force']
        return self

    def update_neighbors(self) -> np.ndarray:
        # Rebuild a spatial index each frame and return the sparse undirected
        # neighbor graph. cKDTree performs both index construction and radius
        # queries in compiled code, avoiding Python work per occupied grid cell.
        perception_thresh = self.config['perception_thresh']
        num = len(self.pos)

        if num > 1 and perception_thresh > 0:
            tree = cKDTree(
                self.pos, compact_nodes=False, balanced_tree=False
            )
            pairs = tree.query_pairs(
                perception_thresh, output_type='ndarray'
            )
            if len(pairs):
                # query_pairs includes pairs exactly on the radius. Preserve
                # the historical strict ``distance < perception_thresh`` rule
                # and retain the pair geometry for separation below.
                neigh_vec = self.pos[pairs[:, 0]] - self.pos[pairs[:, 1]]
                neigh_dist2 = np.sum(neigh_vec * neigh_vec, axis=1)
                flags = neigh_dist2 < perception_thresh ** 2
                if not np.all(flags):
                    pairs = pairs[flags]
                    neigh_vec = neigh_vec[flags]
                    neigh_dist2 = neigh_dist2[flags]
            else:
                neigh_vec = np.empty((0, 2), dtype=float)
                neigh_dist2 = np.empty(0, dtype=float)
        else:
            pairs = np.empty((0, 2), dtype=np.int64)
            neigh_vec = np.empty((0, 2), dtype=float)
            neigh_dist2 = np.empty(0, dtype=float)

        self.neighbor_pairs = pairs
        self.neighbor_vec = neigh_vec
        self.neighbor_dist2 = neigh_dist2
        return pairs

    def compute_forces(self) -> np.ndarray:
        pairs = self.update_neighbors()
        max_speed = self.config['max_speed']
        num = len(self.pos)

        align_steering = np.zeros((num, 2))
        com_steering = np.zeros((num, 2))
        sep_steering = np.zeros((num, 2))

        if len(pairs):
            rxs1 = pairs[:, 0]
            rxs2 = pairs[:, 1]

            # Each pair contributes one neighbor to each endpoint. Aggregate
            # over the sparse edge list instead of issuing many small numpy
            # reductions inside a Python loop over boids.
            neighbor_count = np.bincount(pairs.ravel(), minlength=num)
            has_neighbors = neighbor_count > 0

            # Alignment: average neighbor velocity.
            neigh_vel_sum = np.empty((num, 2), dtype=float)
            for dim in range(2):
                neigh_vel_sum[:, dim] = (
                    np.bincount(
                        rxs1, weights=self.vel[rxs2, dim], minlength=num
                    )
                    + np.bincount(
                        rxs2, weights=self.vel[rxs1, dim], minlength=num
                    )
                )
            ave_vel = np.zeros((num, 2), dtype=float)
            ave_vel[has_neighbors] = (
                neigh_vel_sum[has_neighbors]
                / neighbor_count[has_neighbors, None]
            )
            ave_speed = np.linalg.norm(ave_vel, axis=1)
            moving = ave_speed > 0
            ave_vel[moving] *= (max_speed / ave_speed[moving])[:, None]
            align_steering[has_neighbors] = (
                ave_vel[has_neighbors] - self.vel[has_neighbors]
            )

            # Cohesion: average neighbor position.
            neigh_pos_sum = np.empty((num, 2), dtype=float)
            for dim in range(2):
                neigh_pos_sum[:, dim] = (
                    np.bincount(
                        rxs1, weights=self.pos[rxs2, dim], minlength=num
                    )
                    + np.bincount(
                        rxs2, weights=self.pos[rxs1, dim], minlength=num
                    )
                )
            center_of_mass = np.zeros((num, 2), dtype=float)
            center_of_mass[has_neighbors] = (
                neigh_pos_sum[has_neighbors]
                / neighbor_count[has_neighbors, None]
            )
            com_vec = center_of_mass - self.pos
            com_dist = np.linalg.norm(com_vec, axis=1)
            com_dir = com_vec.copy()
            moving = has_neighbors & (com_dist > 0)
            com_dir[moving] *= (max_speed / com_dist[moving])[:, None]
            com_steering[has_neighbors] = (
                com_dir[has_neighbors] - self.vel[has_neighbors]
            )

            # Separation: only repel boids that are within the smaller
            # separation radius. A pair contributes equal and opposite
            # repulsion to its endpoints. Coincident boids are excluded.
            neigh_vec = self.neighbor_vec
            neigh_dist2 = self.neighbor_dist2
            separation_thresh = min(
                self.config['separation_thresh'],
                self.config['perception_thresh'],
            )
            valid = (
                (neigh_dist2 > 0)
                & (neigh_dist2 < separation_thresh ** 2)
            )
            if np.any(valid):
                valid_rxs1 = rxs1[valid]
                valid_rxs2 = rxs2[valid]
                repel = neigh_vec[valid] / neigh_dist2[valid, None]
                sep_count = (
                    np.bincount(valid_rxs1, minlength=num)
                    + np.bincount(valid_rxs2, minlength=num)
                )
                sep_sum = np.empty((num, 2), dtype=float)
                for dim in range(2):
                    sep_sum[:, dim] = (
                        np.bincount(
                            valid_rxs1, weights=repel[:, dim], minlength=num
                        )
                        - np.bincount(
                            valid_rxs2, weights=repel[:, dim], minlength=num
                        )
                    )
                has_sep_neighbors = sep_count > 0
                ave_neigh_dir = np.zeros((num, 2), dtype=float)
                ave_neigh_dir[has_sep_neighbors] = (
                    sep_sum[has_sep_neighbors]
                    / sep_count[has_sep_neighbors, None]
                )
                sep_steering[has_sep_neighbors] = (
                    ave_neigh_dir[has_sep_neighbors]
                    - self.vel[has_sep_neighbors]
                )

        align_steering = clamp_mag(
            align_steering,
            self.config['alignment_strength'] * self.config['max_force'],
            axis=1,
        )
        com_steering = clamp_mag(
            com_steering,
            self.config['cohesion_strength'] * self.config['max_force'],
            axis=1,
        )

        # Separation and obstical avoidance should override alignment and COM
        sep_steering = clamp_mag(
            sep_steering,
            self.config['separation_strength'] * self.config['max_force'],
            axis=1,
        )

        # Add some small random movement
        rand_steering = clamp_mag(
            self.rng.randn(*self.pos.shape),
            self.config['random_strength'] * self.config['max_force'],
            axis=1,
        )

        self.sep_steering = sep_steering
        self.com_steering = com_steering
        self.align_steering = align_steering
        self.rand_steering = rand_steering

        steering = sum(
            [
                com_steering,
                align_steering,
                sep_steering,
                rand_steering,
            ]
        )

        if 1:
            # # Edge avoidance
            # # Each boid does not want to hit an edge
            avoid_steering = np.zeros_like(self.pos)
            edge_thresh = self.config['edge_thresh']
            edges = {
                'bot': np.array([(0, 0), (1, 0)]),
                'left': np.array([(0, 0), (0, 1)]),
                'top': np.array([(0, 1), (1, 1)]),
                'right': np.array([(1, 0), (1, 1)]),
            }
            for edge_name, edge in edges.items():
                e1, e2 = edge
                edge_pt = closest_point_on_line_segment(self.pos, e1, e2)
                edge_vec = self.pos - edge_pt
                edge_dist = np.linalg.norm(edge_vec, axis=1, keepdims=True)
                flags = ((edge_dist < edge_thresh) & (edge_dist > 0)).ravel()
                rxs = np.where(flags)[0]

                if len(rxs):
                    avoid_vec = edge_vec[rxs]
                    avoid_dist = edge_dist[rxs]
                    avoid_vec = avoid_vec / (avoid_dist**3)
                    avoid_steering_ = avoid_vec - self.vel[rxs]
                    avoid_steering[rxs] += avoid_steering_

            avoid_steering = clamp_mag(
                avoid_steering,
                self.config['avoidance_strength'] * self.config['max_force'],
                axis=1,
            )
            steering += avoid_steering
            self.avoid_steering = avoid_steering

        return steering

    def boundary_conditions(self) -> None:
        # Clamp positions
        lower_boundry_violators = self.pos < 0
        upper_boundry_violators = self.pos > 1
        if np.any(lower_boundry_violators):
            self.pos[lower_boundry_violators] = 0
            self.vel[lower_boundry_violators] *= -1.0  # bounce
            self.acc[lower_boundry_violators] *= -1.0

        if np.any(upper_boundry_violators):
            self.pos[upper_boundry_violators] = 1
            self.vel[upper_boundry_violators] *= -1.0
            self.acc[upper_boundry_violators] *= -1.0

    def step(self) -> np.ndarray:
        """
        Update positions, velocities, and accelerations
        """
        self.boundary_conditions()
        force = self.compute_forces()
        self.acc += force
        self.pos += self.vel
        self.vel += self.acc

        self.acc = clamp_mag(self.acc, self.config['max_force'], axis=1)
        self.vel = clamp_mag(self.vel, self.config['max_speed'], axis=1)

        # Dampen acceleration
        self.acc[:] *= min(1, max(0, (1 - self.config['damping'])))
        # self.acc[:] = 0

        self.boundary_conditions()
        return self.pos

    def paths(self, num_steps: int) -> np.ndarray:
        positions = []
        for _ in range(num_steps):
            pos = self.step().copy()
            positions.append(pos)
        paths = np.concatenate([p[:, None] for p in positions], axis=1)
        return paths


def clamp_mag(
    vec: np.ndarray, mag: float, axis: Optional[int] = None
) -> np.ndarray:
    """
    vec = np.random.rand(10, 2)
    mag = 1.0
    axis = 1
    new_vec = clamp_mag(vec, mag, axis)
    np.linalg.norm(new_vec, axis=axis)
    """
    norm = np.asarray(np.linalg.norm(vec, axis=axis, keepdims=True))
    scale = np.minimum(1.0, mag / np.maximum(norm, 1e-12))
    vec[...] = vec * scale
    return vec


def triu_condense_multi_index(
    multi_index: Tuple[np.ndarray, np.ndarray],
    dims: Sequence[int],
    symmetric: bool = False,
) -> np.ndarray:
    r"""
    Like np.ravel_multi_index but returns positions in an upper triangular
    condensed square matrix

    Examples:
        multi_index (Tuple[ArrayLike]):
            indexes for each dimension into the square matrix

        dims (Tuple[int]):
            shape of each dimension in the square matrix (should all be the
            same)

        symmetric (bool):
            if True, converts lower triangular indices to their upper
            triangular location. This may cause a copy to occur.

    References:
        https://stackoverflow.com/a/36867493/887074
        https://numpy.org/doc/stable/reference/generated/numpy.ravel_multi_index.html#numpy.ravel_multi_index

    Examples:
        >>> dims = (3, 3)
        >>> symmetric = True
        >>> multi_index = (np.array([0, 0, 1]), np.array([1, 2, 2]))
        >>> condensed_idxs = triu_condense_multi_index(multi_index, dims, symmetric=symmetric)
        >>> assert condensed_idxs.tolist() == [0, 1, 2]

        >>> n = 7
        >>> symmetric = True
        >>> multi_index = np.triu_indices(n=n, k=1)
        >>> condensed_idxs = triu_condense_multi_index(multi_index, [n] * 2, symmetric=symmetric)
        >>> assert condensed_idxs.tolist() == list(range(n * (n - 1) // 2))
        >>> from scipy.spatial.distance import pdist, squareform
        >>> square_mat = np.zeros((n, n))
        >>> conden_mat = squareform(square_mat)
        >>> conden_mat[condensed_idxs] = np.arange(len(condensed_idxs)) + 1
        >>> square_mat = squareform(conden_mat)
        >>> print('square_mat =\n{}'.format(ub.urepr(square_mat, nl=1)))

        >>> n = 7
        >>> symmetric = True
        >>> multi_index = np.tril_indices(n=n, k=-1)
        >>> condensed_idxs = triu_condense_multi_index(multi_index, [n] * 2, symmetric=symmetric)
        >>> assert sorted(condensed_idxs.tolist()) == list(range(n * (n - 1) // 2))
        >>> from scipy.spatial.distance import pdist, squareform
        >>> square_mat = np.zeros((n, n))
        >>> conden_mat = squareform(square_mat, checks=False)
        >>> conden_mat[condensed_idxs] = np.arange(len(condensed_idxs)) + 1
        >>> square_mat = squareform(conden_mat)
        >>> print('square_mat =\n{}'.format(ub.urepr(square_mat, nl=1)))

    Ignore:
        >>> import xdev
        >>> n = 30
        >>> symmetric = True
        >>> multi_index = np.triu_indices(n=n, k=1)
        >>> condensed_idxs = xdev.profile_now(triu_condense_multi_index)(multi_index, [n] * 2)

        # Numba helps here when ub.allsame is gone
        from numba import jit
        triu_condense_multi_index2 = jit(nopython=True)(triu_condense_multi_index)
        triu_condense_multi_index2 = jit()(triu_condense_multi_index)
        triu_condense_multi_index2(multi_index, [n] * 2)
        %timeit triu_condense_multi_index(multi_index, [n] * 2)
        %timeit triu_condense_multi_index2(multi_index, [n] * 2)
    """
    if len(dims) != 2:
        raise NotImplementedError('only 2d matrices for now')
    if not ub.allsame(dims):
        raise NotImplementedError('only square matrices for now')

    rxs, cxs = multi_index

    triu_flags = rxs < cxs
    if not np.all(triu_flags):
        if np.any(rxs == cxs):
            raise NotImplementedError(
                'multi_index contains diagonal elements, which are not '
                'allowed in a condensed matrix'
            )

        tril_flags = ~triu_flags

        if not symmetric:
            raise ValueError(
                'multi_index cannot contain inputs from '
                'lower triangle unless symmetric=True'
            )
        else:
            rxs = rxs.copy()
            cxs = cxs.copy()
            _tmp_rxs = rxs[tril_flags]
            rxs[tril_flags] = cxs[tril_flags]
            cxs[tril_flags] = _tmp_rxs

    n = dims[0]
    # Let i = rxs
    # Let j = cxs
    # with i*n + j you go to the position in the square-formed matrix;
    # with - i*(i+1)/2 you remove lower triangle (including diagonal) in all lines before i;
    # with - i you remove positions in line i before the diagonal;
    # with - 1 you remove positions in line i on the diagonal.
    """
    import sympy
    rxs, n, cxs = sympy.symbols(['rxs', 'n', 'cxs'])
    condensed_indices = (n * rxs + cxs) - (rxs * (rxs + 1) // 2) - rxs - 1
    sympy.simplify(condensed_indices)
    %timeit cxs + (n - 1) * rxs - rxs*(rxs + 1)//2 - 1
    %timeit (n * rxs + cxs) - (rxs * (rxs + 1) // 2) - rxs - 1
    """
    condensed_indices = cxs + (n - 1) * rxs - (rxs * (rxs + 1) // 2) - 1
    # condensed_indices = (n * rxs + cxs) - (rxs * (rxs + 1) // 2) - rxs - 1
    return condensed_indices


def _spatial_index_scratch() -> None:
    """
    Ignore:
        !pip install git+git://github.com/carsonfarmer/fastpair.git

        from fastpair import FastPair
        fp = FastPair()
        fp.build(list(map(tuple, self.pos.tolist())))  #

        !pip install pyqtree
        from pyqtree import Index
        spindex = Index(bbox=(0, 0, 1., 1.))

        # this example assumes you have a list of items with bbox attribute
        for item in items:
            spindex.insert(item, item.bbox)

        https://stackoverflow.com/questions/41946007/efficient-and-well-explained-implementation-of-a-quadtree-for-2d-collision-det

        !pip install grispy
        import grispy

        index = grispy.GriSPy(data=self.pos)
        index.bubble_neighbors(self.pos, distance_upper_bound=0.1)
    """


def closest_point_on_line_segment(
    pts: np.ndarray, e1: np.ndarray, e2: np.ndarray
) -> np.ndarray:
    """
    Finds the closet point from p on line segment (e1, e2)

    Args:
        pts (ndarray): xy points [Nx2]
        e1 (ndarray): the first xy endpoint of the segment
        e2 (ndarray): the second xy endpoint of the segment

    Returns:
        ndarray: pt_on_seg - the closest xy point on (e1, e2) from ptp

    References:
        http://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
        http://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment

    Example:
        >>> # ENABLE_DOCTEST
        >>> from kwcoco.demo.boids import *  # NOQA
        >>> verts = np.array([[ 21.83012702,  13.16987298],
        >>>                   [ 16.83012702,  21.83012702],
        >>>                   [  8.16987298,  16.83012702],
        >>>                   [ 13.16987298,   8.16987298],
        >>>                   [ 21.83012702,  13.16987298]])
        >>> rng = np.random.RandomState(0)
        >>> pts = rng.rand(64, 2) * 20 + 5
        >>> e1, e2 = verts[0:2]
        >>> closest_point_on_line_segment(pts, e1, e2)

    Ignore:
        from numba import jit
        closest_point_on_line_segment2 = jit(closest_point_on_line_segment)
        closest_point_on_line_segment2(pts, e1, e2)
        %timeit closest_point_on_line_segment(pts, e1, e2)
        %timeit closest_point_on_line_segment2(pts, e1, e2)
    """
    # shift e1 to origin
    de = (e2 - e1)[None, :]
    # make point vector wrt origin
    pv = pts - e1
    # Project pv onto de
    mag = np.linalg.norm(de, axis=1)
    de_norm = de / mag

    pt_on_line_ = pv.dot(de_norm.T) * de_norm

    # Check if normalized dot product is between 0 and 1
    # Determines if pt is between 0,0 and de
    t = (de.dot(pt_on_line_.T) / (mag**2))[0]

    # t is an interpolation factor indicating how far past the line segment we
    # are. We are on the line segment if it is in the range 0 to 1.
    oob_left = t < 0
    oob_right = t > 1

    # Compute the point on the extended line defined by the line segment.
    pt_on_seg = pt_on_line_ + e1
    # Clamp to the endpoints if out of bounds
    pt_on_seg[oob_left] = e1
    pt_on_seg[oob_right] = e2
    return pt_on_seg


def _pygame_render_boids() -> None:
    """
    Fast and responsive BOID rendering. This is an easter egg.

    Requirements:
        pip install pygame

    CommandLine:
        python -m kwcoco.demo.boids
        pip install pygame kwcoco -U && python -m kwcoco.demo.boids
    """
    ASK_TO_INSTALL = 1
    if ASK_TO_INSTALL:
        if ub.modname_to_modpath('pygame') is None:
            from rich.prompt import Confirm

            ans = Confirm.ask('Pygame was not detected. Do you want to install it?')
            if ans:
                import subprocess
                import sys

                subprocess.check_call(
                    [sys.executable, '-m', 'pip', 'install', 'pygame']
                )

    try:
        import pygame
    except ImportError:
        raise Exception('Please pip install pygame before using this function')
    pygame.init()
    display_info = pygame.display.Info()

    w = display_info.current_w // 2
    h = display_info.current_h // 2

    screen = pygame.display.set_mode([w, h], flags=pygame.RESIZABLE)
    pygame.display.set_caption('YEAH BOID!')
    strokeweight = 2
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 22)

    param_specs = [
        ('perception_thresh', 0.01, 0.01, 1.0, '{:.2f}'),
        ('separation_thresh', 0.005, 0.0, 1.0, '{:.3f}'),
        ('max_speed', 0.001, 0.001, 0.05, '{:.3f}'),
        ('max_force', 0.00005, 0.00005, 0.01, '{:.5f}'),
        ('damping', 0.01, 0.0, 1.0, '{:.2f}'),
        ('alignment_strength', 0.05, 0.0, 3.0, '{:.2f}'),
        ('cohesion_strength', 0.05, 0.0, 3.0, '{:.2f}'),
        ('separation_strength', 0.05, 0.0, 3.0, '{:.2f}'),
        ('random_strength', 0.01, 0.0, 1.0, '{:.2f}'),
        ('avoidance_strength', 0.05, 0.0, 3.0, '{:.2f}'),
        ('edge_thresh', 0.01, 0.0, 1.0, '{:.2f}'),
    ]

    # Boids.__init__ is the single source of truth for demo defaults.
    flock = Boids(256)
    initial_rng_state = flock.rng.get_state()
    flock.initialize()

    def reset_flock() -> None:
        # Keep the parameters currently selected in the UI, but replay the
        # initial random draws so their effect can be compared from the same
        # starting conditions.
        flock.rng.set_state(initial_rng_state)
        flock.initialize()
        flock.neighbor_pairs = np.empty((0, 2), dtype=np.int64)
        flock.neighbor_vec = np.empty((0, 2), dtype=float)
        flock.neighbor_dist2 = np.empty(0, dtype=float)

    control_buttons = []
    controls_expanded = True
    toggle_rect = None
    reset_rect = None
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if reset_rect is not None and reset_rect.collidepoint(event.pos):
                    reset_flock()
                elif toggle_rect is not None and toggle_rect.collidepoint(event.pos):
                    controls_expanded = not controls_expanded
                elif controls_expanded:
                    for key, delta, lower, upper, rect in control_buttons:
                        if rect.collidepoint(event.pos):
                            value = flock.config[key] + delta
                            flock.config[key] = min(upper, max(lower, value))
                            if key == 'separation_thresh':
                                flock.config[key] = min(
                                    flock.config[key],
                                    flock.config['perception_thresh'],
                                )
                            elif key == 'perception_thresh':
                                flock.config['separation_thresh'] = min(
                                    flock.config['separation_thresh'],
                                    flock.config[key],
                                )
                            break

        screen.fill((255, 255, 255))
        w, h = screen.get_size()

        positions = flock.step()
        for pos in positions:
            x, y = pos * (w, h)
            r = 8
            pygame.draw.ellipse(screen, (255, 0, 0), (x, y, r, r))
            pygame.draw.ellipse(screen, (0, 0, 0), (x, y, r, r), strokeweight)

        # Draw a small live-control HUD in the top-left corner.
        margin = 8
        row_h = 24
        panel_w = 300
        num_panel_rows = (len(param_specs) + 2) if controls_expanded else 2
        panel_h = row_h * num_panel_rows + margin * 2
        panel_rect = pygame.Rect(margin, margin, panel_w, panel_h)
        pygame.draw.rect(screen, (255, 255, 255), panel_rect)
        pygame.draw.rect(screen, (0, 0, 0), panel_rect, 1)

        reset_rect = pygame.Rect(
            margin + panel_w - 100, margin + 4, 64, 20
        )
        pygame.draw.rect(screen, (235, 235, 235), reset_rect)
        pygame.draw.rect(screen, (0, 0, 0), reset_rect, 1)
        reset_surface = font.render('reset', True, (0, 0, 0))
        reset_surface_rect = reset_surface.get_rect(center=reset_rect.center)
        screen.blit(reset_surface, reset_surface_rect)

        toggle_rect = pygame.Rect(
            margin + panel_w - 30, margin + 4, 22, 20
        )
        pygame.draw.rect(screen, (235, 235, 235), toggle_rect)
        pygame.draw.rect(screen, (0, 0, 0), toggle_rect, 1)
        toggle_symbol = '-' if controls_expanded else '+'
        toggle_surface = font.render(toggle_symbol, True, (0, 0, 0))
        toggle_surface_rect = toggle_surface.get_rect(center=toggle_rect.center)
        screen.blit(toggle_surface, toggle_surface_rect)

        fps_text = font.render('FPS: {:5.1f}'.format(clock.get_fps()), True, (0, 0, 0))
        screen.blit(fps_text, (margin * 2, margin * 2))
        num_pairs = len(flock.neighbor_pairs)
        mean_neighbors = (2 * num_pairs / flock.num) if flock.num else 0
        neighbor_text = font.render(
            'pairs: {}  mean neighbors: {:.1f}'.format(num_pairs, mean_neighbors),
            True,
            (0, 0, 0),
        )
        screen.blit(neighbor_text, (margin * 2, margin * 2 + row_h))

        control_buttons = []
        if controls_expanded:
            label_x = margin * 2
            minus_x = margin + panel_w - 58
            plus_x = margin + panel_w - 30
            button_w = 24
            button_h = 20
            for row, (key, step, lower, upper, fmt) in enumerate(param_specs, start=2):
                y = margin * 2 + row * row_h
                value_text = '{}: {}'.format(key, fmt.format(flock.config[key]))
                text_surface = font.render(value_text, True, (0, 0, 0))
                screen.blit(text_surface, (label_x, y + 2))

                for symbol, delta, x in [('-', -step, minus_x), ('+', step, plus_x)]:
                    rect = pygame.Rect(x, y, button_w, button_h)
                    pygame.draw.rect(screen, (235, 235, 235), rect)
                    pygame.draw.rect(screen, (0, 0, 0), rect, 1)
                    symbol_surface = font.render(symbol, True, (0, 0, 0))
                    symbol_rect = symbol_surface.get_rect(center=rect.center)
                    screen.blit(symbol_surface, symbol_rect)
                    control_buttons.append((key, delta, lower, upper, rect))

        pygame.display.flip()
        clock.tick()

    pygame.quit()


def _yeah_boid() -> None:
    _pygame_render_boids()


if __name__ == '__main__':
    _yeah_boid()
