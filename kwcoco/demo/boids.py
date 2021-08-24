import kwarray
import numpy as np
import ubelt as ub
from scipy.spatial.distance import pdist


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
        >>> ax = plt.gca(projection='3d')
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
                plt.arrow(tip[0], tip[1], s * acc[1], s * acc[1], color='purple')

                plt.arrow(tip[0], tip[1], s * spsteer[0], s * spsteer[1], color='dodgerblue')
                plt.arrow(tip[0], tip[1], s * cmsteer[0], s * cmsteer[1], color='orange')
                plt.arrow(tip[0], tip[1], s * alsteer[0], s * alsteer[1], color='pink')
                plt.arrow(tip[0], tip[1], s * avsteer[0], s * avsteer[1], color='black')

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            xdev.InteractiveIter.draw()
        rx = 0
    """

    def __init__(self, num, dims=2, rng=None, **kwargs):
        self.rng = kwarray.ensure_rng(rng)
        self.num = num
        self.dims = dims

        # self.config = {
        #     'perception_thresh': 0.08,
        #     'max_speed': 0.005,
        #     'max_force': 0.0003,
        # }
        self.config = {
            'perception_thresh': 0.2,
            'max_speed': 0.01,
            'max_force': 0.001,
            'damping': 0.99,
        }
        self.config.update(ub.dict_isect(kwargs, self.config))

        self.pos = None
        self.vel = None
        self.acc = None

    def __nice__(self):
        return '{}'.format(self.num)

    def initialize(self):
        # Generate random starting positions, velocities, and accelerations
        self.pos = self.rng.rand(self.num, self.dims)
        self.vel = self.rng.randn(self.num, self.dims) * self.config['max_speed']
        self.acc = self.rng.randn(self.num, self.dims) * self.config['max_force']
        return self

    def update_neighbors(self):
        # TODO: this should be done with a fast spatial index, but
        # unfortunately I don't see any existing implementations that make it
        # easy to support moving points.
        utriu_dists = pdist(self.pos)
        utriu_flags = utriu_dists < self.config['perception_thresh']
        utriu_rx, utriu_cx = np.triu_indices(len(self.pos), k=1)

        utriu_neighb_rxs = utriu_rx[utriu_flags]
        utriu_neighb_cxs = utriu_cx[utriu_flags]

        neighb_rxs = np.r_[utriu_neighb_rxs, utriu_neighb_cxs]
        neighb_cxs = np.r_[utriu_neighb_cxs, utriu_neighb_rxs]

        group_rxs, groupxs = kwarray.group_indices(neighb_rxs)
        group_cxs = kwarray.apply_grouping(neighb_cxs, groupxs)

        rx_to_neighb_cxs = ub.dzip(group_rxs, group_cxs)

        # n = len(self.pos)
        # rx_to_neighb_utriu_idxs = {}
        # for rx, cxs in rx_to_neighb_cxs.items():
        #     rxs = np.full_like(cxs, fill_value=rx)
        #     multi_index = (rxs, cxs)
        #     utriu_idxs = triu_condense_multi_index(
        #         multi_index, dims=(n, n), symetric=True)
        #     rx_to_neighb_utriu_idxs[rx] = utriu_idxs

        # self.utriu_dists = utriu_dists
        self.rx_to_neighb_cxs = rx_to_neighb_cxs
        # self.rx_to_neighb_utriu_idxs = rx_to_neighb_utriu_idxs

        # Compute speed and direction of every boid
        self.speeds = np.linalg.norm(self.vel, axis=1)
        self.dirs = self.vel / self.speeds[:, None]

    def compute_forces(self):
        self.update_neighbors()
        max_speed = self.config['max_speed']

        # Randomly drop perception of neighbors
        # neighbors[self.rng.rand(*neighbors.shape) > 0.3] = 0
        num = len(self.pos)

        align_steering = np.zeros((num, 2))
        com_steering = np.zeros((num, 2))
        sep_steering = np.zeros((num, 2))

        for rx in self.rx_to_neighb_cxs.keys():
            cxs = self.rx_to_neighb_cxs[rx]

            # Alignment
            # Each boid wants its direction to agree with the
            # the average direction of its neighbors
            if 1:
                neigh_vel = self.vel[cxs]
                ave_vel = neigh_vel.mean(axis=0)
                ave_speed = np.linalg.norm(ave_vel)
                if ave_speed > 0:
                    ave_dir = (ave_vel / ave_speed) * max_speed
                else:
                    ave_dir = ave_vel
                align_steering_ = ave_dir - self.vel[rx]
                align_steering[rx] = align_steering_

            # Cohesion
            # Each boid wants to be in the center of its neighbors
            if 1:
                center_of_mass = self.pos[cxs].mean(axis=0)
                com_vec = center_of_mass - self.pos[rx]
                com_dist = np.linalg.norm(com_vec)
                if com_dist > 0:
                    com_dir = (com_vec / com_dist) * max_speed
                else:
                    com_dir = com_vec
                com_steering_ = com_dir - self.vel[rx]
                com_steering[rx] = com_steering_

            # Separation
            # Each boid does not want to be too close to its neighbors
            if 1:
                neigh_vec = self.pos[rx] - self.pos[cxs]

                # utriu_idxs = self.rx_to_neighb_utriu_idxs[rx]
                # neigh_dist1 = self.utriu_dists[utriu_idxs][:, None]
                # neigh_dist = neigh_dist1
                neigh_dist2 = np.linalg.norm(neigh_vec, axis=1, keepdims=1)
                neigh_dist = neigh_dist2
                assert neigh_dist.max() <= self.config['perception_thresh']

                flags = neigh_dist.ravel() > 0
                # neigh_dir = neigh_vec[flags]
                # neigh_dir = neigh_vec[flags] / neigh_dist[flags]
                neigh_dir = neigh_vec[flags] / (neigh_dist[flags] ** 2)

                ave_neigh_dir = neigh_dir.mean(axis=0)
                sep_steering_ = ave_neigh_dir - self.vel[rx]
                sep_steering[rx] = sep_steering_

        def dist_point_to_line(line_pts, pt):
            pt1, pt2 = line_pts
            x1, y1 = pt1
            x2, y2 = pt2
            x0, y0 = pt
            numer = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
            denom = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            dist = numer / denom
            return dist

        # align_steering[
        # com_steering = clamp_mag(com_steering, self.config['max_force'], axis=None)

        align_steering = clamp_mag(align_steering, 0.33 * self.config['max_force'], axis=None)
        com_steering = clamp_mag(com_steering, 0.33 * self.config['max_force'], axis=None)

        # Separation and obstical avoidance should override alignment and COM
        sep_steering = clamp_mag(sep_steering, 1.0 * self.config['max_force'], axis=None)

        # Add some small random movement
        rand_steering = clamp_mag(np.random.randn(*self.pos.shape), 0.08 * self.config['max_force'], axis=None)

        self.sep_steering = sep_steering
        self.com_steering = com_steering
        self.align_steering = align_steering
        self.rand_steering = rand_steering

        steering = sum([
            com_steering,
            align_steering,
            sep_steering,
            rand_steering,
        ])

        if 1:
            # # Edge avoidance
            # # Each boid does not want to hit an edge
            avoid_steering = np.zeros_like(self.pos)
            edge_thresh = self.config['perception_thresh'] * 1
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
                edge_dist = np.linalg.norm(edge_vec, axis=1, keepdims=1)
                flags = ((edge_dist < edge_thresh) & (edge_dist > 0)).ravel()
                rxs = np.where(flags)[0]

                if len(rxs):
                    avoid_vec = edge_vec[rxs]
                    avoid_dist = edge_dist[rxs]
                    avoid_vec = avoid_vec / (avoid_dist ** 3)
                    avoid_steering_ = avoid_vec - self.vel[rxs]
                    avoid_steering[rxs] += avoid_steering_

            avoid_steering = clamp_mag(avoid_steering, 1.0 * self.config['max_force'], axis=None)
            steering += avoid_steering
            self.avoid_steering = avoid_steering

        return steering

    def boundary_conditions(self):
        # Clamp positions
        lower_boundry_violators = (self.pos < 0)
        upper_boundry_violators = (self.pos > 1)
        if np.any(lower_boundry_violators):
            self.pos[lower_boundry_violators] = 0
            self.vel[lower_boundry_violators] *= -1.0  # bounce
            self.acc[lower_boundry_violators] *= -1.0

        if np.any(upper_boundry_violators):
            self.pos[upper_boundry_violators] = 1
            self.vel[upper_boundry_violators] *= -1.0
            self.acc[upper_boundry_violators] *= -1.0

    def step(self):
        """
        Update positions, velocities, and accelerations
        """
        self.boundary_conditions()
        self.acc += self.compute_forces()
        self.pos += self.vel
        self.vel += self.acc

        self.acc = clamp_mag(self.acc, self.config['max_force'], axis=1)
        self.vel = clamp_mag(self.vel, self.config['max_speed'], axis=1)

        # Dampen acceleration
        self.acc[:] *= max(1, min(0, (1 - self.config['damping'])))
        # self.acc[:] = 0

        self.boundary_conditions()
        return self.pos

    def paths(self, num_steps):
        positions = []
        for _ in range(num_steps):
            pos = self.step().copy()
            positions.append(pos)
        paths = np.concatenate([p[:, None] for p in positions], axis=1)
        return paths


def clamp_mag(vec, mag, axis=None):
    """
    vec = np.random.rand(10, 2)
    mag = 1.0
    axis = 1
    new_vec = clamp_mag(vec, mag, axis)
    np.linalg.norm(new_vec, axis=axis)
    """
    norm = np.linalg.norm(vec, axis=axis, keepdims=True)
    flags = norm > mag
    flags = np.squeeze(flags, axis)
    vec[flags] = (vec[flags] / norm[flags]) * mag
    return vec


def triu_condense_multi_index(multi_index, dims, symetric=False):
    r"""
    Like np.ravel_multi_index but returns positions in an upper triangular
    condensed square matrix

    Examples:
        multi_index (Tuple[ArrayLike]):
            indexes for each dimension into the square matrix

        dims (Tuple[int]):
            shape of each dimension in the square matrix (should all be the
            same)

        symetric (bool):
            if True, converts lower triangular indices to their upper
            triangular location. This may cause a copy to occur.

    References:
        https://stackoverflow.com/a/36867493/887074
        https://numpy.org/doc/stable/reference/generated/numpy.ravel_multi_index.html#numpy.ravel_multi_index

    Examples:
        >>> dims = (3, 3)
        >>> symetric = True
        >>> multi_index = (np.array([0, 0, 1]), np.array([1, 2, 2]))
        >>> condensed_idxs = triu_condense_multi_index(multi_index, dims, symetric=symetric)
        >>> assert condensed_idxs.tolist() == [0, 1, 2]

        >>> n = 7
        >>> symetric = True
        >>> multi_index = np.triu_indices(n=n, k=1)
        >>> condensed_idxs = triu_condense_multi_index(multi_index, [n] * 2, symetric=symetric)
        >>> assert condensed_idxs.tolist() == list(range(n * (n - 1) // 2))
        >>> from scipy.spatial.distance import pdist, squareform
        >>> square_mat = np.zeros((n, n))
        >>> conden_mat = squareform(square_mat)
        >>> conden_mat[condensed_idxs] = np.arange(len(condensed_idxs)) + 1
        >>> square_mat = squareform(conden_mat)
        >>> print('square_mat =\n{}'.format(ub.repr2(square_mat, nl=1)))

        >>> n = 7
        >>> symetric = True
        >>> multi_index = np.tril_indices(n=n, k=-1)
        >>> condensed_idxs = triu_condense_multi_index(multi_index, [n] * 2, symetric=symetric)
        >>> assert sorted(condensed_idxs.tolist()) == list(range(n * (n - 1) // 2))
        >>> from scipy.spatial.distance import pdist, squareform
        >>> square_mat = np.zeros((n, n))
        >>> conden_mat = squareform(square_mat, checks=False)
        >>> conden_mat[condensed_idxs] = np.arange(len(condensed_idxs)) + 1
        >>> square_mat = squareform(conden_mat)
        >>> print('square_mat =\n{}'.format(ub.repr2(square_mat, nl=1)))

    Ignore:
        >>> import xdev
        >>> n = 30
        >>> symetric = True
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
                'allowed in a condensed matrix')

        tril_flags = ~triu_flags

        if not symetric:
            raise ValueError(
                'multi_index cannot contain inputs from '
                'lower triangle unless symetric=True')
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


def _spatial_index_scratch():
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


def closest_point_on_line_segment(pts, e1, e2):
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
    # make point vector wrt orgin
    pv = pts - e1
    # Project pv onto de
    mag = np.linalg.norm(de, axis=1)
    de_norm = (de / mag)

    pt_on_line_ = pv.dot(de_norm.T) * de_norm

    # Check if normalized dot product is between 0 and 1
    # Determines if pt is between 0,0 and de
    t = (de.dot(pt_on_line_.T) / (mag ** 2))[0]

    # t is an interpolation factor indicating how far past the line segment we
    # are. We are on the line segment if it is in the range 0 to 1.
    oob_left  = t < 0
    oob_right = t > 1

    # Compute the point on the extended line defined by the line segment.
    pt_on_seg = pt_on_line_ + e1
    # Clamp to the endpoints if out of bounds
    pt_on_seg[oob_left] = e1
    pt_on_seg[oob_right] = e2
    return pt_on_seg


def _pygame_render_boids():
    """
    Fast and responsive BOID rendering. This is an easter egg.

    Requirements:
        pip install pygame

    CommandLine:
        python -m kwcoco.demo.boids
        pip install pygame kwcoco -U && python -m kwcoco.demo.boids
    """
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

    kw = {
        'perception_thresh': 0.20,
        'max_speed': 0.005,
        'max_force': 0.0003,
        'damping': 0.99,
    }

    flock = Boids(256, **kw).initialize()

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

            screen.fill((255, 255, 255))

            positions = flock.step()
            for pos in positions:
                x, y = pos * (w, h)
                r = 8
                pygame.draw.ellipse(screen, (255, 0, 0), (x, y, r, r))
                pygame.draw.ellipse(screen, (0, 0, 0), (x, y, r, r), strokeweight)

            pygame.display.flip()
    except pygame.error:
        print('exiting')


def _yeah_boid():
    _pygame_render_boids()


if __name__ == '__main__':
    _yeah_boid()
