"""Forward kinematics utilities for trajectory optimization.

This module provides backend-agnostic FK computation functions
shared across different solvers (scipy, jaxls, gradient_descent).
"""

from skrobot.backend import rodrigues_rotation
from skrobot.kinematics.differentiable import pose_error_se3_log as pose_error_log
from skrobot.kinematics.differentiable import rotation_error_so3_log as rotation_error_log


def build_fk_functions(fk_data, backend):
    """Build forward kinematics helper functions.

    Parameters
    ----------
    fk_data : dict
        FK parameters including:
        - link_translations: (n_joints, 3) link translations
        - link_rotations: (n_joints, 3, 3) link rotations
        - joint_axes: (n_joints, 3) joint axes
        - base_position: (3,) base position
        - base_rotation: (3, 3) base rotation
        - n_joints: int
        - collision_link_to_chain_idx: (n_coll_links,) indices
        - collision_link_offsets_pos: (n_coll_links, 3)
        - collision_link_offsets_rot: (n_coll_links, 3, 3)
        - sphere_centers_local: (n_spheres, 3) local positions
        - collision_link_indices: (n_spheres,) link index per sphere
    backend : module
        Array module (numpy, jax.numpy, or skrobot backend).

    Returns
    -------
    tuple
        (get_link_transforms, get_sphere_positions) functions.
    """
    xp = backend

    link_trans = fk_data['link_translations']
    link_rots = fk_data['link_rotations']
    joint_axes = fk_data['joint_axes']
    base_pos = fk_data['base_position']
    base_rot = fk_data['base_rotation']
    n_joints = fk_data['n_joints']
    ref_angles = fk_data.get('ref_angles')

    coll_link_idx = fk_data.get('collision_link_to_chain_idx')
    coll_offsets_pos = fk_data.get('collision_link_offsets_pos')
    coll_offsets_rot = fk_data.get('collision_link_offsets_rot')
    sphere_centers = fk_data.get('sphere_centers_local')
    sphere_link_indices = fk_data.get('collision_link_indices')

    def get_link_transforms(angles):
        """Compute link transforms for given joint angles.

        Parameters
        ----------
        angles : array
            Joint angles (n_joints,).

        Returns
        -------
        tuple
            (positions, rotations) arrays of shape
            (n_joints, 3) and (n_joints, 3, 3).
        """
        positions = []
        rotations = []
        current_pos = base_pos
        current_rot = base_rot

        for i in range(n_joints):
            current_pos = current_pos + current_rot @ link_trans[i]
            current_rot = current_rot @ link_rots[i]
            # Subtract ref_angles because link_rots already includes
            # the rotation at the reference configuration
            delta = angles[i]
            if ref_angles is not None:
                delta = delta - ref_angles[i]
            joint_rot = rodrigues_rotation(xp, joint_axes[i], delta)
            current_rot = current_rot @ joint_rot
            positions.append(current_pos)
            rotations.append(current_rot)

        return xp.stack(positions), xp.stack(rotations)

    ee_offset_pos = fk_data.get('ee_offset_position')
    ee_offset_rot = fk_data.get('ee_offset_rotation')

    def get_ee_position(angles):
        """Compute end-effector position for given joint angles.

        Parameters
        ----------
        angles : array
            Joint angles (n_joints,).

        Returns
        -------
        array
            End-effector position in world frame (3,).
        """
        pos, _ = get_ee_pose(angles)
        return pos

    def get_ee_pose(angles):
        """Compute end-effector position and rotation for given joint angles.

        Parameters
        ----------
        angles : array
            Joint angles (n_joints,).

        Returns
        -------
        position : array
            End-effector position in world frame (3,).
        rotation : array
            End-effector rotation matrix in world frame (3, 3).
        """
        positions, rotations = get_link_transforms(angles)
        last_pos = positions[-1]
        last_rot = rotations[-1]
        if ee_offset_pos is not None:
            ee_pos = last_pos + last_rot @ ee_offset_pos
        else:
            ee_pos = last_pos
        if ee_offset_rot is not None:
            ee_rot = last_rot @ ee_offset_rot
        else:
            ee_rot = last_rot
        return ee_pos, ee_rot

    def get_sphere_positions(angles):
        """Compute collision sphere positions for given joint angles.

        Spheres approximate collision geometries (spheres or capsules)
        attached to robot links.

        Parameters
        ----------
        angles : array
            Joint angles (n_joints,).

        Returns
        -------
        array
            Sphere positions in world frame (n_spheres, 3).
        """
        if sphere_centers is None:
            return xp.zeros((0, 3))

        link_positions, link_rotations = get_link_transforms(angles)

        chain_idx = coll_link_idx[sphere_link_indices]
        sphere_link_pos = link_positions[chain_idx]
        sphere_link_rot = link_rotations[chain_idx]

        offsets_pos = coll_offsets_pos[sphere_link_indices]
        offsets_rot = coll_offsets_rot[sphere_link_indices]

        local = xp.einsum('ijk,ik->ij', offsets_rot, sphere_centers) \
            + offsets_pos
        world = sphere_link_pos \
            + xp.einsum('ijk,ik->ij', sphere_link_rot, local)
        return world

    return get_link_transforms, get_sphere_positions, get_ee_position, get_ee_pose


def compute_sphere_obstacle_distances(sphere_positions, sphere_radii,
                                       obstacle_centers, obstacle_radii,
                                       backend):
    """Compute signed distances between collision spheres and obstacles.

    Parameters
    ----------
    sphere_positions : array
        Collision sphere positions (n_spheres, 3).
    sphere_radii : array
        Collision sphere radii (n_spheres,).
    obstacle_centers : array
        Obstacle centers (n_obstacles, 3).
    obstacle_radii : array
        Obstacle radii (n_obstacles,).
    backend : module
        Array module.

    Returns
    -------
    array
        Signed distances (n_spheres, n_obstacles).
        Positive = separated, negative = penetrating.
    """
    xp = backend
    # sphere_positions: (n_spheres, 3)
    # obstacle_centers: (n_obstacles, 3)
    diff = sphere_positions[:, None, :] - obstacle_centers[None, :, :]
    dists = xp.sqrt(xp.sum(diff ** 2, axis=-1) + 1e-10)
    signed_dists = dists - sphere_radii[:, None] - obstacle_radii[None, :]
    return signed_dists


def compute_cylinder_obstacle_distances(sphere_positions, sphere_radii,
                                         obstacle_centers,
                                         obstacle_rotations,
                                         obstacle_radii,
                                         obstacle_half_heights,
                                         backend):
    """Compute signed distances between collision spheres and cylinders.

    Each obstacle is a finite (flat-capped) cylinder, matching the
    ``skrobot.model.primitives.Cylinder`` objects
    ``aero_demo.solve_palm_ik.human_body_obstacles`` builds around the
    skeleton (as opposed to approximating each one with a handful of
    spheres swept along its axis, which leaves gaps between spheres and
    under-estimates penetration -- see ``aero_demo/scripts/plan_
    handshake_motion.py`` for the caller that builds ``obstacle_*``
    directly from that same capsule geometry).

    Unlike :func:`aero_demo`'s (unused, non-differentiable)
    ``point_to_cylinder_distance``, this returns a true *signed*
    distance that stays negative (and keeps a non-zero gradient) for a
    sphere centre anywhere inside the cylinder, not just at its
    surface -- required so that gradient-based optimisers still get a
    push-out direction when a waypoint starts out penetrating.

    Parameters
    ----------
    sphere_positions : array
        Collision sphere positions, world frame (n_spheres, 3).
    sphere_radii : array
        Collision sphere radii (n_spheres,).
    obstacle_centers : array
        Cylinder centre (mid-axis point), world frame (n_obstacles, 3).
    obstacle_rotations : array
        Cylinder orientation, local->world rotation matrices with the
        local +Z axis along the cylinder axis (n_obstacles, 3, 3).
    obstacle_radii : array
        Cylinder radii (n_obstacles,).
    obstacle_half_heights : array
        Half of each cylinder's height along its axis (n_obstacles,).
    backend : module
        Array module.

    Returns
    -------
    array
        Signed distances (n_spheres, n_obstacles).
        Positive = separated, negative = penetrating.
    """
    xp = backend
    diff = sphere_positions[:, None, :] - obstacle_centers[None, :, :]
    # local[s, o, :] = obstacle_rotations[o].T @ diff[s, o, :]
    local = xp.einsum('oji,soj->soi', obstacle_rotations, diff)

    xy_dist = xp.sqrt(local[..., 0] ** 2 + local[..., 1] ** 2 + 1e-10)
    z_abs = xp.abs(local[..., 2])
    radius = obstacle_radii[None, :]
    half_height = obstacle_half_heights[None, :]

    inside_radius = xy_dist <= radius
    inside_height = z_abs <= half_height

    side_dist = xy_dist - radius
    cap_dist = z_abs - half_height
    corner_dist = xp.sqrt(
        xp.maximum(xy_dist - radius, 0.0) ** 2
        + xp.maximum(z_abs - half_height, 0.0) ** 2 + 1e-10)
    interior_dist = -xp.minimum(radius - xy_dist, half_height - z_abs)

    surface_dist = xp.where(
        inside_radius & inside_height, interior_dist,
        xp.where(inside_radius, cap_dist,
                xp.where(inside_height, side_dist, corner_dist)))
    return surface_dist - sphere_radii[:, None]


def _safe_norm(x, backend, axis=-1, eps=1e-10):
    xp = backend
    return xp.sqrt(xp.sum(x ** 2, axis=axis) + eps)


def closest_point_on_box(points, box_centers, box_rotations,
                         box_half_extents, backend):
    """Signed distance from point(s) to a box, and the nearest point on
    the box *surface*.

    Broadcasts over any shared leading batch shape of ``points``,
    ``box_centers``, ``box_rotations`` and ``box_half_extents``. Matches
    :func:`skrobot.planner.trajectory_optimization.collision.
    point_to_box_distance` when a point is outside the box, but (a) is
    branch-free (``jnp.clip``/``jnp.where`` only) so it works under
    ``jax.jit``/``jax.grad``, and (b) returns a *signed* distance and a
    surface point (instead of clamping to 0) when a point is inside the
    box -- needed so a gradient-based optimiser still gets a push-out
    direction for a waypoint that starts out penetrating (mirrors the
    ``interior_dist`` branch of :func:`compute_cylinder_obstacle_distances`
    below).

    Parameters
    ----------
    points : array, (..., 3)
        Query points, world frame.
    box_centers : array, (..., 3)
        Box centers, world frame.
    box_rotations : array, (..., 3, 3)
        Box local->world rotation (columns = box local axes in world).
    box_half_extents : array, (..., 3)
    backend : module

    Returns
    -------
    signed_dist : array, (...,)
        Positive outside the box, negative inside (distance to the
        nearest face).
    closest_point : array, (..., 3)
        Nearest point on the box surface, world frame.
    """
    xp = backend
    diff = points - box_centers
    local = xp.einsum('...ji,...j->...i', box_rotations, diff)

    clipped = xp.clip(local, -box_half_extents, box_half_extents)
    outside_dist = _safe_norm(local - clipped, xp)
    is_outside = xp.any(xp.abs(local) > box_half_extents, axis=-1)

    slack = box_half_extents - xp.abs(local)
    push_axis = xp.argmin(slack, axis=-1)
    axis_onehot = xp.arange(local.shape[-1]) == push_axis[..., None]
    sign = xp.where(local >= 0, 1.0, -1.0)
    interior_point = xp.where(axis_onehot, sign * box_half_extents, local)
    interior_dist = -xp.min(slack, axis=-1)

    closest_local = xp.where(is_outside[..., None], clipped, interior_point)
    signed_dist = xp.where(is_outside, outside_dist, interior_dist)
    closest_world = box_centers + xp.einsum(
        '...ij,...j->...i', box_rotations, closest_local)
    return signed_dist, closest_world


def closest_point_on_cylinder(points, cyl_centers, cyl_rotations,
                              cyl_radii, cyl_half_heights, backend):
    """Signed distance from point(s) to a (flat-capped) cylinder, and the
    nearest point on the cylinder *surface*.

    Local +Z axis is the cylinder axis, matching
    :func:`compute_cylinder_obstacle_distances`'s convention. Same
    branch-free / signed-with-surface-point design as
    :func:`closest_point_on_box` (see its docstring); the exterior
    distance formula matches
    :func:`skrobot.planner.trajectory_optimization.collision.
    point_to_cylinder_distance`.

    Parameters
    ----------
    points : array, (..., 3)
    cyl_centers : array, (..., 3)
    cyl_rotations : array, (..., 3, 3)
    cyl_radii : array, (...,)
    cyl_half_heights : array, (...,)
    backend : module

    Returns
    -------
    signed_dist : array, (...,)
    closest_point : array, (..., 3)
        Nearest point on the cylinder surface, world frame.
    """
    xp = backend
    diff = points - cyl_centers
    local = xp.einsum('...ji,...j->...i', cyl_rotations, diff)
    xy = local[..., :2]
    z = local[..., 2]
    xy_dist = _safe_norm(xy, xp)
    z_abs = xp.abs(z)

    # A scalar (single-primitive) radius/half_height is a plain Python
    # float, not an array -- ``radius[..., None]`` below needs an array
    # (even 0-d) to be indexable.
    radius = xp.asarray(cyl_radii)
    half_height = xp.asarray(cyl_half_heights)

    inside_radius = xy_dist <= radius
    inside_height = z_abs <= half_height

    # Radial direction; falls back to an arbitrary axis when the point
    # sits exactly on the cylinder axis (xy_dist ~ 0), where the nearest
    # side-wall point is undefined anyway.
    safe_xy_dist = xp.maximum(xy_dist, 1e-8)
    on_axis = xy_dist < 1e-6
    fallback_dir = xp.zeros_like(xy)
    fallback_dir = xp.concatenate(
        [xp.ones_like(z)[..., None], xp.zeros_like(z)[..., None]], axis=-1)
    radial_dir = xp.where(
        on_axis[..., None], fallback_dir, xy / safe_xy_dist[..., None])
    side_xy = radial_dir * radius[..., None]
    cap_z = xp.where(z >= 0, half_height, -half_height)

    side_dist = xy_dist - radius
    cap_dist = z_abs - half_height
    corner_dist = _safe_norm(
        xp.concatenate(
            [xp.maximum(xy_dist - radius, 0.0)[..., None],
             xp.maximum(z_abs - half_height, 0.0)[..., None]], axis=-1),
        xp)

    radial_slack = radius - xy_dist
    axial_slack = half_height - z_abs
    push_radial = radial_slack < axial_slack

    outside_xy = xp.where(inside_radius[..., None], xy, side_xy)
    outside_z = xp.where(inside_height, z, cap_z)
    interior_xy = xp.where(push_radial[..., None], side_xy, xy)
    interior_z = xp.where(push_radial, z, cap_z)

    both_inside = inside_radius & inside_height
    closest_xy = xp.where(both_inside[..., None], interior_xy, outside_xy)
    closest_z = xp.where(both_inside, interior_z, outside_z)
    closest_local = xp.concatenate([closest_xy, closest_z[..., None]],
                                   axis=-1)

    signed_dist = xp.where(
        both_inside, -xp.minimum(radial_slack, axial_slack),
        xp.where(inside_radius, cap_dist,
                xp.where(inside_height, side_dist, corner_dist)))

    closest_world = cyl_centers + xp.einsum(
        '...ij,...j->...i', cyl_rotations, closest_local)
    return signed_dist, closest_world


def closest_point_on_sphere(points, sphere_centers, sphere_radii, backend):
    """Signed distance from point(s) to a sphere, and the nearest point
    on the sphere surface. Trivial counterpart to
    :func:`closest_point_on_box` / :func:`closest_point_on_cylinder`, for
    a uniform primitive interface (a sphere's surface point is always
    ``center + radius`` along the direction from the center to the
    query point, regardless of whether that point is inside or outside).
    """
    xp = backend
    sphere_radii = xp.asarray(sphere_radii)
    diff = points - sphere_centers
    dist_to_center = _safe_norm(diff, xp)
    direction = diff / dist_to_center[..., None]
    signed_dist = dist_to_center - sphere_radii
    closest_world = sphere_centers + direction * sphere_radii[..., None]
    return signed_dist, closest_world


_PRIMITIVE_CLOSEST_POINT_FNS = {
    'box': closest_point_on_box,
    'cylinder': closest_point_on_cylinder,
    'sphere': closest_point_on_sphere,
}


def primitive_pair_signed_distance(prim_a, prim_b, backend, n_iters=6):
    """Signed distance between two (batches of) convex primitives, each a
    box, cylinder or sphere.

    When either primitive is a sphere, this reduces to the exact,
    closed-form point-to-primitive distance (a sphere's surface is
    rotationally symmetric, so no iteration is needed). Otherwise (box
    vs box, box vs cylinder, cylinder vs cylinder -- shapes with no
    simple closed-form separation distance at arbitrary relative pose)
    this alternates a fixed number of nearest-surface-point projections
    between the two shapes (``center of B -> project onto A -> project
    onto B -> ...``) and reports the distance between the last two
    projected points. This is a heuristic, not a certified global
    optimum: for two convex, *separated* shapes it converges quickly to
    the true nearest points, but for deeply overlapping shapes it is not
    guaranteed to find the maximum-penetration axis. Optimisation costs
    built on this are therefore soft hints only, exactly like the
    sphere-based cost they replace -- the caller must still verify
    waypoints against the exact mesh (see ``collision_pairs_min_distance``
    in ``aero_demo/scripts/solve_palm_ik.py``).

    Parameters
    ----------
    prim_a, prim_b : dict
        ``{'type': 'box'|'cylinder'|'sphere', 'center': (..., 3),
        'rotation': (..., 3, 3), ...}`` in world frame, with the extra
        per-type keys :func:`closest_point_on_box` /
        :func:`closest_point_on_cylinder` / :func:`closest_point_on_sphere`
        expect (``half_extents`` / ``radius``+``half_height`` / ``radius``
        respectively; ``rotation`` is unused for ``sphere``). All arrays
        share a common leading batch shape.
    backend : module
    n_iters : int
        Fixed number of alternating projections (only used when neither
        primitive is a sphere).

    Returns
    -------
    array, (...,)
        Signed distance: positive when separated, negative when (the
        alternating projection detects) overlapping.
    """
    xp = backend

    def _project(point, prim):
        fn = _PRIMITIVE_CLOSEST_POINT_FNS[prim['type']]
        if prim['type'] == 'sphere':
            return fn(point, prim['center'], prim['radius'], xp)
        elif prim['type'] == 'box':
            return fn(point, prim['center'], prim['rotation'],
                      prim['half_extents'], xp)
        else:
            return fn(point, prim['center'], prim['rotation'],
                      prim['radius'], prim['half_height'], xp)

    if prim_a['type'] == 'sphere':
        signed_dist, _ = _project(prim_a['center'], prim_b)
        return signed_dist - prim_a['radius']
    if prim_b['type'] == 'sphere':
        signed_dist, _ = _project(prim_b['center'], prim_a)
        return signed_dist - prim_b['radius']

    point = prim_b['center']
    for _ in range(n_iters):
        _, point_a = _project(point, prim_a)
        dist_b, point = _project(point_a, prim_b)

    # ``dist_b`` (signed distance from the last A-projected point to B)
    # is negative exactly when that point ended up inside B, i.e. the
    # projected pair overlaps -- use it as the overlap sign, but report
    # the actual gap between the two converged surface points as the
    # magnitude (both are ~equal once the iteration has converged for a
    # separated pair).
    gap = _safe_norm(point_a - point, xp)
    return xp.where(dist_b < 0.0, -gap, gap)


def get_primitive_world_pose(bucket, link_positions, link_rotations,
                             backend):
    """World center (and, for box/cylinder, world rotation) of a batch of
    collision primitives anchored to specific kinematic-chain links.

    Counterpart to :func:`build_fk_functions`'s ``get_sphere_positions``
    for box/cylinder/sphere primitives (see
    :func:`skrobot.planner.trajectory_optimization.problem.
    TrajectoryProblem._compute_collision_primitives`, which builds
    ``bucket`` -- one such dict per primitive type, already composed
    with the kinematic-chain-link-to-actual-link offset the way
    ``sphere_centers_local`` is for spheres).

    Parameters
    ----------
    bucket : dict
        ``{'chain_idx': (n,) int, 'local_center': (n, 3)}``, plus
        ``'local_rotation': (n, 3, 3)`` for box/cylinder buckets (absent
        for sphere, whose orientation is irrelevant). Any extra keys
        (``half_extents``/``radius``/``half_height``) are ignored here.
    link_positions, link_rotations : array
        Per-kinematic-chain-link world pose, from ``get_link_transforms``
        (shape ``(n_joints, 3)`` / ``(n_joints, 3, 3)``).
    backend : module

    Returns
    -------
    world_center : array, (n, 3)
    world_rotation : array, (n, 3, 3) or None
        None when ``bucket`` has no ``'local_rotation'`` (sphere bucket).
    """
    xp = backend
    chain_idx = bucket['chain_idx']
    link_pos = link_positions[chain_idx]
    link_rot = link_rotations[chain_idx]
    world_center = link_pos + xp.einsum(
        '...ij,...j->...i', link_rot, bucket['local_center'])
    if 'local_rotation' in bucket:
        world_rotation = xp.einsum(
            '...ij,...jk->...ik', link_rot, bucket['local_rotation'])
    else:
        world_rotation = None
    return world_center, world_rotation


def compute_self_collision_distances(sphere_positions, sphere_radii,
                                     pairs_i, pairs_j, backend):
    """Compute signed distances for self-collision pairs.

    Parameters
    ----------
    sphere_positions : array
        Collision sphere positions (n_spheres, 3).
    sphere_radii : array
        Collision sphere radii (n_spheres,).
    pairs_i : array
        First sphere indices for each pair.
    pairs_j : array
        Second sphere indices for each pair.
    backend : module
        Array module.

    Returns
    -------
    array
        Signed distances for each pair.
    """
    xp = backend
    pos_i = sphere_positions[pairs_i]
    pos_j = sphere_positions[pairs_j]
    rad_i = sphere_radii[pairs_i]
    rad_j = sphere_radii[pairs_j]

    diff = pos_i - pos_j
    dists = xp.sqrt(xp.sum(diff ** 2, axis=-1) + 1e-10)
    signed_dists = dists - rad_i - rad_j
    return signed_dists


def rotation_error_vector(actual_rot, target_rot, backend):
    """Compute rotation error vector from anti-symmetric part of R_err.

    Extracts three independent components from the anti-symmetric part
    of ``actual_rot @ target_rot^T``.  The resulting 3-vector is zero
    when the two rotations are identical.

    Parameters
    ----------
    actual_rot : array
        Actual rotation matrix (3, 3).
    target_rot : array
        Target rotation matrix (3, 3).
    backend : module
        Array module (numpy or jax.numpy).

    Returns
    -------
    array
        Rotation error vector (3,).
    """
    xp = backend
    R_err = xp.matmul(actual_rot, xp.transpose(target_rot))
    return xp.stack([
        R_err[1, 0] - R_err[0, 1],
        R_err[2, 0] - R_err[0, 2],
        R_err[2, 1] - R_err[1, 2],
    ])


def compute_collision_residuals(signed_distances, activation_distance, backend):
    """Convert signed distances to collision residuals.

    Parameters
    ----------
    signed_distances : array
        Signed distances (positive = separated).
    activation_distance : float
        Distance threshold for activation.
    backend : module
        Array module.

    Returns
    -------
    array
        Collision residuals (positive when too close).
    """
    xp = backend
    return xp.maximum(0.0, activation_distance - signed_distances)


def build_chain_link_transforms_with_base(fk_data, backend):
    """Like :func:`build_fk_functions`'s ``get_link_transforms`` but the
    base pose is an *argument*, not a closure-captured constant.

    Needed for trajectory optimisation where the base translation /
    rotation are part of the per-waypoint variable (floating-base DoF).

    Parameters
    ----------
    fk_data : dict
        Same FK data dict consumed by :func:`build_fk_functions`.
    backend : module
        Array module.

    Returns
    -------
    callable
        ``get_link_transforms(angles, base_pos, base_rot)`` returning
        ``(positions, rotations)``.
    """
    xp = backend
    link_trans = fk_data['link_translations']
    link_rots = fk_data['link_rotations']
    joint_axes = fk_data['joint_axes']
    n_joints = fk_data['n_joints']
    ref_angles = fk_data.get('ref_angles')

    def get_link_transforms(angles, base_pos, base_rot):
        positions = []
        rotations = []
        current_pos = base_pos
        current_rot = base_rot
        for i in range(n_joints):
            current_pos = current_pos + current_rot @ link_trans[i]
            current_rot = current_rot @ link_rots[i]
            delta = angles[i]
            if ref_angles is not None:
                delta = delta - ref_angles[i]
            joint_rot = rodrigues_rotation(xp, joint_axes[i], delta)
            current_rot = current_rot @ joint_rot
            positions.append(current_pos)
            rotations.append(current_rot)
        return xp.stack(positions), xp.stack(rotations)

    return get_link_transforms


def build_chain_ee_pose_with_base(fk_data, backend):
    """EE-pose function with explicit base pose argument.

    Counterpart to :func:`build_chain_link_transforms_with_base`. The
    EE offset (last_link -> move_target) baked into ``fk_data`` is
    applied at the end.
    """
    get_link_transforms = build_chain_link_transforms_with_base(fk_data, backend)
    ee_off_pos = fk_data.get('ee_offset_position')
    ee_off_rot = fk_data.get('ee_offset_rotation')

    def get_ee_pose(angles, base_pos, base_rot):
        positions, rotations = get_link_transforms(angles, base_pos, base_rot)
        last_pos = positions[-1]
        last_rot = rotations[-1]
        if ee_off_pos is not None:
            ee_pos = last_pos + last_rot @ ee_off_pos
        else:
            ee_pos = last_pos
        if ee_off_rot is not None:
            ee_rot = last_rot @ ee_off_rot
        else:
            ee_rot = last_rot
        return ee_pos, ee_rot

    return get_ee_pose


def prepare_fk_data(problem, backend):
    """Prepare FK data dictionary from problem definition.

    Parameters
    ----------
    problem : TrajectoryProblem
        Trajectory optimization problem.
    backend : module
        Array module for array conversion.

    Returns
    -------
    dict
        FK data dictionary for build_fk_functions().
    """
    xp = backend
    fk_params = problem.fk_params

    fk_data = {
        'link_translations': xp.array(fk_params['link_translations']),
        'link_rotations': xp.array(fk_params['link_rotations']),
        'joint_axes': xp.array(fk_params['joint_axes']),
        'base_position': xp.array(fk_params['base_position']),
        'base_rotation': xp.array(fk_params['base_rotation']),
        'n_joints': fk_params['n_joints'],
        'ee_offset_position': xp.array(fk_params['ee_offset_position']),
        'ee_offset_rotation': xp.array(fk_params['ee_offset_rotation']),
        'ref_angles': xp.array(fk_params['ref_angles']),
    }

    # Add collision data if available
    if problem.collision_spheres is not None:
        fk_data['collision_link_to_chain_idx'] = xp.array(
            problem.collision_link_to_chain_idx)
        fk_data['collision_link_offsets_pos'] = xp.array(
            problem.collision_link_offsets_pos)
        fk_data['collision_link_offsets_rot'] = xp.array(
            problem.collision_link_offsets_rot)
        fk_data['sphere_centers_local'] = xp.array(
            problem.collision_spheres['sphere_centers_local'])
        fk_data['sphere_radii'] = xp.array(
            problem.collision_spheres['sphere_radii'])
        fk_data['collision_link_indices'] = xp.array(
            problem.collision_spheres['link_indices'])

    # Add primitive (box/cylinder/sphere) collision data if available
    # -- only the jaxls backend consumes these; other backends keep
    # using 'sphere_radii'/'sphere_centers_local' above unchanged.
    if getattr(problem, 'collision_primitives', None):
        fk_data['collision_primitives'] = {
            ptype: {k: xp.array(v) for k, v in bucket.items()}
            for ptype, bucket in problem.collision_primitives.items()
        }
    if getattr(problem, 'self_collision_primitive_pairs', None):
        fk_data['self_collision_primitive_pairs'] = {
            combo: {k: xp.array(v) for k, v in pair_rows.items()}
            for combo, pair_rows in
            problem.self_collision_primitive_pairs.items()
        }

    return fk_data


__all__ = [
    'build_fk_functions',
    'rotation_error_vector',
    'rotation_error_log',
    'pose_error_log',
    'compute_sphere_obstacle_distances',
    'compute_cylinder_obstacle_distances',
    'compute_self_collision_distances',
    'compute_collision_residuals',
    'prepare_fk_data',
    'closest_point_on_box',
    'closest_point_on_cylinder',
    'closest_point_on_sphere',
    'primitive_pair_signed_distance',
    'get_primitive_world_pose',
]
