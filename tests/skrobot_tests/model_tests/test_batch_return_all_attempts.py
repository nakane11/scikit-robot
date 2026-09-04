"""Tests for batch_inverse_kinematics(return_all_attempts=True)."""
import numpy as np
import pytest

import skrobot
from skrobot.coordinates import Coordinates
from skrobot.pycompat import HAS_JAX


ATTEMPTS = 4


def _targets(robot, n):
    ee = robot.rarm_end_coords.worldpos()
    return [Coordinates(pos=ee + np.array([0.05 * i, 0.0, 0.0]))
            for i in range(n)]


@pytest.mark.skipif(not HAS_JAX, reason='requires the jax backend')
def test_return_all_attempts_returns_one_solution_per_attempt():
    """Every (target, attempt) pair comes back, target-major."""
    robot = skrobot.models.Fetch()
    targets = _targets(robot, 3)
    kwargs = dict(
        move_target=robot.rarm_end_coords,
        link_list=robot.rarm.link_list,
        rotation_mask=False,
        stop=50, thre=0.01,
        backend='jax', initial_angles='current',
        attempts_per_pose=ATTEMPTS,
    )

    solutions, success_flags, _ = robot.batch_inverse_kinematics(
        target_coords=targets, return_all_attempts=True, **kwargs)
    assert len(solutions) == len(targets) * ATTEMPTS
    assert len(success_flags) == len(targets) * ATTEMPTS

    # The default (best-of-attempts) path keeps returning one per target,
    # and its solutions are a subset of the ones returned above -- the
    # selection happens after the same solve.
    best, best_flags, _ = robot.batch_inverse_kinematics(
        target_coords=targets, **kwargs)
    assert len(best) == len(targets)
    assert len(best_flags) == len(targets)

    # Attempt 0 of each target starts from the current angles, so it is the
    # same problem in both calls: whenever it converged, the same target's
    # best-of pick must have converged too.
    for i in range(len(targets)):
        if success_flags[i * ATTEMPTS]:
            assert best_flags[i]


@pytest.mark.skipif(not HAS_JAX, reason='requires the jax backend')
def test_return_all_attempts_with_use_base():
    """base_poses are returned per (target, attempt) as well."""
    robot = skrobot.models.Fetch()
    ee = robot.rarm_end_coords.worldpos()
    targets = [Coordinates(pos=ee + np.array([0.8, 0.0, 0.0])),
               Coordinates(pos=ee + np.array([0.9, 0.0, 0.0]))]

    solutions, base_poses, success_flags, _ = \
        robot.batch_inverse_kinematics(
            target_coords=targets,
            move_target=robot.rarm_end_coords,
            link_list=robot.rarm.link_list,
            rotation_mask=False,
            stop=50, thre=0.01,
            backend='jax', initial_angles='current',
            attempts_per_pose=ATTEMPTS,
            use_base='planar',
            base_limits=[(-2.0, 2.0), (-2.0, 2.0), (-np.pi, np.pi)],
            return_all_attempts=True)

    n = len(targets) * ATTEMPTS
    assert len(solutions) == n
    assert len(base_poses) == n
    assert len(success_flags) == n


def test_return_all_attempts_rejects_numpy_backend():
    robot = skrobot.models.Fetch()
    with pytest.raises(ValueError):
        robot.batch_inverse_kinematics(
            target_coords=_targets(robot, 2),
            move_target=robot.rarm_end_coords,
            link_list=robot.rarm.link_list,
            rotation_mask=False,
            stop=10, thre=0.01,
            backend='numpy', initial_angles='current',
            attempts_per_pose=ATTEMPTS,
            return_all_attempts=True)
