import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest

import numpy as np

from skrobot.pycompat import HAS_JAX


HAS_JAXLS = False
if HAS_JAX:
    try:
        import jaxls  # noqa: F401
        if hasattr(jaxls, 'LeastSquaresProblem'):
            HAS_JAXLS = True
    except ImportError:
        pass

requires_jax = unittest.skipUnless(HAS_JAX, 'JAX is required')
requires_jaxls = unittest.skipUnless(
    HAS_JAX and HAS_JAXLS, 'JAX and jaxls are required')


class TestJaxlsConstants(unittest.TestCase):
    """What JaxlsSolver bakes into the traced graph must be reproducible."""

    @requires_jax
    def test_quantize_nested_fk_data(self):
        from skrobot.planner.trajectory_optimization.solvers.jaxls_solver import _quantize_fk_constants

        data = {
            'n_joints': 7,
            'indices': np.array([0, 1, 2]),
            'translations': np.array([[1.0 + 3e-13, -1e-14], [0.5, -0.0]]),
            'nested': {'axis': np.array([0.1234567891234, -2e-17])},
        }
        # numpy stands in for jnp: float64 whatever the process's JAX mode.
        out = _quantize_fk_constants(data, np)

        # Integers are left alone.
        self.assertEqual(out['n_joints'], 7)
        np.testing.assert_array_equal(out['indices'], [0, 1, 2])
        # Floats are rounded and carry no negative zero.
        np.testing.assert_array_equal(
            np.asarray(out['translations']), [[1.0, 0.0], [0.5, 0.0]])
        np.testing.assert_array_equal(
            np.asarray(out['nested']['axis']), [0.123456789, 0.0])
        for value in (out['translations'], out['nested']['axis']):
            self.assertFalse(np.signbit(np.asarray(value)).any())

    @requires_jax
    def test_pin_cost_group_order(self):
        """The cost groups get names that sort in the order they were added."""
        from skrobot.planner.trajectory_optimization.solvers.jaxls_solver import _pin_cost_group_order

        def make_cost(name):
            # Like jaxls's Cost.factory: every residual has the same name.
            return types.SimpleNamespace(
                name=name, compute_residual=lambda: None)

        costs = [make_cost(n) for n in ('smoothness', 'acceleration',
                                        'world_collision', 'posture')]
        self.assertEqual(
            len({c.compute_residual.__qualname__ for c in costs}), 1)

        _pin_cost_group_order(costs)

        names = [c.compute_residual.__qualname__ for c in costs]
        self.assertEqual(names, sorted(names))
        self.assertEqual(len(set(names)), len(costs))
        self.assertTrue(names[2].endswith('world_collision'))


# What the worker does is what examples/collision_free_trajectory.py does
# with the jaxls solver: a PR2 right-arm trajectory with smoothness,
# acceleration, world collision and self collision costs.
_WORKER = textwrap.dedent('''
    import collections
    import json
    import sys
    import time

    import numpy as np

    import jax
    import jax.monitoring
    import jax.numpy as jnp
    import skrobot
    from skrobot.backend import enable_persistent_cache
    from skrobot.planner.trajectory_optimization import TrajectoryProblem
    from skrobot.planner.trajectory_optimization.solvers import create_solver
    from skrobot.planner.trajectory_optimization.trajectory import (
        interpolate_trajectory)

    cache_dir, seed, out = sys.argv[1:4]
    seed = int(seed)

    events = collections.Counter()
    durations = collections.Counter()
    jax.monitoring.register_event_listener(
        lambda name, **kwargs: events.update([name]))
    jax.monitoring.register_event_duration_secs_listener(
        lambda name, secs, **kwargs: durations.update({name: secs}))

    # JAX is already in use when the cache is switched on.
    jnp.arange(3).sum().block_until_ready()
    events.clear()
    durations.clear()
    enable_persistent_cache(cache_dir)

    robot = skrobot.models.PR2()
    if seed:
        # The way the model got to its pose differs from process to process.
        rng = np.random.RandomState(seed)
        for joint in robot.joint_list:
            if np.isfinite(joint.min_angle) and np.isfinite(joint.max_angle):
                joint.joint_angle(rng.uniform(joint.min_angle, joint.max_angle))
    robot.init_pose()

    link_list = [
        robot.r_shoulder_pan_link, robot.r_shoulder_lift_link,
        robot.r_upper_arm_roll_link, robot.r_elbow_flex_link,
        robot.r_forearm_roll_link, robot.r_wrist_flex_link,
        robot.r_wrist_roll_link]
    coll_link_list = [
        robot.r_upper_arm_link, robot.r_forearm_link,
        robot.r_gripper_palm_link, robot.r_gripper_r_finger_link,
        robot.r_gripper_l_finger_link]
    end_coords = skrobot.coordinates.CascadedCoords(
        parent=robot.r_gripper_tool_frame, name='right_arm_end_coords')
    start = np.array([0.564, 0.35, -0.74, -0.7, -0.7, -0.17, -0.63])
    goal = np.deg2rad([-60, 74, -70, -120, -20, -30, 180])

    start_time = time.perf_counter()
    problem = TrajectoryProblem(
        robot_model=robot, link_list=link_list, n_waypoints=10, dt=0.1,
        move_target=end_coords)
    problem.add_smoothness_cost(weight=1.0)
    problem.add_acceleration_cost(weight=0.1)
    problem.add_collision_cost(
        collision_link_list=coll_link_list,
        world_obstacles=[{'type': 'sphere', 'center': [0.9, -0.2, 0.9],
                          'radius': 0.5}],
        weight=1000.0, activation_distance=0.15)
    problem.add_self_collision_cost(weight=1000.0, activation_distance=0.02)
    solver = create_solver('jaxls', max_iterations=100)
    result = solver.solve(problem, interpolate_trajectory(start, goal, 10))
    trajectory = np.asarray(result.trajectory)
    elapsed = time.perf_counter() - start_time

    np.save(out, trajectory)
    print(json.dumps({
        'platform': jax.default_backend(),
        'hits': events['/jax/compilation_cache/cache_hits'],
        'misses': events['/jax/compilation_cache/cache_misses'],
        'compile_secs': durations['/jax/core/compile/backend_compile_duration'],
        'elapsed_secs': elapsed,
    }))
''')


def _describe(run):
    return '{hits} hits, {misses} misses, {elapsed_secs:.2f} s ' \
        '(compiling {compile_secs:.2f} s)'.format(**run)


class TestJaxlsPersistentCacheAcrossProcesses(unittest.TestCase):
    """The trajectory optimization compiled by one process is reused.

    Hits and misses are what JAX itself reports. The times are printed for
    the reader (``pytest -rP`` or ``-s``) but not asserted: they depend on
    the machine.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.tmp, 'cache')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, seed):
        out = os.path.join(self.tmp, 'trajectory_{}.npy'.format(seed))
        env = dict(os.environ)
        # CPU unless the caller picked a platform (JAX_PLATFORMS=cuda runs
        # the same checks on a GPU).
        env.setdefault('JAX_PLATFORMS', 'cpu')
        for key in ('SKROBOT_JAX_CACHE_DIR', 'JAX_COMPILATION_CACHE_DIR'):
            env.pop(key, None)
        proc = subprocess.run(
            [sys.executable, '-c', _WORKER, self.cache_dir, str(seed), out],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=900)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        return json.loads(proc.stdout.splitlines()[-1]), np.load(out)

    @requires_jaxls
    def test_trajectory_optimization_cache_hit_in_next_process(self):
        first, first_trajectory = self._run(seed=0)
        report = ['[{}] jaxls trajectory optimization, first process: {}'
                  .format(first['platform'], _describe(first))]
        later = []
        for seed in (1, 2):
            run, trajectory = self._run(seed=seed)
            difference = np.abs(trajectory - first_trajectory).max()
            later.append((run, difference))
            report.append('  later process (model posed differently, '
                          'seed {}): {}; trajectory differs by {:.1e} rad'
                          .format(seed, _describe(run), difference))
        report = '\n'.join(report)
        print(report)

        self.assertEqual(first['hits'], 0, report)
        self.assertGreater(first['misses'], 0, report)
        for run, difference in later:
            # The whole solver (jit_solve included) is loaded, not compiled.
            self.assertEqual(run['misses'], 0, report)
            self.assertEqual(run['hits'], first['misses'], report)
            # The solver runs in float32, and a GPU does not add up in a
            # fixed order, so only the same solution is expected.
            self.assertLess(difference, 1e-3, report)


if __name__ == '__main__':
    unittest.main()
