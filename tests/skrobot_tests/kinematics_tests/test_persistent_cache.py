import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

import numpy as np

from skrobot.pycompat import HAS_JAX


def requires_jax(test_func):
    """Decorator to skip tests if JAX is not available."""
    return unittest.skipUnless(HAS_JAX, "JAX not available")(test_func)


def _random_pose(robot, seed):
    """Move the arm to a random pose, as earlier work on the model would."""
    rng = np.random.RandomState(seed)
    for link in robot.rarm.link_list:
        joint = link.joint
        if np.isfinite(joint.min_angle) and np.isfinite(joint.max_angle):
            joint.joint_angle(rng.uniform(joint.min_angle, joint.max_angle))


class TestFKConstantQuantization(unittest.TestCase):
    """The FK constants must not depend on the model's earlier poses."""

    @classmethod
    def setUpClass(cls):
        if not HAS_JAX:
            return

        from skrobot.models import Panda
        from skrobot.models import R8_6

        cls.robots = {'panda': Panda(), 'r8_6': R8_6()}

    def _fk_params(self, robot):
        from skrobot.kinematics.differentiable import extract_fk_parameters

        return extract_fk_parameters(
            robot, robot.rarm.link_list, robot.rarm.end_coords)

    @requires_jax
    def test_quantize_rounds_and_removes_negative_zero(self):
        from skrobot.kinematics.differentiable import _quantize_fk_constant

        values = _quantize_fk_constant(
            [[1.0 + 3e-13, -1e-14], [0.1234567891234, -0.0]])
        np.testing.assert_array_equal(
            values, [[1.0, 0.0], [0.123456789, 0.0]])
        # -0.0 would trace to a different graph than 0.0.
        self.assertFalse(np.signbit(values).any())

    @requires_jax
    def test_fk_params_independent_of_pose_history(self):
        """Same model, different earlier poses: bit-identical constants."""
        keys = ('link_translations', 'link_rotations', 'joint_axes',
                'base_position', 'base_rotation',
                'ee_offset_position', 'ee_offset_rotation')
        for name, robot in self.robots.items():
            robot.reset_pose()
            reference = self._fk_params(robot)
            for seed in (1, 2, 3):
                _random_pose(robot, seed)
                params = self._fk_params(robot)
                for key in keys:
                    np.testing.assert_array_equal(
                        params[key], reference[key],
                        err_msg='{} {} (seed {})'.format(name, key, seed))
                    self.assertFalse(
                        np.signbit(params[key][params[key] == 0]).any(),
                        '{} {} has a negative zero'.format(name, key))

    @requires_jax
    def test_fk_params_still_match_robot(self):
        """Rounding to 1e-9 must not change what the FK computes."""
        from skrobot.backend import get_backend
        from skrobot.kinematics.differentiable import forward_kinematics_ee

        backend = get_backend('jax')
        for robot in self.robots.values():
            _random_pose(robot, 7)
            link_list = robot.rarm.link_list
            fk_params = self._fk_params(robot)
            angles = np.array([l.joint.joint_angle() for l in link_list])
            pos, rot = forward_kinematics_ee(
                backend, backend.array(angles), fk_params)
            move_target = robot.rarm.end_coords
            np.testing.assert_allclose(
                backend.to_numpy(pos), move_target.worldpos(), atol=1e-6)
            np.testing.assert_allclose(
                backend.to_numpy(rot), move_target.worldrot(), atol=1e-6)


class TestEnablePersistentCache(unittest.TestCase):
    """enable_persistent_cache sets JAX's options and the directory."""

    _OPTIONS = ('jax_compilation_cache_dir',
                'jax_persistent_cache_min_compile_time_secs',
                'jax_persistent_cache_min_entry_size_bytes')
    _ENV = ('SKROBOT_JAX_CACHE_DIR', 'JAX_COMPILATION_CACHE_DIR',
            'XDG_CACHE_HOME')

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.env = {k: os.environ.pop(k, None) for k in self._ENV}
        if HAS_JAX:
            import jax
            self.options = {k: getattr(jax.config, k) for k in self._OPTIONS}

    def tearDown(self):
        for k, v in self.env.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
        if HAS_JAX:
            import jax
            for k, v in self.options.items():
                jax.config.update(k, v)
        shutil.rmtree(self.tmp, ignore_errors=True)

    @requires_jax
    def test_sets_jax_options_and_creates_directory(self):
        import jax

        from skrobot.backend import enable_persistent_cache

        path = os.path.join(self.tmp, 'a', 'b')
        used = enable_persistent_cache(
            path, min_compile_time_secs=2.5, min_entry_size_bytes=10)

        self.assertEqual(used, path)
        self.assertTrue(os.path.isdir(path))
        self.assertEqual(jax.config.jax_compilation_cache_dir, path)
        self.assertEqual(
            jax.config.jax_persistent_cache_min_compile_time_secs, 2.5)
        self.assertEqual(
            jax.config.jax_persistent_cache_min_entry_size_bytes, 10)

    @requires_jax
    def test_default_caches_everything(self):
        import jax

        from skrobot.backend import enable_persistent_cache

        enable_persistent_cache(os.path.join(self.tmp, 'c'))

        self.assertEqual(
            jax.config.jax_persistent_cache_min_compile_time_secs, 0.0)
        self.assertEqual(
            jax.config.jax_persistent_cache_min_entry_size_bytes, 0)

    def test_default_dir_priority(self):
        from skrobot.backend.jax_cache import default_cache_dir

        os.environ['XDG_CACHE_HOME'] = os.path.join(self.tmp, 'xdg')
        self.assertEqual(
            default_cache_dir(),
            os.path.join(self.tmp, 'xdg', 'skrobot', 'jax_compilation_cache'))

        os.environ['JAX_COMPILATION_CACHE_DIR'] = os.path.join(self.tmp, 'j')
        self.assertEqual(default_cache_dir(), os.path.join(self.tmp, 'j'))

        os.environ['SKROBOT_JAX_CACHE_DIR'] = os.path.join(self.tmp, 's')
        self.assertEqual(default_cache_dir(), os.path.join(self.tmp, 's'))


_WORKER = textwrap.dedent('''
    import collections
    import json
    import sys
    import time

    import numpy as np

    import jax
    import jax.monitoring
    import jax.numpy as jnp
    from skrobot.backend import enable_persistent_cache
    from skrobot.kinematics.differentiable import create_batch_ik_solver
    from skrobot.models import Panda
    from skrobot.models import R8_6

    cache_dir, robot_name, seed, out = sys.argv[1:5]
    seed = int(seed)

    # What JAX itself reports: a hit is an executable loaded from the cache
    # instead of compiled.
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

    robot = Panda() if robot_name == 'panda' else R8_6()
    link_list = robot.rarm.link_list
    if seed:
        rng = np.random.RandomState(seed)
        for link in link_list:
            joint = link.joint
            if np.isfinite(joint.min_angle) and np.isfinite(joint.max_angle):
                joint.joint_angle(rng.uniform(joint.min_angle, joint.max_angle))

    # Back to the standard pose: the model is the same in every process, but
    # the way it got there (the history) is not.
    if robot_name == 'panda':
        robot.reset_manip_pose()
    else:
        robot.reset_pose()
    move_target = robot.rarm.end_coords
    initial = np.array([[l.joint.joint_angle() for l in link_list]])
    positions = np.array([move_target.worldpos()])
    rotations = np.array([move_target.worldrot()])

    start = time.perf_counter()
    solver = create_batch_ik_solver(
        robot, link_list, move_target, backend_name='jax')
    solutions, success, errors = solver(
        positions, rotations, initial_angles=initial, max_iterations=30)
    solutions = np.asarray(solutions)
    elapsed = time.perf_counter() - start
    np.save(out, solutions)
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


class TestPersistentCacheAcrossProcesses(unittest.TestCase):
    """The cache written by one process is hit by the next one.

    Hits and misses are what JAX itself reports. The times are printed for
    the reader (``pytest -rP`` or ``-s``) but not asserted: they depend on
    the machine.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, cache_dir, robot_name, seed):
        out = os.path.join(self.tmp, '{}_{}.npy'.format(robot_name, seed))
        env = dict(os.environ)
        # CPU unless the caller picked a platform (JAX_PLATFORMS=cuda runs
        # the same checks on a GPU).
        env.setdefault('JAX_PLATFORMS', 'cpu')
        for key in ('SKROBOT_JAX_CACHE_DIR', 'JAX_COMPILATION_CACHE_DIR'):
            env.pop(key, None)
        proc = subprocess.run(
            [sys.executable, '-c', _WORKER,
             cache_dir, robot_name, str(seed), out],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=600)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        return json.loads(proc.stdout.splitlines()[-1]), np.load(out)

    @requires_jax
    def test_batch_ik_cache_hit_in_next_process(self):
        """Plain batch IK, models that were posed differently before."""
        for robot_name in ('panda', 'r8_6'):
            with self.subTest(robot=robot_name):
                cache_dir = os.path.join(self.tmp, robot_name)
                first, first_solutions = self._run(cache_dir, robot_name, 0)
                report = ['[{}] {} first process: {}'.format(
                    first['platform'], robot_name, _describe(first))]
                later = []
                for seed in (1, 2):
                    run, solutions = self._run(cache_dir, robot_name, seed)
                    later.append((seed, run, solutions))
                    report.append('  later process (seed {}): {}'.format(
                        seed, _describe(run)))
                report = '\n'.join(report)
                print(report)

                self.assertEqual(first['hits'], 0, report)
                self.assertGreater(first['misses'], 0, report)
                for seed, run, solutions in later:
                    # Everything compiled by the first process is loaded.
                    self.assertEqual(run['misses'], 0, report)
                    self.assertEqual(run['hits'], first['misses'], report)
                    np.testing.assert_allclose(
                        solutions, first_solutions, atol=1e-9)


if __name__ == '__main__':
    unittest.main()
