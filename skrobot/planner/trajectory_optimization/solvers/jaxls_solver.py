"""JAXls-based trajectory optimization solver.

This solver uses jaxls (JAX Least Squares) for robust nonlinear
least squares optimization with constraints.
"""

import os
import platform


# Ensure CPU backend on Mac before JAX imports
if platform.system() == 'Darwin':
    if 'JAX_PLATFORMS' not in os.environ:
        os.environ['JAX_PLATFORMS'] = 'cpu'

import numpy as np

from skrobot.planner.trajectory_optimization.solvers.base import BaseSolver
from skrobot.planner.trajectory_optimization.solvers.base import SolverResult


_JAXLS_INSTALL_HINT = (
    "jaxls is required for JaxlsSolver but is not importable.\n"
    "It is not on PyPI, so 'pip install jaxls' / 'uv pip install jaxls' "
    "will not work.\n"
    "Install it from source instead:\n"
    "    pip install \"git+https://github.com/brentyi/jaxls.git\""
)


def _require_jaxls():
    """Import jaxls or raise ImportError with the correct install hint."""
    try:
        import jaxls  # noqa: F401
        if not hasattr(jaxls, 'LeastSquaresProblem'):
            raise ImportError("Imported 'jaxls' package is missing 'LeastSquaresProblem'.")
    except ImportError as e:
        raise ImportError(
            "{}\nOriginal error: {}".format(_JAXLS_INSTALL_HINT, e)) from e


def _root_link_world_pose(problem):
    """Return the world pose of the robot's root link as numpy arrays."""
    root = problem.robot_model.root_link.worldcoords()
    return root.worldpos().astype(np.float64), \
        root.worldrot().astype(np.float64)


def _chain_parent_relative_to_root(fk_params, root_pos, root_rot):
    """Compute (rel_pos, rel_rot): chain parent pose in root_link frame."""
    natural_pos = np.asarray(fk_params['base_position'], dtype=np.float64)
    natural_rot = np.asarray(fk_params['base_rotation'], dtype=np.float64)
    root_T = np.eye(4)
    root_T[:3, :3] = root_rot
    root_T[:3, 3] = root_pos
    inv = np.linalg.inv(root_T)
    rel_pos = inv[:3, :3] @ natural_pos + inv[:3, 3]
    rel_rot = inv[:3, :3] @ natural_rot
    return rel_pos, rel_rot


def _build_root_relative_chain_fks(problem, fk_data, jnp_module):
    """Per-chain link FK that accepts root_link world pose as base.

    Each callable composes the chain's natural parent transform
    (captured at extraction time, relative to root_link) with the
    floating-base pose at call time, then runs the chain's serial FK.
    """
    from skrobot.planner.trajectory_optimization.fk_utils import build_chain_link_transforms_with_base

    root_pos_np, root_rot_np = _root_link_world_pose(problem)
    callables = []
    for fkp in problem.fk_params_per_chain:
        jdata = {
            'link_translations': jnp_module.array(fkp['link_translations']),
            'link_rotations':    jnp_module.array(fkp['link_rotations']),
            'joint_axes':        jnp_module.array(fkp['joint_axes']),
            'n_joints':          fkp['n_joints'],
            'ref_angles':        jnp_module.array(fkp['ref_angles']),
        }
        rel_pos_np, rel_rot_np = _chain_parent_relative_to_root(
            fkp, root_pos_np, root_rot_np)
        rel_pos = jnp_module.asarray(rel_pos_np)
        rel_rot = jnp_module.asarray(rel_rot_np)
        inner = build_chain_link_transforms_with_base(jdata, jnp_module)

        def _make(inner=inner, rel_pos=rel_pos, rel_rot=rel_rot):
            def chain_fk(angles, base_pos, base_rot):
                parent_pos = base_pos + base_rot @ rel_pos
                parent_rot = base_rot @ rel_rot
                return inner(angles, parent_pos, parent_rot)
            return chain_fk
        callables.append(_make())
    return callables


def _build_root_relative_chain_ee_fks(problem, jnp_module):
    """Per-chain EE pose FK that accepts root_link world pose as base."""
    from skrobot.planner.trajectory_optimization.fk_utils import build_chain_ee_pose_with_base

    root_pos_np, root_rot_np = _root_link_world_pose(problem)
    callables = []
    for fkp in problem.fk_params_per_chain:
        jdata = {
            'link_translations': jnp_module.array(fkp['link_translations']),
            'link_rotations':    jnp_module.array(fkp['link_rotations']),
            'joint_axes':        jnp_module.array(fkp['joint_axes']),
            'n_joints':          fkp['n_joints'],
            'ref_angles':        jnp_module.array(fkp['ref_angles']),
            'ee_offset_position': jnp_module.array(
                fkp['ee_offset_position']),
            'ee_offset_rotation': jnp_module.array(
                fkp['ee_offset_rotation']),
        }
        rel_pos_np, rel_rot_np = _chain_parent_relative_to_root(
            fkp, root_pos_np, root_rot_np)
        rel_pos = jnp_module.asarray(rel_pos_np)
        rel_rot = jnp_module.asarray(rel_rot_np)
        inner = build_chain_ee_pose_with_base(jdata, jnp_module)

        def _make(inner=inner, rel_pos=rel_pos, rel_rot=rel_rot):
            def ee_fk(angles, base_pos, base_rot):
                parent_pos = base_pos + base_rot @ rel_pos
                parent_rot = base_rot @ rel_rot
                return inner(angles, parent_pos, parent_rot)
            return ee_fk
        callables.append(_make())
    return callables


class JaxlsSolver(BaseSolver):
    """JAXls-based trajectory optimization solver.

    Uses Levenberg-Marquardt algorithm for robust optimization.
    Supports both soft costs and hard constraints via augmented Lagrangian.

    Dynamic values (constraint targets, Cartesian path targets) are passed
    as frozen ``jaxls.Var`` objects (``tangent_dim=0``) so that the JIT-
    compiled problem can be reused when only these values change.
    """

    def __init__(
        self,
        max_iterations=100,
        verbose=False,
    ):
        """Initialize JAXls solver.

        Parameters
        ----------
        max_iterations : int
            Maximum optimization iterations.
        verbose : bool
            Print optimization progress.
        """
        super().__init__(verbose=verbose)
        # Fail fast with an actionable hint if jaxls is missing — it is
        # not on PyPI, so the usual ``pip install jaxls`` will not work.
        _require_jaxls()
        self.max_iterations = max_iterations
        self._cached_problem = None
        self._cached_traj_var = None
        self._cached_constraint_param_var = None
        self._cached_cartesian_pos_param_var = None
        self._cached_cartesian_rot_param_var = None
        self._cached_ee_wp_pos_param_var = None
        self._cached_ee_wp_rot_param_var = None
        self._cached_sphere_obs_param_var = None
        self._cached_cyl_geom_param_var = None
        self._cached_cyl_rotation_param_var = None
        self._cached_constraint_ids = None
        self._cached_has_cartesian = False
        self._cached_has_cart_rot = False
        self._cached_has_ee_waypoints = False
        self._cached_has_sphere_obs = False
        self._cached_has_cyl_obs = False
        self._cache_key = None

    def _make_cache_key(self, problem):
        """Create a structure-only cache key.

        The key captures the problem *structure* (number of waypoints,
        residual names/weights, constraint layout) but NOT the dynamic
        values (constraint targets, Cartesian targets, initial trajectory).
        This allows the compiled problem to be reused when only values
        change.

        Parameters
        ----------
        problem : TrajectoryProblem
            Problem definition.

        Returns
        -------
        tuple
            Cache key tuple.
        """
        residual_names = tuple(r.name for r in problem.residuals)
        residual_weights = tuple(r.weight for r in problem.residuals)

        wp_constraint_indices = tuple(
            idx for idx, _ in problem.waypoint_constraints
        )

        has_cart_rot = any(
            r.name == 'cartesian_path'
            and r.params.get('target_rotations') is not None
            for r in problem.residuals
        )

        ee_wp_key = tuple(
            (c['waypoint_index'], c['position_weight'], c['rotation_weight'])
            for c in problem.ee_waypoint_costs
        )

        # Obstacle *structure* only (counts per type), NOT their
        # positions/radii/rotations -- those are passed to the cost as
        # frozen ParamVars (see solve()/_make_world_collision_cost) so
        # the compiled problem can be reused as obstacles move. This
        # relies on the Nth sphere/cylinder in ``obstacles`` referring to
        # the same logical obstacle across calls (true for aero_demo's
        # fixed-slot human-body obstacle list, which pads missing body
        # parts with dummies instead of shrinking the list).
        obstacle_key = (0, 0)
        for r in problem.residuals:
            if r.name == 'world_collision':
                obstacles = r.params.get('obstacles', [])
                n_sphere_obs = sum(
                    1 for o in obstacles if o['type'] == 'sphere')
                n_cyl_obs = sum(
                    1 for o in obstacles if o['type'] == 'cylinder')
                obstacle_key = (n_sphere_obs, n_cyl_obs)
                break

        key = (
            problem.n_waypoints,
            problem.n_joints,
            residual_names,
            residual_weights,
            problem.fixed_start,
            problem.fixed_end,
            tuple(problem.joint_limits_lower.tolist()),
            tuple(problem.joint_limits_upper.tolist()),
            wp_constraint_indices,
            has_cart_rot,
            ee_wp_key,
            obstacle_key,  # (n_sphere_obs, n_cyl_obs) -- structure only
        )
        return key

    def solve(
        self,
        problem,
        initial_trajectory,
        **kwargs,
    ):
        """Solve trajectory optimization using jaxls.

        Parameters
        ----------
        problem : TrajectoryProblem
            Problem definition.
        initial_trajectory : ndarray
            Initial trajectory (n_waypoints, n_joints).
        **kwargs
            Additional options:
            - max_iterations: Override default max iterations.

        Returns
        -------
        SolverResult
            Optimization result.
        """
        import jax.numpy as jnp
        import jaxls

        initial_trajectory = self._validate_trajectory(
            initial_trajectory, problem
        )

        max_iterations = kwargs.get('max_iterations', self.max_iterations)
        T = problem.n_waypoints
        n_joints = problem.n_joints
        n_total_dof = getattr(problem, 'n_total_dof', n_joints)
        n_base_dof = getattr(problem, 'n_base_dof', 0)

        cache_key = self._make_cache_key(problem)

        if self._cache_key == cache_key and self._cached_problem is not None:
            ls_problem = self._cached_problem
            TrajectoryVar = self._cached_traj_var
            ConstraintParamVar = self._cached_constraint_param_var
            CartesianPosParamVar = self._cached_cartesian_pos_param_var
            CartesianRotParamVar = self._cached_cartesian_rot_param_var
            EEWpPosParamVar = self._cached_ee_wp_pos_param_var
            EEWpRotParamVar = self._cached_ee_wp_rot_param_var
            SphereObsParamVar = self._cached_sphere_obs_param_var
            CylGeomParamVar = self._cached_cyl_geom_param_var
            CylRotationParamVar = self._cached_cyl_rotation_param_var
            constraint_ids = self._cached_constraint_ids
            has_cartesian = self._cached_has_cartesian
            has_cart_rot = self._cached_has_cart_rot
            has_ee_waypoints = self._cached_has_ee_waypoints
            has_sphere_obs = self._cached_has_sphere_obs
            has_cyl_obs = self._cached_has_cyl_obs
        else:
            # Pre-scan world_collision obstacle counts so the frozen
            # ParamVar classes below can be sized correctly (each one
            # holds *all* obstacles of a type flattened into a single
            # batch-of-1 instance -- see the comment further down).
            n_sphere_obs = 0
            n_cyl_obs = 0
            for r in problem.residuals:
                if r.name == 'world_collision':
                    _obs = r.params.get('obstacles', [])
                    n_sphere_obs = sum(
                        1 for o in _obs if o['type'] == 'sphere')
                    n_cyl_obs = sum(
                        1 for o in _obs if o['type'] == 'cylinder')
                    break
            has_sphere_obs = n_sphere_obs > 0
            has_cyl_obs = n_cyl_obs > 0

            default_cfg = jnp.zeros(n_total_dof)
            default_pos = jnp.zeros(3)
            default_rot = jnp.zeros(9)

            class TrajectoryVar(
                jaxls.Var[jnp.ndarray],
                default_factory=lambda: default_cfg,
            ):
                pass

            # Frozen param vars: tangent_dim=0 means the solver never
            # updates them; retract_fn returns the original value.
            class ConstraintParamVar(
                jaxls.Var[jnp.ndarray],
                default_factory=lambda: default_cfg,
                retract_fn=lambda x, delta: x,
                tangent_dim=0,
            ):
                pass

            class CartesianPosParamVar(
                jaxls.Var[jnp.ndarray],
                default_factory=lambda: default_pos,
                retract_fn=lambda x, delta: x,
                tangent_dim=0,
            ):
                pass

            class CartesianRotParamVar(
                jaxls.Var[jnp.ndarray],
                default_factory=lambda: default_rot,
                retract_fn=lambda x, delta: x,
                tangent_dim=0,
            ):
                pass

            class EEWpPosParamVar(
                jaxls.Var[jnp.ndarray],
                default_factory=lambda: default_pos,
                retract_fn=lambda x, delta: x,
                tangent_dim=0,
            ):
                pass

            class EEWpRotParamVar(
                jaxls.Var[jnp.ndarray],
                default_factory=lambda: default_rot,
                retract_fn=lambda x, delta: x,
                tangent_dim=0,
            ):
                pass

            # World-collision obstacle geometry as a single frozen
            # ParamVar *instance* per attribute (id=[0], batch size 1),
            # holding every obstacle of that type flattened into one
            # value. jaxls broadcasts a batch-of-1 Var's id against the
            # TrajectoryVar's batch of T (see jaxls.Cost docstring: "
            # Leading axes of shape (1,) are broadcasted"), so the same
            # obstacle geometry is shared across all T waypoints instead
            # of being paired index-for-index with them (which is what
            # a batch-of-N-obstacles Var would do, and N != T in
            # general). This lets obstacles move between solves without
            # invalidating the compiled problem, as long as the obstacle
            # counts per type stay the same (see _make_cache_key).
            # Sphere: N_sphere * [cx, cy, cz, radius] flattened. Cylinder
            # geometry: N_cyl * [cx, cy, cz, radius, half_height]
            # flattened; cylinder rotations (N_cyl * flattened 3x3) are
            # kept in a separate Var, like Cartesian/EE rotations above.
            default_sphere_obs = jnp.zeros(n_sphere_obs * 4)
            default_cyl_geom = jnp.zeros(n_cyl_obs * 5)
            default_cyl_rot = jnp.zeros(n_cyl_obs * 9)

            class SphereObsParamVar(
                jaxls.Var[jnp.ndarray],
                default_factory=lambda: default_sphere_obs,
                retract_fn=lambda x, delta: x,
                tangent_dim=0,
            ):
                pass

            class CylGeomParamVar(
                jaxls.Var[jnp.ndarray],
                default_factory=lambda: default_cyl_geom,
                retract_fn=lambda x, delta: x,
                tangent_dim=0,
            ):
                pass

            class CylRotationParamVar(
                jaxls.Var[jnp.ndarray],
                default_factory=lambda: default_cyl_rot,
                retract_fn=lambda x, delta: x,
                tangent_dim=0,
            ):
                pass

            traj_vars = TrajectoryVar(jnp.arange(T))

            # Prepare FK data
            from skrobot.planner.trajectory_optimization.fk_utils import prepare_fk_data
            fk_data = prepare_fk_data(problem, jnp)

            costs = []

            has_cartesian = False
            has_cart_rot = False
            has_ee_waypoints = len(problem.ee_waypoint_costs) > 0

            for residual_spec in problem.residuals:
                if residual_spec.name == 'smoothness':
                    costs.append(self._make_smoothness_cost(
                        problem, TrajectoryVar, residual_spec
                    ))
                elif residual_spec.name == 'acceleration':
                    costs.append(self._make_acceleration_cost(
                        problem, TrajectoryVar, residual_spec
                    ))
                elif residual_spec.name == 'world_collision':
                    costs.append(self._make_world_collision_cost(
                        problem, TrajectoryVar,
                        SphereObsParamVar, CylGeomParamVar,
                        CylRotationParamVar,
                        fk_data, residual_spec,
                    ))
                elif residual_spec.name == 'self_collision':
                    costs.append(self._make_self_collision_cost(
                        problem, TrajectoryVar, fk_data, residual_spec
                    ))
                elif residual_spec.name == 'posture':
                    costs.append(self._make_posture_cost(
                        problem, TrajectoryVar, residual_spec
                    ))
                elif residual_spec.name == 'cartesian_path':
                    has_cartesian = True
                    has_cart_rot = (
                        residual_spec.params.get('target_rotations')
                        is not None
                    )
                    costs.append(self._make_cartesian_path_cost(
                        problem, TrajectoryVar,
                        CartesianPosParamVar, CartesianRotParamVar,
                        fk_data, residual_spec,
                    ))
                elif residual_spec.name == 'joint_velocity_limit':
                    costs.append(self._make_joint_velocity_limit(
                        problem, TrajectoryVar, residual_spec
                    ))
                elif residual_spec.name == 'five_point_velocity':
                    costs.append(self._make_five_point_velocity_cost(
                        problem, TrajectoryVar, residual_spec
                    ))
                elif residual_spec.name == 'five_point_acceleration':
                    costs.append(self._make_five_point_acceleration_cost(
                        problem, TrajectoryVar, residual_spec
                    ))
                elif residual_spec.name == 'five_point_jerk':
                    costs.append(self._make_five_point_jerk_cost(
                        problem, TrajectoryVar, residual_spec
                    ))
                elif residual_spec.name == 'acceleration_limit':
                    costs.append(self._make_acceleration_limit_cost(
                        problem, TrajectoryVar, residual_spec
                    ))
                elif residual_spec.name == 'jerk_limit':
                    costs.append(self._make_jerk_limit_cost(
                        problem, TrajectoryVar, residual_spec
                    ))
                elif residual_spec.name == 'com':
                    costs.append(self._make_com_cost(
                        problem, TrajectoryVar, fk_data, residual_spec
                    ))
                elif residual_spec.name == 'multi_ee_waypoint':
                    costs.append(self._make_multi_ee_waypoint_cost(
                        problem, TrajectoryVar, fk_data, residual_spec
                    ))
                elif residual_spec.name == 'base_pose':
                    costs.append(self._make_base_pose_cost(
                        problem, TrajectoryVar, residual_spec
                    ))

            # --- EE waypoint costs ---
            if has_ee_waypoints:
                costs.append(self._make_ee_waypoint_costs(
                    problem, TrajectoryVar,
                    EEWpPosParamVar, EEWpRotParamVar,
                    fk_data,
                ))

            # --- Constraint targets as frozen ParamVars ---
            constraint_ids = {}
            next_ct_id = 0

            if problem.fixed_start:
                constraint_ids['start'] = next_ct_id

                @jaxls.Cost.factory(
                    kind='constraint_eq_zero',
                    name='start_constraint',
                )
                def start_constraint(vals, var, param):
                    return (vals[var] - vals[param]).flatten()

                costs.append(start_constraint(
                    TrajectoryVar(jnp.array([0])),
                    ConstraintParamVar(jnp.array([next_ct_id])),
                ))
                next_ct_id += 1

            if problem.fixed_end:
                constraint_ids['end'] = next_ct_id

                @jaxls.Cost.factory(
                    kind='constraint_eq_zero',
                    name='end_constraint',
                )
                def end_constraint(vals, var, param):
                    return (vals[var] - vals[param]).flatten()

                costs.append(end_constraint(
                    TrajectoryVar(jnp.array([T - 1])),
                    ConstraintParamVar(jnp.array([next_ct_id])),
                ))
                next_ct_id += 1

            wp_ct_ids = {}
            for wp_idx, _wp_angles in problem.waypoint_constraints:
                wp_ct_ids[wp_idx] = next_ct_id

                @jaxls.Cost.factory(
                    kind='constraint_eq_zero',
                    name='waypoint_constraint',
                )
                def waypoint_constraint(vals, var, param):
                    return (vals[var] - vals[param]).flatten()

                costs.append(waypoint_constraint(
                    TrajectoryVar(jnp.array([wp_idx])),
                    ConstraintParamVar(jnp.array([next_ct_id])),
                ))
                next_ct_id += 1

            constraint_ids['waypoints'] = wp_ct_ids
            constraint_ids['n_params'] = next_ct_id

            # Joint limits (extend with -inf/+inf for base DoF so the
            # base translation/rotation are unconstrained).
            if n_base_dof > 0:
                lower_full = np.concatenate([
                    problem.joint_limits_lower,
                    np.full(n_base_dof, -1e6, dtype=np.float64),
                ])
                upper_full = np.concatenate([
                    problem.joint_limits_upper,
                    np.full(n_base_dof, +1e6, dtype=np.float64),
                ])
            else:
                lower_full = problem.joint_limits_lower
                upper_full = problem.joint_limits_upper
            lower = jnp.array(lower_full)
            upper = jnp.array(upper_full)

            @jaxls.Cost.factory(
                kind='constraint_geq_zero', name='joint_limits',
            )
            def joint_limit_cost(vals, var):
                q = vals[var]
                lower_margin = q - lower
                upper_margin = upper - q
                return jnp.concatenate(
                    [lower_margin, upper_margin]
                ).flatten()

            costs.append(joint_limit_cost(traj_vars))

            # Build variable list (ParamVars included but frozen)
            all_variables = [traj_vars]
            if next_ct_id > 0:
                all_variables.append(
                    ConstraintParamVar(jnp.arange(next_ct_id))
                )
            if has_cartesian:
                all_variables.append(
                    CartesianPosParamVar(jnp.arange(T))
                )
                if has_cart_rot:
                    all_variables.append(
                        CartesianRotParamVar(jnp.arange(T))
                    )
            if has_ee_waypoints:
                n_ee_wps = len(problem.ee_waypoint_costs)
                all_variables.append(
                    EEWpPosParamVar(jnp.arange(n_ee_wps))
                )
                all_variables.append(
                    EEWpRotParamVar(jnp.arange(n_ee_wps))
                )
            if has_sphere_obs:
                all_variables.append(SphereObsParamVar(jnp.array([0])))
            if has_cyl_obs:
                all_variables.append(CylGeomParamVar(jnp.array([0])))
                all_variables.append(CylRotationParamVar(jnp.array([0])))

            # CoG cost ParamVars (one per add_com_cost call). The cost
            # factory stashes them on ``problem._com_param_vars``.
            for entry in getattr(problem, '_com_param_vars', []):
                vc = entry['var_class']
                n = entry['targets_sel'].shape[0]
                all_variables.append(vc(jnp.arange(n)))
            # Multi-EE waypoint cost ParamVars.
            for entry in getattr(problem, '_multi_ee_param_vars', []):
                pos_vc = entry['pos_var_class']
                all_variables.append(pos_vc(jnp.arange(T)))
                if entry['rot_var_class'] is not None:
                    all_variables.append(
                        entry['rot_var_class'](jnp.arange(T)))
            # Base-pose cost ParamVars.
            for entry in getattr(problem, '_base_pose_param_vars', []):
                pos_vc = entry['pos_var_class']
                all_variables.append(pos_vc(jnp.arange(T)))
                if entry['rot_var_class'] is not None:
                    all_variables.append(
                        entry['rot_var_class'](jnp.arange(T)))

            ls_problem = jaxls.LeastSquaresProblem(
                costs=costs,
                variables=all_variables,
            ).analyze()

            self._cached_problem = ls_problem
            self._cached_traj_var = TrajectoryVar
            self._cached_constraint_param_var = ConstraintParamVar
            self._cached_cartesian_pos_param_var = CartesianPosParamVar
            self._cached_cartesian_rot_param_var = CartesianRotParamVar
            self._cached_ee_wp_pos_param_var = EEWpPosParamVar
            self._cached_ee_wp_rot_param_var = EEWpRotParamVar
            self._cached_sphere_obs_param_var = SphereObsParamVar
            self._cached_cyl_geom_param_var = CylGeomParamVar
            self._cached_cyl_rotation_param_var = CylRotationParamVar
            self._cached_constraint_ids = constraint_ids
            self._cached_has_cartesian = has_cartesian
            self._cached_has_cart_rot = has_cart_rot
            self._cached_has_ee_waypoints = has_ee_waypoints
            self._cached_has_sphere_obs = has_sphere_obs
            self._cached_has_cyl_obs = has_cyl_obs
            self._cache_key = cache_key

        # --- Build init_vals with current dynamic values ---
        traj_vars = TrajectoryVar(jnp.arange(T))
        init_pairs = [
            traj_vars.with_value(jnp.array(initial_trajectory)),
        ]

        # Constraint param values
        n_ct = constraint_ids['n_params']
        if n_ct > 0:
            ct_values = np.zeros((n_ct, n_total_dof))
            if 'start' in constraint_ids:
                ct_values[constraint_ids['start']] = initial_trajectory[0]
            if 'end' in constraint_ids:
                ct_values[constraint_ids['end']] = initial_trajectory[-1]
            for wp_idx, wp_angles in problem.waypoint_constraints:
                ct_id = constraint_ids['waypoints'][wp_idx]
                wp_full = np.zeros(n_total_dof)
                wp_full[:len(wp_angles)] = wp_angles
                ct_values[ct_id] = wp_full
            init_pairs.append(
                ConstraintParamVar(jnp.arange(n_ct)).with_value(
                    jnp.array(ct_values)
                )
            )

        # EE waypoint param values
        if has_ee_waypoints:
            ee_wps = problem.ee_waypoint_costs
            n_ee_wps = len(ee_wps)
            ee_pos_values = np.stack(
                [c['target_position'] for c in ee_wps])
            ee_rot_values = np.stack(
                [c['target_rotation'].flatten() for c in ee_wps])
            init_pairs.append(
                EEWpPosParamVar(jnp.arange(n_ee_wps)).with_value(
                    jnp.array(ee_pos_values)
                )
            )
            init_pairs.append(
                EEWpRotParamVar(jnp.arange(n_ee_wps)).with_value(
                    jnp.array(ee_rot_values)
                )
            )

        # World-collision obstacle ParamVar values (read fresh from the
        # incoming ``problem`` every call -- even on a cache hit, this
        # ``problem`` is a new instance whose obstacles may have moved).
        if has_sphere_obs or has_cyl_obs:
            for r in problem.residuals:
                if r.name != 'world_collision':
                    continue
                obstacles = r.params.get('obstacles', [])
                if has_sphere_obs:
                    sphere_obs = [
                        o for o in obstacles if o['type'] == 'sphere']
                    sphere_vals = np.array([
                        list(o['center']) + [o['radius']]
                        for o in sphere_obs
                    ])
                    init_pairs.append(
                        SphereObsParamVar(
                            jnp.arange(len(sphere_obs))
                        ).with_value(jnp.array(sphere_vals))
                    )
                if has_cyl_obs:
                    cylinder_obs = [
                        o for o in obstacles if o['type'] == 'cylinder']
                    cyl_geom_vals = np.array([
                        list(o['center']) + [o['radius'], o['half_height']]
                        for o in cylinder_obs
                    ])
                    cyl_rot_vals = np.array([
                        np.asarray(o['rotation']).flatten()
                        for o in cylinder_obs
                    ])
                    init_pairs.append(
                        CylGeomParamVar(
                            jnp.arange(len(cylinder_obs))
                        ).with_value(jnp.array(cyl_geom_vals))
                    )
                    init_pairs.append(
                        CylRotationParamVar(
                            jnp.arange(len(cylinder_obs))
                        ).with_value(jnp.array(cyl_rot_vals))
                    )
                break

        # Cartesian param values
        if has_cartesian:
            for r in problem.residuals:
                if r.name == 'cartesian_path':
                    init_pairs.append(
                        CartesianPosParamVar(jnp.arange(T)).with_value(
                            jnp.array(r.params['target_positions'])
                        )
                    )
                    if has_cart_rot:
                        rot_flat = np.array(
                            r.params['target_rotations']
                        ).reshape(T, 9)
                        init_pairs.append(
                            CartesianRotParamVar(jnp.arange(T)).with_value(
                                jnp.array(rot_flat)
                            )
                        )
                    break

        # CoG ParamVar values: write the per-waypoint targets stashed
        # by ``_make_com_cost`` so the closure-frozen ParamVars carry
        # the right numbers when the JIT plan executes.
        for entry in getattr(problem, '_com_param_vars', []):
            vc = entry['var_class']
            n = entry['targets_sel'].shape[0]
            init_pairs.append(
                vc(jnp.arange(n)).with_value(entry['targets_sel'])
            )
        # Multi-EE waypoint ParamVar values.
        for entry in getattr(problem, '_multi_ee_param_vars', []):
            pos_vc = entry['pos_var_class']
            init_pairs.append(
                pos_vc(jnp.arange(T)).with_value(entry['target_pos'])
            )
            if entry['rot_var_class'] is not None:
                init_pairs.append(
                    entry['rot_var_class'](jnp.arange(T)).with_value(
                        entry['target_rot'])
                )
        # Base-pose ParamVar values.
        for entry in getattr(problem, '_base_pose_param_vars', []):
            pos_vc = entry['pos_var_class']
            init_pairs.append(
                pos_vc(jnp.arange(T)).with_value(entry['target_pos'])
            )
            if entry['rot_var_class'] is not None:
                init_pairs.append(
                    entry['rot_var_class'](jnp.arange(T)).with_value(
                        entry['target_rot'])
                )

        init_vals = jaxls.VarValues.make(tuple(init_pairs))

        solution = ls_problem.solve(
            initial_vals=init_vals,
            verbose=self.verbose,
            termination=jaxls.TerminationConfig(
                max_iterations=int(max_iterations),
            ),
        )

        result_traj = np.array(solution[traj_vars])

        return SolverResult(
            trajectory=result_traj,
            success=True,
            iterations=max_iterations,
            message='Optimization completed',
        )

    def _make_smoothness_cost(self, problem, TrajectoryVar, spec):
        """Create smoothness cost."""
        import jax.numpy as jnp
        import jaxls

        T = problem.n_waypoints
        weight = jnp.sqrt(spec.weight)

        @jaxls.Cost.factory(name='smoothness')
        def smoothness_cost(vals, curr_var, prev_var):
            q_curr = vals[curr_var]
            q_prev = vals[prev_var]
            return weight * (q_curr - q_prev).flatten()

        return smoothness_cost(
            TrajectoryVar(jnp.arange(1, T)),
            TrajectoryVar(jnp.arange(0, T - 1)),
        )

    def _make_acceleration_cost(self, problem, TrajectoryVar, spec):
        """Create acceleration cost."""
        import jax.numpy as jnp
        import jaxls

        T = problem.n_waypoints
        dt = spec.params['dt']
        weight = jnp.sqrt(spec.weight)

        @jaxls.Cost.factory(name='acceleration')
        def acceleration_cost(vals, curr_var, next_var, prev_var):
            q_curr = vals[curr_var]
            q_next = vals[next_var]
            q_prev = vals[prev_var]
            acc = (q_next - 2 * q_curr + q_prev) / (dt ** 2)
            return weight * acc.flatten()

        return acceleration_cost(
            TrajectoryVar(jnp.arange(1, T - 1)),
            TrajectoryVar(jnp.arange(2, T)),
            TrajectoryVar(jnp.arange(0, T - 2)),
        )

    def _make_posture_cost(self, problem, TrajectoryVar, spec):
        """Create posture regularization cost.

        Penalizes deviation from nominal joint angles at each waypoint.
        """
        import jax.numpy as jnp
        import jaxls

        T = problem.n_waypoints
        nominal = jnp.array(spec.params['nominal_angles'])
        weight = jnp.sqrt(spec.weight)

        @jaxls.Cost.factory(name='posture')
        def posture_cost(vals, var):
            q = vals[var]
            diff = q - nominal
            return (weight * diff).flatten()

        return posture_cost(TrajectoryVar(jnp.arange(T)))

    def _make_ee_waypoint_costs(
        self, problem, TrajectoryVar,
        EEWpPosParamVar, EEWpRotParamVar,
        fk_data,
    ):
        """Create end-effector waypoint tracking costs.

        Constrains end-effector pose at specific trajectory waypoints
        without fixing joint angles, leaving the optimizer free to find
        natural joint configurations.  Targets are stored in frozen
        ParamVar objects for JIT cache reuse.
        """
        import jax.numpy as jnp
        import jaxls

        from skrobot.planner.trajectory_optimization.fk_utils import build_fk_functions
        from skrobot.planner.trajectory_optimization.fk_utils import pose_error_log

        _, _, _, get_ee_pose = build_fk_functions(fk_data, jnp)

        ee_wps = problem.ee_waypoint_costs
        n_ee_wps = len(ee_wps)
        indices = jnp.array([c['waypoint_index'] for c in ee_wps])

        # Use uniform weight (from first EE waypoint); weights are
        # part of the cache key so changing them rebuilds the problem.
        pos_weight = jnp.sqrt(ee_wps[0]['position_weight'])
        rot_weight = jnp.sqrt(ee_wps[0]['rotation_weight'])

        @jaxls.Cost.factory(name='ee_waypoint')
        def ee_waypoint_cost(vals, var, pos_param, rot_param):
            angles = vals[var]
            ee_pos, ee_rot = get_ee_pose(angles)
            target_pos = vals[pos_param]
            target_rot = vals[rot_param].reshape(3, 3)
            # Use SE(3) logarithmic map for pose error
            pose_err = pose_error_log(ee_pos, ee_rot, target_pos, target_rot)
            # pose_err is (6,): [tx, ty, tz, rx, ry, rz]
            pos_err = pos_weight * pose_err[:3]
            rot_err = rot_weight * pose_err[3:]
            return jnp.concatenate([pos_err, rot_err]).flatten()

        return ee_waypoint_cost(
            TrajectoryVar(indices),
            EEWpPosParamVar(jnp.arange(n_ee_wps)),
            EEWpRotParamVar(jnp.arange(n_ee_wps)),
        )

    def _make_world_collision_cost(
        self, problem, TrajectoryVar,
        SphereObsParamVar, CylGeomParamVar, CylRotationParamVar,
        fk_data, spec,
    ):
        """Create world collision avoidance cost.

        Supports two obstacle types (``obstacle['type']``): ``'sphere'``
        and ``'cylinder'`` (finite flat-capped cylinder -- matches
        ``skrobot.model.primitives.Cylinder``, the shape
        ``aero_demo.solve_palm_ik.human_body_obstacles`` builds around a
        skeleton, so callers can pass that geometry directly instead of
        approximating it with a handful of spheres).

        The robot side is represented by its exact box/cylinder/sphere
        collision primitives (``fk_data['collision_primitives']``, built
        by ``TrajectoryProblem._compute_collision_primitives`` from
        ``collision.extract_collision_primitives`` -- one primitive per
        link, matching ``aero_demo.solve_palm_ik.apply_collision_model``,
        not the old fixed-N-spheres-per-link bounding-capsule
        approximation). Distances against a robot box/cylinder use
        :func:`~...fk_utils.primitive_pair_signed_distance`'s alternating
        projection; against a robot sphere they reduce to the exact
        point-to-primitive formula (see that function's docstring for
        the accuracy caveat this implies for box/cylinder-vs-cylinder
        pairs under deep penetration).

        Obstacle geometry (center/radius/rotation/half_height) is read
        from frozen ``tangent_dim=0`` ParamVars (``SphereObsParamVar``,
        ``CylGeomParamVar``, ``CylRotationParamVar``) instead of being
        baked into the cost closure as Python constants, so the compiled
        ``ls_problem`` can be reused across solves where only the
        obstacles move -- as long as the obstacle *counts* per type stay
        the same (see ``_make_cache_key``). This assumes the Nth sphere/
        cylinder in ``obstacles`` refers to the same logical obstacle
        across calls (true for aero_demo's fixed-slot human-body
        obstacle list; see
        ``plan_handshake_motion.human_body_cylinder_obstacles``). The
        robot-side primitive geometry is *not* parametric -- it is fixed
        for a given robot/collision_link_list, so it is baked into the
        cost closure like ``sphere_radii`` was before.
        """
        import jax.numpy as jnp
        import jaxls

        from skrobot.planner.trajectory_optimization.fk_utils import build_fk_functions
        from skrobot.planner.trajectory_optimization.fk_utils import compute_collision_residuals
        from skrobot.planner.trajectory_optimization.fk_utils import get_primitive_world_pose
        from skrobot.planner.trajectory_optimization.fk_utils import primitive_pair_signed_distance

        T = problem.n_waypoints
        obstacles = spec.params['obstacles']
        activation_dist = spec.params['activation_distance']
        weight = jnp.sqrt(spec.weight)

        # Parse obstacles
        sphere_obs = [obs for obs in obstacles if obs['type'] == 'sphere']
        cylinder_obs = [obs for obs in obstacles if obs['type'] == 'cylinder']
        robot_buckets = fk_data.get('collision_primitives', {})

        if (not sphere_obs and not cylinder_obs) or not robot_buckets:
            # No obstacles, or no robot-side collision geometry at all.
            @jaxls.Cost.factory(name='world_collision_dummy')
            def dummy_cost(vals, var):
                return jnp.array([0.0])

            return dummy_cost(TrajectoryVar(jnp.array([0])))

        n_sphere = len(sphere_obs)
        n_cyl = len(cylinder_obs)
        get_link_transforms, _, _, _ = build_fk_functions(fk_data, jnp)

        def _expand(prim, axis):
            # Add a broadcasting axis to every array value (all but
            # 'type') so a (n_a,) batch of primitives can be compared
            # against a (n_b,) batch cross-product-style, without the
            # (n_a, n_b) grid having to be a Python-level primitive type.
            return {k: (v if k == 'type' else jnp.expand_dims(v, axis))
                    for k, v in prim.items()}

        def _robot_primitive(ptype, link_positions, link_rotations):
            bucket = robot_buckets[ptype]
            center, rotation = get_primitive_world_pose(
                bucket, link_positions, link_rotations, jnp)
            prim = {'type': ptype, 'center': center}
            if rotation is not None:
                prim['rotation'] = rotation
            if ptype == 'box':
                prim['half_extents'] = bucket['half_extents']
            elif ptype == 'cylinder':
                prim['radius'] = bucket['radius']
                prim['half_height'] = bucket['half_height']
            else:
                prim['radius'] = bucket['radius']
            return prim

        def _sphere_obstacle_primitive(sphere_geom):
            return {'type': 'sphere', 'center': sphere_geom[:, :3],
                   'radius': sphere_geom[:, 3]}

        def _cylinder_obstacle_primitive(cyl_geom, cyl_rotation_flat):
            return {
                'type': 'cylinder',
                'center': cyl_geom[:, :3],
                'radius': cyl_geom[:, 3],
                'half_height': cyl_geom[:, 4],
                'rotation': cyl_rotation_flat.reshape(n_cyl, 3, 3),
            }

        def _residual_for(robot_prim, obstacle_prim):
            dists = primitive_pair_signed_distance(
                _expand(robot_prim, 1), _expand(obstacle_prim, 0), jnp)
            return compute_collision_residuals(
                dists, activation_dist, jnp).flatten()

        def _all_residuals(link_positions, link_rotations, obstacle_prims):
            parts = [
                _residual_for(
                    _robot_primitive(ptype, link_positions, link_rotations),
                    obstacle_prim)
                for ptype in robot_buckets
                for obstacle_prim in obstacle_prims
            ]
            return jnp.concatenate(parts)

        if sphere_obs and cylinder_obs:
            @jaxls.Cost.factory(name='world_collision')
            def world_collision_cost(
                vals, var, sphere_param, cyl_geom_param, cyl_rot_param,
            ):
                link_positions, link_rotations = get_link_transforms(
                    vals[var])
                obstacle_prims = [
                    _sphere_obstacle_primitive(vals[sphere_param]),
                    _cylinder_obstacle_primitive(
                        vals[cyl_geom_param], vals[cyl_rot_param]),
                ]
                return weight * _all_residuals(
                    link_positions, link_rotations, obstacle_prims)

            return world_collision_cost(
                TrajectoryVar(jnp.arange(T)),
                SphereObsParamVar(jnp.arange(n_sphere)),
                CylGeomParamVar(jnp.arange(n_cyl)),
                CylRotationParamVar(jnp.arange(n_cyl)),
            )
        elif sphere_obs:
            @jaxls.Cost.factory(name='world_collision')
            def world_collision_cost(vals, var, sphere_param):
                link_positions, link_rotations = get_link_transforms(
                    vals[var])
                obstacle_prims = [_sphere_obstacle_primitive(
                    vals[sphere_param])]
                return weight * _all_residuals(
                    link_positions, link_rotations, obstacle_prims)

            return world_collision_cost(
                TrajectoryVar(jnp.arange(T)),
                SphereObsParamVar(jnp.arange(n_sphere)),
            )
        else:
            @jaxls.Cost.factory(name='world_collision')
            def world_collision_cost(
                vals, var, cyl_geom_param, cyl_rot_param,
            ):
                link_positions, link_rotations = get_link_transforms(
                    vals[var])
                obstacle_prims = [_cylinder_obstacle_primitive(
                    vals[cyl_geom_param], vals[cyl_rot_param])]
                return weight * _all_residuals(
                    link_positions, link_rotations, obstacle_prims)

            return world_collision_cost(
                TrajectoryVar(jnp.arange(T)),
                CylGeomParamVar(jnp.arange(n_cyl)),
                CylRotationParamVar(jnp.arange(n_cyl)),
            )

    def _make_self_collision_cost(self, problem, TrajectoryVar, fk_data, spec):
        """Create self-collision avoidance cost.

        Robot-side geometry is the same exact box/cylinder/sphere
        collision primitives used by ``_make_world_collision_cost`` (see
        its docstring), one per link. Self-collision pairs are grouped
        by (type_a, type_b) primitive combination
        (``fk_data['self_collision_primitive_pairs']``, built by
        ``TrajectoryProblem._compute_self_collision_primitive_pairs``)
        so each group can be evaluated as one batched, aligned (not
        cross-product) call to
        :func:`~...fk_utils.primitive_pair_signed_distance` -- row *k*
        of group (type_a, type_b) is the distance for one specific
        self-collision link pair, unlike the cross-product used against
        world obstacles.
        """
        import jax.numpy as jnp
        import jaxls

        from skrobot.planner.trajectory_optimization.fk_utils import build_fk_functions
        from skrobot.planner.trajectory_optimization.fk_utils import compute_collision_residuals
        from skrobot.planner.trajectory_optimization.fk_utils import get_primitive_world_pose
        from skrobot.planner.trajectory_optimization.fk_utils import primitive_pair_signed_distance

        T = problem.n_waypoints
        activation_dist = spec.params['activation_distance']
        weight = jnp.sqrt(spec.weight)

        robot_buckets = fk_data.get('collision_primitives', {})
        pair_groups = {
            combo: rows for combo, rows in
            fk_data.get('self_collision_primitive_pairs', {}).items()
            if rows['rows_a'].shape[0] > 0
        }

        if not pair_groups:
            @jaxls.Cost.factory(name='self_collision_dummy')
            def dummy_cost(vals, var):
                return jnp.array([0.0])

            return dummy_cost(TrajectoryVar(jnp.array([0])))

        get_link_transforms, _, _, _ = build_fk_functions(fk_data, jnp)

        def _primitive_from_rows(ptype, rows, link_positions, link_rotations):
            sub = {k: v[rows] for k, v in robot_buckets[ptype].items()}
            center, rotation = get_primitive_world_pose(
                sub, link_positions, link_rotations, jnp)
            prim = {'type': ptype, 'center': center}
            if rotation is not None:
                prim['rotation'] = rotation
            if ptype == 'box':
                prim['half_extents'] = sub['half_extents']
            elif ptype == 'cylinder':
                prim['radius'] = sub['radius']
                prim['half_height'] = sub['half_height']
            else:
                prim['radius'] = sub['radius']
            return prim

        @jaxls.Cost.factory(name='self_collision')
        def self_collision_cost(vals, var):
            angles = vals[var]
            link_positions, link_rotations = get_link_transforms(angles)

            parts = []
            for (type_a, type_b), rows in pair_groups.items():
                prim_a = _primitive_from_rows(
                    type_a, rows['rows_a'], link_positions, link_rotations)
                prim_b = _primitive_from_rows(
                    type_b, rows['rows_b'], link_positions, link_rotations)
                signed_dists = primitive_pair_signed_distance(
                    prim_a, prim_b, jnp)
                parts.append(compute_collision_residuals(
                    signed_dists, activation_dist, jnp).flatten())
            return weight * jnp.concatenate(parts)

        return self_collision_cost(TrajectoryVar(jnp.arange(T)))

    def _make_cartesian_path_cost(
        self, problem, TrajectoryVar,
        CartesianPosParamVar, CartesianRotParamVar,
        fk_data, spec,
    ):
        """Create Cartesian path tracking cost with parametric targets.

        Targets are read from frozen ParamVar objects so that the
        compiled problem can be reused when targets change.
        """
        import jax.numpy as jnp
        import jaxls

        from skrobot.planner.trajectory_optimization.fk_utils import build_fk_functions
        from skrobot.planner.trajectory_optimization.fk_utils import pose_error_log

        T = problem.n_waypoints
        rotation_weight = spec.params.get('rotation_weight', 1.0)
        pos_weight = jnp.sqrt(spec.weight)

        _, _, _, get_ee_pose = build_fk_functions(fk_data, jnp)

        has_rot = spec.params.get('target_rotations') is not None

        if has_rot:
            rot_weight = jnp.sqrt(spec.weight * rotation_weight)

            @jaxls.Cost.factory(name='cartesian_path')
            def cartesian_path_cost(vals, var, pos_param, rot_param):
                angles = vals[var]
                ee_pos, ee_rot = get_ee_pose(angles)
                target_pos = vals[pos_param]
                target_rot = vals[rot_param].reshape(3, 3)
                # Use SE(3) logarithmic map for pose error
                pose_err = pose_error_log(ee_pos, ee_rot, target_pos, target_rot)
                pos_err = pos_weight * pose_err[:3]
                rot_err = rot_weight * pose_err[3:]
                return jnp.concatenate([pos_err, rot_err]).flatten()

            return cartesian_path_cost(
                TrajectoryVar(jnp.arange(T)),
                CartesianPosParamVar(jnp.arange(T)),
                CartesianRotParamVar(jnp.arange(T)),
            )
        else:
            @jaxls.Cost.factory(name='cartesian_path')
            def cartesian_path_cost(vals, var, pos_param):
                angles = vals[var]
                ee_pos, _ = get_ee_pose(angles)
                target_pos = vals[pos_param]
                return (pos_weight * (ee_pos - target_pos)).flatten()

            return cartesian_path_cost(
                TrajectoryVar(jnp.arange(T)),
                CartesianPosParamVar(jnp.arange(T)),
            )

    def _make_joint_velocity_limit(self, problem, TrajectoryVar, spec):
        """Create joint velocity limit constraint.

        Enforces ``|q[t+1] - q[t]| / dt <= v_max`` for each joint,
        expressed as two ``geq_zero`` inequalities per step:

            v_max * dt - (q_next - q_prev)  >= 0
            v_max * dt + (q_next - q_prev)  >= 0
        """
        import jax.numpy as jnp
        import jaxls

        T = problem.n_waypoints
        dt = spec.params['dt']
        max_velocities = jnp.array(spec.params['max_velocities'])
        v_limit = max_velocities * dt  # max allowable delta per step

        @jaxls.Cost.factory(
            kind='constraint_geq_zero',
            name='joint_velocity_limit',
        )
        def velocity_limit(vals, curr_var, prev_var):
            dq = vals[curr_var] - vals[prev_var]
            upper_margin = v_limit - dq   # v_limit - dq >= 0
            lower_margin = v_limit + dq   # v_limit + dq >= 0
            return jnp.concatenate([upper_margin, lower_margin]).flatten()

        return velocity_limit(
            TrajectoryVar(jnp.arange(1, T)),
            TrajectoryVar(jnp.arange(0, T - 1)),
        )

    def _make_five_point_velocity_cost(self, problem, TrajectoryVar, spec):
        """Create velocity cost using 5-point stencil.

        Computes velocity with O(h^4) accuracy:
            v = (-q[t+2] + 8*q[t+1] - 8*q[t-1] + q[t-2]) / (12*dt)

        Penalizes velocities that exceed the velocity limits.
        """
        import jax.numpy as jnp
        import jaxls

        T = problem.n_waypoints
        dt = spec.params['dt']
        velocity_limits = jnp.array(spec.params['velocity_limits'])
        weight = jnp.sqrt(spec.weight)

        @jaxls.Cost.factory(name='five_point_velocity')
        def five_point_velocity_cost(
            vals, var_tp2, var_tp1, var_tm1, var_tm2
        ):
            q_tp2 = vals[var_tp2]
            q_tp1 = vals[var_tp1]
            q_tm1 = vals[var_tm1]
            q_tm2 = vals[var_tm2]

            velocity = (-q_tp2 + 8 * q_tp1 - 8 * q_tm1 + q_tm2) / (12 * dt)
            # Penalize only when |velocity| > limit
            residual = jnp.maximum(0.0, jnp.abs(velocity) - velocity_limits)
            return (weight * residual).flatten()

        # Apply to waypoints [2, T-2] (need 2 points on each side)
        return five_point_velocity_cost(
            TrajectoryVar(jnp.arange(4, T)),      # t+2
            TrajectoryVar(jnp.arange(3, T - 1)),  # t+1
            TrajectoryVar(jnp.arange(1, T - 3)),  # t-1
            TrajectoryVar(jnp.arange(0, T - 4)),  # t-2
        )

    def _make_five_point_acceleration_cost(self, problem, TrajectoryVar, spec):
        """Create acceleration cost using 5-point stencil.

        Computes acceleration with O(h^4) accuracy:
            a = (-q[t+2] + 16*q[t+1] - 30*q[t] + 16*q[t-1] - q[t-2]) / (12*dt^2)
        """
        import jax.numpy as jnp
        import jaxls

        T = problem.n_waypoints
        dt = spec.params['dt']
        weight = jnp.sqrt(spec.weight)

        @jaxls.Cost.factory(name='five_point_acceleration')
        def five_point_acceleration_cost(
            vals, var_t, var_tp2, var_tp1, var_tm1, var_tm2
        ):
            q_t = vals[var_t]
            q_tp2 = vals[var_tp2]
            q_tp1 = vals[var_tp1]
            q_tm1 = vals[var_tm1]
            q_tm2 = vals[var_tm2]

            acceleration = (
                -q_tp2 + 16 * q_tp1 - 30 * q_t + 16 * q_tm1 - q_tm2
            ) / (12 * dt ** 2)
            return (weight * jnp.abs(acceleration)).flatten()

        # Apply to waypoints [2, T-2]
        return five_point_acceleration_cost(
            TrajectoryVar(jnp.arange(2, T - 2)),  # t
            TrajectoryVar(jnp.arange(4, T)),      # t+2
            TrajectoryVar(jnp.arange(3, T - 1)),  # t+1
            TrajectoryVar(jnp.arange(1, T - 3)),  # t-1
            TrajectoryVar(jnp.arange(0, T - 4)),  # t-2
        )

    def _make_five_point_jerk_cost(self, problem, TrajectoryVar, spec):
        """Create jerk cost using 7-point stencil.

        Computes jerk with O(h^4) accuracy:
            j = (-q[t+3] + 8*q[t+2] - 13*q[t+1] + 13*q[t-1] - 8*q[t-2] + q[t-3])
                / (8*dt^3)
        """
        import jax.numpy as jnp
        import jaxls

        T = problem.n_waypoints
        dt = spec.params['dt']
        weight = jnp.sqrt(spec.weight)

        @jaxls.Cost.factory(name='five_point_jerk')
        def five_point_jerk_cost(
            vals, var_tp3, var_tp2, var_tp1, var_tm1, var_tm2, var_tm3
        ):
            q_tp3 = vals[var_tp3]
            q_tp2 = vals[var_tp2]
            q_tp1 = vals[var_tp1]
            q_tm1 = vals[var_tm1]
            q_tm2 = vals[var_tm2]
            q_tm3 = vals[var_tm3]

            jerk = (
                -q_tp3 + 8 * q_tp2 - 13 * q_tp1 + 13 * q_tm1 - 8 * q_tm2 + q_tm3
            ) / (8 * dt ** 3)
            return (weight * jnp.abs(jerk)).flatten()

        # Apply to waypoints [3, T-3]
        return five_point_jerk_cost(
            TrajectoryVar(jnp.arange(6, T)),      # t+3
            TrajectoryVar(jnp.arange(5, T - 1)),  # t+2
            TrajectoryVar(jnp.arange(4, T - 2)),  # t+1
            TrajectoryVar(jnp.arange(2, T - 4)),  # t-1
            TrajectoryVar(jnp.arange(1, T - 5)),  # t-2
            TrajectoryVar(jnp.arange(0, T - 6)),  # t-3
        )

    def _make_acceleration_limit_cost(self, problem, TrajectoryVar, spec):
        """Create acceleration limit cost using 5-point stencil.

        Penalizes accelerations that exceed the specified limit.
        """
        import jax.numpy as jnp
        import jaxls

        T = problem.n_waypoints
        dt = spec.params['dt']
        acceleration_limit = jnp.array(spec.params['acceleration_limit'])
        weight = jnp.sqrt(spec.weight)

        @jaxls.Cost.factory(name='acceleration_limit')
        def acceleration_limit_cost(
            vals, var_t, var_tp2, var_tp1, var_tm1, var_tm2
        ):
            q_t = vals[var_t]
            q_tp2 = vals[var_tp2]
            q_tp1 = vals[var_tp1]
            q_tm1 = vals[var_tm1]
            q_tm2 = vals[var_tm2]

            acceleration = (
                -q_tp2 + 16 * q_tp1 - 30 * q_t + 16 * q_tm1 - q_tm2
            ) / (12 * dt ** 2)
            # Penalize only when |acceleration| > limit
            residual = jnp.maximum(
                0.0, jnp.abs(acceleration) - acceleration_limit
            )
            return (weight * residual).flatten()

        # Apply to waypoints [2, T-2]
        return acceleration_limit_cost(
            TrajectoryVar(jnp.arange(2, T - 2)),  # t
            TrajectoryVar(jnp.arange(4, T)),      # t+2
            TrajectoryVar(jnp.arange(3, T - 1)),  # t+1
            TrajectoryVar(jnp.arange(1, T - 3)),  # t-1
            TrajectoryVar(jnp.arange(0, T - 4)),  # t-2
        )

    def _make_jerk_limit_cost(self, problem, TrajectoryVar, spec):
        """Create jerk limit cost using 7-point stencil.

        Penalizes jerks that exceed the specified limit.
        """
        import jax.numpy as jnp
        import jaxls

        T = problem.n_waypoints
        dt = spec.params['dt']
        jerk_limit = jnp.array(spec.params['jerk_limit'])
        weight = jnp.sqrt(spec.weight)

        @jaxls.Cost.factory(name='jerk_limit')
        def jerk_limit_cost(
            vals, var_tp3, var_tp2, var_tp1, var_tm1, var_tm2, var_tm3
        ):
            q_tp3 = vals[var_tp3]
            q_tp2 = vals[var_tp2]
            q_tp1 = vals[var_tp1]
            q_tm1 = vals[var_tm1]
            q_tm2 = vals[var_tm2]
            q_tm3 = vals[var_tm3]

            jerk = (
                -q_tp3 + 8 * q_tp2 - 13 * q_tp1 + 13 * q_tm1 - 8 * q_tm2 + q_tm3
            ) / (8 * dt ** 3)
            # Penalize only when |jerk| > limit
            residual = jnp.maximum(0.0, jnp.abs(jerk) - jerk_limit)
            return (weight * residual).flatten()

        # Apply to waypoints [3, T-3]
        return jerk_limit_cost(
            TrajectoryVar(jnp.arange(6, T)),      # t+3
            TrajectoryVar(jnp.arange(5, T - 1)),  # t+2
            TrajectoryVar(jnp.arange(4, T - 2)),  # t+1
            TrajectoryVar(jnp.arange(2, T - 4)),  # t-1
            TrajectoryVar(jnp.arange(1, T - 5)),  # t-2
            TrajectoryVar(jnp.arange(0, T - 6)),  # t-3
        )

    def _make_com_cost(self, problem, TrajectoryVar, fk_data, spec):
        """Per-waypoint centre-of-gravity tracking cost.

        Splits the per-waypoint augmented variable
        ``aug = [chain0_q | chain1_q | ... | base_xyz | base_rpy]``
        and routes:
          * each chain's joint slice through its own FK to recover
            chain-link world poses (mass contribution from the chain),
          * the base portion through the unified centroid forward
            from :mod:`skrobot.dynamics` so fixed-to-base links also
            pick up the right rigid transform.

        Per-waypoint targets are wired through a frozen
        ``ComTargetParamVar`` (``tangent_dim=0``) so the optimisation
        treats them as constants while still benefiting from the
        cached JIT plan.
        """
        import jax.numpy as jnp
        import jaxls

        from skrobot.coordinates.math import normalize_mask
        from skrobot.dynamics import build_world_centroid_fn

        targets = np.asarray(spec.params['target_positions'],
                             dtype=np.float64)
        wp_indices = np.asarray(spec.params['waypoint_indices'],
                                dtype=np.int64)
        axis_mask = np.asarray(normalize_mask(spec.params['translation_axis']))
        sel_np = np.where(axis_mask == 1)[0].astype(np.int64)
        n_axes = int(sel_np.size)
        sel = jnp.asarray(sel_np)
        weight = jnp.sqrt(spec.weight)

        n_base_dof = getattr(problem, 'n_base_dof', 0)
        chain_offsets = []
        offset = 0
        for chain in problem.link_lists:
            chain_offsets.append((offset, offset + len(chain)))
            offset += len(chain)

        # Build per-chain link FK that takes the root_link world pose
        # as its base argument. The chain's natural parent (e.g.
        # ``torso_lift_link`` for Fetch's right arm) is recovered by
        # composing with its relative-to-root transform, so a single
        # ``base_pos`` / ``base_rot`` (the floating-base pose) drives
        # both the chain FK and the fixed-link contribution
        # consistently.
        get_lt_per_chain = _build_root_relative_chain_fks(
            problem, fk_data, jnp)
        root_pos_np, root_rot_np = _root_link_world_pose(problem)
        compute_centroid_mc = build_world_centroid_fn(
            problem.centroid_data, get_lt_per_chain, backend=jnp,
            base_pos_default=root_pos_np,
            base_rot_default=root_rot_np,
        )

        base_pos_default = jnp.asarray(root_pos_np)
        base_rot_default = jnp.asarray(root_rot_np)
        n_joints_total = problem.n_joints

        def _euler_xyz_to_matrix(rx, ry, rz):
            cx, sx = jnp.cos(rx), jnp.sin(rx)
            cy, sy = jnp.cos(ry), jnp.sin(ry)
            cz, sz = jnp.cos(rz), jnp.sin(rz)
            Rx = jnp.array([[1, 0, 0],
                            [0, cx, -sx],
                            [0, sx, cx]])
            Ry = jnp.array([[cy, 0, sy],
                            [0, 1, 0],
                            [-sy, 0, cy]])
            Rz = jnp.array([[cz, -sz, 0],
                            [sz, cz, 0],
                            [0, 0, 1]])
            return Rx @ Ry @ Rz

        def compute_centroid(angles_aug):
            angles_per_chain = [
                angles_aug[a:b] for (a, b) in chain_offsets
            ]
            if n_base_dof == 0:
                return compute_centroid_mc(
                    angles_per_chain,
                    base_pos=base_pos_default,
                    base_rot=base_rot_default,
                )
            base_section = angles_aug[n_joints_total:]
            if n_base_dof == 6:
                bx, by, bz, rx, ry, rz = (
                    base_section[0], base_section[1], base_section[2],
                    base_section[3], base_section[4], base_section[5])
                base_pos = base_pos_default + jnp.array([bx, by, bz])
                base_rot = base_rot_default @ _euler_xyz_to_matrix(
                    rx, ry, rz)
            else:  # planar 3 DoF (x, y, yaw)
                bx, by, ryaw = (
                    base_section[0], base_section[1], base_section[2])
                base_pos = base_pos_default + jnp.array([bx, by, 0.0])
                base_rot = base_rot_default @ _euler_xyz_to_matrix(
                    0.0, 0.0, ryaw)
            return compute_centroid_mc(
                angles_per_chain, base_pos=base_pos, base_rot=base_rot,
            )

        # Targets per waypoint, only the selected axes.
        if targets.ndim == 2:
            targets_sel = targets[:, sel_np]
        else:
            # Single target broadcast to every waypoint.
            targets_sel = np.broadcast_to(
                targets[sel_np], (len(wp_indices), n_axes),
            )

        default_target = jnp.zeros(n_axes)

        class ComTargetParamVar(
            jaxls.Var[jnp.ndarray],
            default_factory=lambda: default_target,
            retract_fn=lambda x, delta: x,
            tangent_dim=0,
        ):
            pass

        @jaxls.Cost.factory(name='com')
        def com_cost(vals, joint_var, target_var):
            angles = vals[joint_var]
            cog = compute_centroid(angles)
            target = vals[target_var]
            err = (cog[sel] - target) * weight
            return err.flatten()

        # Stash the param var class + values so the outer ``solve()``
        # method can populate them in init_pairs alongside the joint
        # trajectory.
        if not hasattr(problem, '_com_param_vars'):
            problem._com_param_vars = []
        problem._com_param_vars.append({
            'var_class': ComTargetParamVar,
            'targets_sel': jnp.asarray(targets_sel, dtype=jnp.float64),
        })

        return com_cost(
            TrajectoryVar(jnp.asarray(wp_indices)),
            ComTargetParamVar(jnp.arange(len(wp_indices))),
        )

    def _make_multi_ee_waypoint_cost(self, problem, TrajectoryVar,
                                      fk_data, spec):
        """Multi-EE per-waypoint pose tracking with floating base.

        At every waypoint, drive every chain's end-effector to its
        corresponding world-frame target. This is the missing piece
        that lets a single trajectory-optimisation solve handle a
        multi-stance gait (foot pose constraints + base 6-DoF + CoM)
        in one shot.
        """
        import jax.numpy as jnp
        import jaxls

        from skrobot.planner.trajectory_optimization.fk_utils import pose_error_log

        T = problem.n_waypoints
        n_joints_total = problem.n_joints
        n_base_dof = getattr(problem, 'n_base_dof', 0)
        n_chains = len(problem.link_lists)

        chain_offsets = []
        offset = 0
        for chain in problem.link_lists:
            chain_offsets.append((offset, offset + len(chain)))
            offset += len(chain)

        target_pos_per_chain = spec.params['target_positions_per_chain']
        target_rot_per_chain = spec.params['target_rotations_per_chain']
        # Per-chain weights: fall back to scalar position/rotation_weight
        # for chains not explicitly weighted.  The sqrt is folded into
        # the residual scale so the LM minimises the squared weighted
        # error (= original weight * squared-error).
        pos_w_pc = spec.params.get('position_weights_per_chain')
        rot_w_pc = spec.params.get('rotation_weights_per_chain')
        if pos_w_pc is not None:
            pos_w_chain = jnp.sqrt(jnp.asarray(pos_w_pc, dtype=jnp.float64))
            np.asarray(pos_w_pc, dtype=np.float64)
        else:
            pos_w_chain = jnp.sqrt(jnp.asarray(
                [spec.params['position_weight']] * n_chains,
                dtype=jnp.float64))
            np.full(
                n_chains, float(spec.params['position_weight']),
                dtype=np.float64)
        if target_rot_per_chain is not None:
            if rot_w_pc is not None:
                rot_w_chain = jnp.sqrt(
                    jnp.asarray(rot_w_pc, dtype=jnp.float64))
                rot_w_pc_arr = np.asarray(rot_w_pc, dtype=np.float64)
            else:
                rot_w_chain = jnp.sqrt(jnp.asarray(
                    [spec.params['rotation_weight']] * n_chains,
                    dtype=jnp.float64))
                rot_w_pc_arr = np.full(
                    n_chains, float(spec.params['rotation_weight']),
                    dtype=np.float64)
        else:
            rot_w_chain = None
            rot_w_pc_arr = np.zeros(n_chains, dtype=np.float64)
        # Chains with zero rotation weight bypass the SE(3) log error to
        # avoid the body-frame translation coupling (J^{-1}(ω) term)
        # that warps gradients when actual_rot != target_rot.  Decided
        # at trace time so JAX sees a static graph per chain.
        chain_track_rot = [
            bool(rot_w_pc_arr[ci] > 0.0) for ci in range(n_chains)]

        # Per-chain EE FK that takes the root_link world pose as its
        # base argument; the chain's natural parent transform relative
        # to root is composed inside.
        get_ee_pose_per_chain = _build_root_relative_chain_ee_fks(
            problem, jnp)
        root_pos_np, root_rot_np = _root_link_world_pose(problem)
        base_pos_default = jnp.asarray(root_pos_np)
        base_rot_default = jnp.asarray(root_rot_np)

        def _euler_xyz_to_matrix(rx, ry, rz):
            cx, sx = jnp.cos(rx), jnp.sin(rx)
            cy, sy = jnp.cos(ry), jnp.sin(ry)
            cz, sz = jnp.cos(rz), jnp.sin(rz)
            Rx = jnp.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            Ry = jnp.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            Rz = jnp.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            return Rx @ Ry @ Rz

        def _split_aug(angles_aug):
            angles_per_chain = [
                angles_aug[a:b] for (a, b) in chain_offsets
            ]
            if n_base_dof == 0:
                return angles_per_chain, base_pos_default, base_rot_default
            base_section = angles_aug[n_joints_total:]
            if n_base_dof == 6:
                base_pos = base_pos_default + base_section[:3]
                rx, ry, rz = base_section[3], base_section[4], base_section[5]
                base_rot = base_rot_default @ _euler_xyz_to_matrix(rx, ry, rz)
            else:
                base_pos = base_pos_default + jnp.array(
                    [base_section[0], base_section[1], 0.0])
                base_rot = base_rot_default @ _euler_xyz_to_matrix(
                    0.0, 0.0, base_section[2])
            return angles_per_chain, base_pos, base_rot

        # Stack per-chain targets to (T, n_chains, 3) and (T, n_chains, 9).
        T_pos = np.stack(target_pos_per_chain, axis=1)  # (T, n_chains, 3)
        if target_rot_per_chain is not None:
            T_rot = np.stack(
                [r.reshape(T, 9) for r in target_rot_per_chain], axis=1,
            )
        else:
            T_rot = None

        default_pos = jnp.zeros((n_chains, 3))
        default_rot = jnp.zeros((n_chains, 9))

        class MEEPosParamVar(
            jaxls.Var[jnp.ndarray],
            default_factory=lambda: default_pos,
            retract_fn=lambda x, delta: x,
            tangent_dim=0,
        ):
            pass

        if T_rot is not None:
            class MEERotParamVar(
                jaxls.Var[jnp.ndarray],
                default_factory=lambda: default_rot,
                retract_fn=lambda x, delta: x,
                tangent_dim=0,
            ):
                pass
        else:
            MEERotParamVar = None

        if T_rot is not None:
            @jaxls.Cost.factory(name='multi_ee_waypoint')
            def cost_pose(vals, var, pos_param, rot_param):
                aug = vals[var]
                apc, bpos, brot = _split_aug(aug)
                tgt_pos = vals[pos_param]    # (n_chains, 3)
                tgt_rot = vals[rot_param]    # (n_chains, 9)
                errs = []
                for ci in range(n_chains):
                    ee_pos, ee_rot = get_ee_pose_per_chain[ci](
                        apc[ci], bpos, brot)
                    if chain_track_rot[ci]:
                        target_rot = tgt_rot[ci].reshape(3, 3)
                        pose_err = pose_error_log(
                            ee_pos, ee_rot, tgt_pos[ci], target_rot)
                        errs.append(pos_w_chain[ci] * pose_err[:3])
                        errs.append(rot_w_chain[ci] * pose_err[3:])
                    else:
                        # World-frame position error only — avoids the
                        # SE(3) log's J^{-1}(ω) coupling that distorts
                        # gradients when rot_w == 0.
                        errs.append(
                            pos_w_chain[ci] * (ee_pos - tgt_pos[ci]))
                return jnp.concatenate(errs).flatten()

            cost_node = cost_pose(
                TrajectoryVar(jnp.arange(T)),
                MEEPosParamVar(jnp.arange(T)),
                MEERotParamVar(jnp.arange(T)),
            )
        else:
            @jaxls.Cost.factory(name='multi_ee_waypoint')
            def cost_pos_only(vals, var, pos_param):
                aug = vals[var]
                apc, bpos, brot = _split_aug(aug)
                tgt_pos = vals[pos_param]    # (n_chains, 3)
                errs = []
                for ci in range(n_chains):
                    ee_pos, _ = get_ee_pose_per_chain[ci](
                        apc[ci], bpos, brot)
                    errs.append(pos_w_chain[ci] * (ee_pos - tgt_pos[ci]))
                return jnp.concatenate(errs).flatten()

            cost_node = cost_pos_only(
                TrajectoryVar(jnp.arange(T)),
                MEEPosParamVar(jnp.arange(T)),
            )

        # Stash so solve() can populate values.
        if not hasattr(problem, '_multi_ee_param_vars'):
            problem._multi_ee_param_vars = []
        problem._multi_ee_param_vars.append({
            'pos_var_class': MEEPosParamVar,
            'rot_var_class': MEERotParamVar,
            'target_pos': jnp.asarray(T_pos, dtype=jnp.float64),
            'target_rot': (jnp.asarray(T_rot, dtype=jnp.float64)
                            if T_rot is not None else None),
            # Metadata so problem.update_multi_ee_targets can swap
            # the per-frame targets without rebuilding / recompiling.
            'n_chains': n_chains,
            'n_waypoints': T,
        })

        return cost_node

    def _make_base_pose_cost(self, problem, TrajectoryVar, spec):
        """Per-waypoint base-pose tracking cost.

        Drives the floating-base 6 DoF (or 3 DoF for planar) so the
        base world pose at every waypoint follows the target trajectory.
        Uses simple (Euclidean translation, axis-angle rotation) errors
        — no SE(3) coupling — because both targets and the base
        parameterisation live in the world frame.
        """
        import jax.numpy as jnp
        import jaxls

        T = problem.n_waypoints
        n_joints_total = problem.n_joints
        n_base_dof = getattr(problem, 'n_base_dof', 0)
        if n_base_dof == 0:
            return None

        target_pos = np.asarray(
            spec.params['target_positions'], dtype=np.float64)
        target_rot = spec.params['target_rotations']
        if target_rot is not None:
            target_rot = np.asarray(target_rot, dtype=np.float64)
        pos_w = jnp.sqrt(jnp.asarray(
            spec.params['position_weight'], dtype=jnp.float64))
        rot_w = jnp.sqrt(jnp.asarray(
            spec.params['rotation_weight'], dtype=jnp.float64))

        root_pos_np, root_rot_np = _root_link_world_pose(problem)
        base_pos_default = jnp.asarray(root_pos_np)
        base_rot_default = jnp.asarray(root_rot_np)

        def _euler_xyz_to_matrix(rx, ry, rz):
            cx, sx = jnp.cos(rx), jnp.sin(rx)
            cy, sy = jnp.cos(ry), jnp.sin(ry)
            cz, sz = jnp.cos(rz), jnp.sin(rz)
            Rx = jnp.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            Ry = jnp.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            Rz = jnp.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            return Rx @ Ry @ Rz

        def _split_aug(angles_aug):
            base_section = angles_aug[n_joints_total:]
            if n_base_dof == 6:
                base_pos = base_pos_default + base_section[:3]
                base_rot = base_rot_default @ _euler_xyz_to_matrix(
                    base_section[3], base_section[4], base_section[5])
            else:
                base_pos = base_pos_default + jnp.array(
                    [base_section[0], base_section[1], 0.0])
                base_rot = base_rot_default @ _euler_xyz_to_matrix(
                    0.0, 0.0, base_section[2])
            return base_pos, base_rot

        def _so3_log(R):
            # Robust axis-angle log: clamp trace to [-1, 3] for safety.
            tr = jnp.trace(R)
            cos_theta = jnp.clip(0.5 * (tr - 1.0), -1.0, 1.0)
            theta = jnp.arccos(cos_theta)
            # Avoid 0/0 with small_angle and singular sin.
            sin_theta = jnp.sin(theta)
            safe = jnp.where(sin_theta < 1e-6, 1.0, sin_theta)
            scale = theta / (2.0 * safe)
            v = jnp.array([
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1],
            ])
            return scale * v

        T_pos_jax = jnp.asarray(target_pos, dtype=jnp.float64)
        if target_rot is not None:
            T_rot_jax = jnp.asarray(target_rot.reshape(T, 9),
                                    dtype=jnp.float64)
        else:
            T_rot_jax = None

        default_pos3 = jnp.zeros(3)
        default_rot9 = jnp.zeros(9)

        class BaseTgtPosVar(
            jaxls.Var[jnp.ndarray],
            default_factory=lambda: default_pos3,
            retract_fn=lambda x, delta: x,
            tangent_dim=0,
        ):
            pass

        if T_rot_jax is not None:
            class BaseTgtRotVar(
                jaxls.Var[jnp.ndarray],
                default_factory=lambda: default_rot9,
                retract_fn=lambda x, delta: x,
                tangent_dim=0,
            ):
                pass
        else:
            BaseTgtRotVar = None

        if T_rot_jax is not None:
            @jaxls.Cost.factory(name='base_pose')
            def cost_base(vals, var, pos_param, rot_param):
                aug = vals[var]
                bpos, brot = _split_aug(aug)
                tgt_pos = vals[pos_param]
                tgt_rot = vals[rot_param].reshape(3, 3)
                pos_err = pos_w * (bpos - tgt_pos)
                rot_err = rot_w * _so3_log(brot.T @ tgt_rot)
                return jnp.concatenate([pos_err, rot_err]).flatten()

            cost_node = cost_base(
                TrajectoryVar(jnp.arange(T)),
                BaseTgtPosVar(jnp.arange(T)),
                BaseTgtRotVar(jnp.arange(T)),
            )
        else:
            @jaxls.Cost.factory(name='base_pose')
            def cost_base_pos(vals, var, pos_param):
                aug = vals[var]
                bpos, _ = _split_aug(aug)
                tgt_pos = vals[pos_param]
                return (pos_w * (bpos - tgt_pos)).flatten()

            cost_node = cost_base_pos(
                TrajectoryVar(jnp.arange(T)),
                BaseTgtPosVar(jnp.arange(T)),
            )

        if not hasattr(problem, '_base_pose_param_vars'):
            problem._base_pose_param_vars = []
        problem._base_pose_param_vars.append({
            'pos_var_class': BaseTgtPosVar,
            'rot_var_class': BaseTgtRotVar,
            'target_pos': T_pos_jax,
            'target_rot': T_rot_jax,
            'n_waypoints': T,
        })
        return cost_node
