#!/usr/bin/env python
"""Batch IK with interference (collision) avoidance demo.

Demonstrates :meth:`~skrobot.model.RobotModel.batch_inverse_kinematics`'s
collision-avoidance arguments: passing ``collision_link_list`` together with
``collision_obstacles`` (and optionally ``self_collision=True``) steers the
batch IK solutions away from user-specified obstacles. Under the hood this
switches the solver from the fast damped-least-squares ("jacobian") method to
a gradient-descent method that adds a smooth collision-distance penalty term
to the optimized cost -- a soft constraint, not a hard one, so a
collision-free result is not guaranteed for every target, especially in
cluttered scenes.

This script solves the same batch of target poses twice against a box
obstacle placed where the naive (no-avoidance) solution's elbow/forearm
would otherwise rest:

1. Without collision avoidance (default jacobian solver) -- fast, precise,
   but the solved poses clip through the box.
2. With collision avoidance (``collision_obstacles=[box]``) -- slower, and
   trading a bit of position accuracy, but the solved poses clear the box.

It then prints a before/after comparison table and (unless
``--no-interactive``) opens a viewer that cycles through each target's naive
pose and its collision-avoiding counterpart, color-coding the approximate
collision spheres red (penetrating) / yellow (clear).

Usage:
    python batch_ik_collision_avoidance_demo.py
    python batch_ik_collision_avoidance_demo.py --no-interactive
    python batch_ik_collision_avoidance_demo.py --save-video out.mp4
"""

import argparse
import time

import numpy as np

from skrobot.collision import RobotCollisionChecker
from skrobot.coordinates import Coordinates
from skrobot.model.primitives import Axis
from skrobot.model.primitives import Box
from skrobot.models import Fetch
from skrobot.utils.video import record_viewer
from skrobot.viewers import VIEWER_HELP
from skrobot.viewers import VIEWER_TYPES


def build_scene():
    """Build the robot, obstacle, and target poses used by this demo."""
    robot = Fetch()
    robot.reset_pose()
    arm = robot.arm
    collision_links = arm.link_list

    # Placed where a naive IK solution's elbow/forearm would otherwise
    # rest -- e.g. a shelf sitting below and in front of the arm.
    obstacle = Box(extents=[0.35, 0.45, 0.35])
    obstacle.set_color([200, 120, 40, 180])
    obstacle.translate([0.45, -0.15, 0.55])

    target_poses = [
        Coordinates(pos=(0.75, -0.15, 0.85)).rotate(np.deg2rad(20), 'y'),
        Coordinates(pos=(0.7, -0.1, 0.95)).rotate(np.deg2rad(-15), 'z'),
        Coordinates(
            pos=(0.78, -0.05, 0.9)
        ).rotate(np.deg2rad(25), 'y').rotate(np.deg2rad(-10), 'z'),
    ]
    return robot, arm, collision_links, obstacle, target_poses


def min_distance_to_obstacle(robot, collision_links, obstacle, angle_vector):
    """Independently verify obstacle clearance for one solution.

    Uses :class:`~skrobot.collision.RobotCollisionChecker`, a code path
    separate from the solver's own internal collision penalty, so this is a
    genuine check rather than re-reporting the optimizer's own cost.
    """
    robot.angle_vector(angle_vector)
    checker = RobotCollisionChecker(robot)
    checker.add_links(collision_links)
    checker.add_world_obstacle(obstacle)
    return float(checker.compute_min_distance())


def main():
    parser = argparse.ArgumentParser(
        description='Batch IK with interference (collision) avoidance demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--no-interactive', action='store_true',
                         help='Skip the interactive viewer (still prints '
                              'the before/after comparison table).')
    parser.add_argument('--viewer', type=str, choices=VIEWER_TYPES,
                         default='pyrender', help=VIEWER_HELP)
    parser.add_argument(
        '--save-video', type=str, default=None,
        help='Record the animation to this video file (e.g. out.mp4). '
             'Works with any --viewer; use --viewer mitsuba to record '
             'headlessly.')
    parser.add_argument('--collision-weight', type=float, default=8.0,
                         help='Weight of the obstacle-avoidance penalty term.')
    parser.add_argument('--collision-margin', type=float, default=0.05,
                         help='Clearance (m) the solver tries to keep from '
                              'the obstacle.')
    parser.add_argument('--attempts-per-pose', type=int, default=5,
                         help='Random-restart attempts per target for the '
                              'collision-avoiding solve.')
    args = parser.parse_args()

    print("BATCH IK WITH INTERFERENCE (COLLISION) AVOIDANCE")
    print("=" * 60)

    np.random.seed(0)  # reproducible random-restart attempts
    robot, arm, collision_links, obstacle, target_poses = build_scene()
    print(f"{len(target_poses)} target pose(s), 1 box obstacle at "
          f"{np.round(obstacle.worldpos(), 3)}")

    print("\n--- Solving WITHOUT collision avoidance (default jacobian solver) ---")
    naive_solutions, naive_success, _ = robot.batch_inverse_kinematics(
        target_poses, move_target=arm.end_coords, link_list=collision_links,
        attempts_per_pose=1,
    )

    print("--- Solving WITH collision avoidance "
          "(gradient-descent + obstacle penalty) ---")
    # Looser thresholds than the jacobian solver's defaults: the collision
    # penalty is a soft constraint that trades a few centimeters of pose
    # accuracy for guaranteed clearance, and gradient descent itself
    # converges more slowly than the damped-least-squares solver above.
    avoid_solutions, avoid_success, _ = robot.batch_inverse_kinematics(
        target_poses, move_target=arm.end_coords, link_list=collision_links,
        attempts_per_pose=args.attempts_per_pose, stop=500,
        thre=0.03, rthre=np.deg2rad(10.0),
        collision_link_list=collision_links, collision_obstacles=[obstacle],
        collision_weight=args.collision_weight,
        collision_margin=args.collision_margin,
    )

    original_av = robot.angle_vector()

    print("\nRESULTS (min_dist < 0 means the arm penetrates the obstacle)")
    print("-" * 78)
    print(f"{'target':>6s} | {'naive pos_err':>13s} {'min_dist':>9s} {'ok':>3s} |"
          f" {'avoid pos_err':>13s} {'min_dist':>9s} {'ok':>3s}")
    print("-" * 78)
    for i, target in enumerate(target_poses):
        robot.angle_vector(naive_solutions[i])
        naive_pos_err = np.linalg.norm(
            arm.end_coords.worldpos() - target.worldpos())
        naive_dmin = min_distance_to_obstacle(
            robot, collision_links, obstacle, naive_solutions[i])

        robot.angle_vector(avoid_solutions[i])
        avoid_pos_err = np.linalg.norm(
            arm.end_coords.worldpos() - target.worldpos())
        avoid_dmin = min_distance_to_obstacle(
            robot, collision_links, obstacle, avoid_solutions[i])

        print(f"{i:>6d} | {naive_pos_err:>13.4f} {naive_dmin:>+9.4f} "
              f"{str(naive_success[i]):>3s} |"
              f" {avoid_pos_err:>13.4f} {avoid_dmin:>+9.4f} "
              f"{str(avoid_success[i]):>3s}")
    print("-" * 78)
    robot.angle_vector(original_av)

    if args.no_interactive and args.save_video is None:
        return

    from skrobot.viewers import create_viewer
    viewer = create_viewer(args.viewer, resolution=(800, 600))
    viewer.add(robot)
    viewer.add(obstacle)
    for target in target_poses:
        axis = Axis.from_coords(target, axis_radius=0.008, axis_length=0.12)
        viewer.add(axis)

    checker = RobotCollisionChecker(robot)
    checker.add_links(collision_links)
    checker.add_world_obstacle(obstacle)
    checker.add_coll_spheres_to_viewer(viewer)

    viewer.show()
    recorder = record_viewer(viewer, args.save_video, fps=2)

    print("\n3D VISUALIZATION")
    print("Cycling per target: naive pose (often red = colliding) then the "
          "collision-avoiding pose (yellow = clear).")
    print("Collision spheres approximate each arm link for visualization; "
          "close the window to exit." if not args.save_video else "")

    frames = []
    for i in range(len(target_poses)):
        frames.append(('naive', i, naive_solutions[i]))
        frames.append(('avoid', i, avoid_solutions[i]))

    idx = 0
    last_change = time.time()
    hold_time = 1.2
    try:
        while viewer.is_active:
            now = time.time()
            if now - last_change > hold_time:
                label, i, av = frames[idx]
                robot.angle_vector(av)
                dmin = checker.compute_min_distance()
                checker.update_color()
                viewer.redraw()
                status = "COLLISION" if dmin < 0 else "clear"
                print(f"\r[{label:5s}] target {i}: min_dist={dmin:+.4f} "
                      f"({status})        ", end="", flush=True)
                idx += 1
                last_change = now
                if recorder is not None and idx >= len(frames):
                    break
                idx %= len(frames)
            viewer.pause(0.05)
    except KeyboardInterrupt:
        pass
    print()

    if recorder is not None:
        print(f"saving video to {recorder.save()}")
    elif not args.no_interactive:
        print("Visualization completed")

    viewer.close()


if __name__ == '__main__':
    main()
