import control_msgs.msg

from skrobot.interfaces.ros.move_base import ROSRobotMoveBaseInterface


class AeroROSRobotInterface(ROSRobotMoveBaseInterface):
    """Aero ROS robot interface."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('base_frame_id', 'base_link')
        super(AeroROSRobotInterface, self).__init__(*args, **kwargs)

    @property
    def larm_controller(self):
        return dict(
            controller_type='larm_controller',
            controller_action='larm_controller/follow_joint_trajectory',
            controller_state='larm_controller/state',
            action_type=control_msgs.msg.FollowJointTrajectoryAction,
            joint_names=['l_shoulder_p_joint',
                         'l_shoulder_r_joint',
                         'l_shoulder_y_joint',
                         'l_elbow_joint',
                         'l_wrist_y_joint',
                         'l_wrist_p_joint',
                         'l_wrist_r_joint',
                         'l_hand_y_joint'])

    @property
    def rarm_controller(self):
        return dict(
            controller_type='rarm_controller',
            controller_action='rarm_controller/follow_joint_trajectory',
            controller_state='rarm_controller/state',
            action_type=control_msgs.msg.FollowJointTrajectoryAction,
            joint_names=['r_shoulder_p_joint',
                         'r_shoulder_r_joint',
                         'r_shoulder_y_joint',
                         'r_elbow_joint',
                         'r_wrist_y_joint',
                         'r_wrist_p_joint',
                         'r_wrist_r_joint',
                         'r_hand_y_joint'])

    @property
    def head_controller(self):
        return dict(
            controller_type='head_controller',
            controller_action='head_controller/follow_joint_trajectory',
            controller_state='head_controller/state',
            action_type=control_msgs.msg.FollowJointTrajectoryAction,
            joint_names=['neck_y_joint',
                         'neck_p_joint',
                         'neck_r_joint'])

    @property
    def waist_controller(self):
        return dict(
            controller_type='waist_controller',
            controller_action='waist_controller/follow_joint_trajectory',
            controller_state='waist_controller/state',
            action_type=control_msgs.msg.FollowJointTrajectoryAction,
            joint_names=['waist_y_joint',
                         'waist_p_joint',
                         'waist_r_joint'])

    @property
    def lifter_controller(self):
        return dict(
            controller_type='lifter_controller',
            controller_action='lifter_controller/follow_joint_trajectory',
            controller_state='lifter_controller/state',
            action_type=control_msgs.msg.FollowJointTrajectoryAction,
            joint_names=['knee_joint',
                         'ankle_joint'])

    def default_controller(self):
        return [self.larm_controller,
                self.rarm_controller,
                self.head_controller,
                self.waist_controller,
                self.lifter_controller]
