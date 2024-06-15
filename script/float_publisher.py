#!/usr/bin/env python3
import sys
import rospy
from moveit_commander import MoveGroupCommander, PlanningSceneInterface, roscpp_initialize, roscpp_shutdown
from moveit_msgs.msg import DisplayTrajectory
from geometry_msgs.msg import Pose
import tf
from math import pi
from threading import Thread
from sensor_msgs.msg import CameraInfo


def main():

    roscpp_initialize(sys.argv)
    rospy.init_node("move_group_interface_tutorial", anonymous=True)

    # Start the spinning thread
    spinner_thread = Thread(target=lambda: rospy.spin())
    spinner_thread.start()

    rospy.sleep(2.0)

    group = MoveGroupCommander("manipulator")
    planning_scene_interface = PlanningSceneInterface()

    display_publisher = rospy.Publisher("/move_group/display_planned_path", DisplayTrajectory, queue_size=1)

    rospy.loginfo("Reference frame: %s", group.get_planning_frame())
    rospy.loginfo("Reference frame: %s", group.get_end_effector_link())


    # Target position
    target_pose1 = Pose()
    orientation = tf.transformations.quaternion_from_euler(3.114, 0.052, -2.319)
    target_pose1.orientation.x = orientation[0]
    target_pose1.orientation.y = orientation[1]
    target_pose1.orientation.z = orientation[2]
    target_pose1.orientation.w = orientation[3]
    target_pose1.position.x = -0.333
    target_pose1.position.y = -0.260
    target_pose1.position.z = 0.554
    group.set_pose_target(target_pose1)

    # Visualize the planning
    plan = group.plan()
    success = plan[0]
    rospy.loginfo("Visualizing plan %s", "SUCCESS" if success else "FAILED")

    # Move the group arm
    if success:
        group.go(wait=True)

    rospy.sleep(1.0)

    roscpp_shutdown()

if __name__ == '__main__':
    main()