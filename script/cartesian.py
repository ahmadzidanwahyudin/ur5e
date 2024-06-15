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

# Global variables to store cx and cy values
cx = None
cy = None


def callback(data):
    global cx, cy
    rospy.loginfo("I heard cx=%s, cy=%s", data.K[2], data.K[5])
    cx = data.K[2]
    cy = data.K[5]


def listener():
    rospy.Subscriber("/camera_info", CameraInfo, callback)


def main():
    # Initialize ROS and the MoveIt! commander
    roscpp_initialize(sys.argv)
    rospy.init_node("move_group_interface_tutorial", anonymous=True)

    # Start the spinning thread for the listener
    listener_thread = Thread(target=listener)
    listener_thread.start()

    group = MoveGroupCommander("manipulator")
    planning_scene_interface = PlanningSceneInterface()

    display_publisher = rospy.Publisher("/move_group/display_planned_path", DisplayTrajectory, queue_size=1)

    rospy.loginfo("Reference frame: %s", group.get_planning_frame())
    rospy.loginfo("End effector link: %s", group.get_end_effector_link())

    # Wait until cx and cy are available
    while cx is None or cy is None:
        rospy.loginfo("Waiting for camera info...")
        rospy.sleep(1.0)

    # Debugging: Print the values of cx and cy
    print(f"cx: {cx}, cy: {cy}")

    # Perform calculations based on the values of cx and cy
    if cx > 320:
        cx_robot = -((cx * 0.00203) + 0.379)
    elif cx < 320:
        cx_robot = ((cx * 0.00203) - 0.379)
    else:
        cx_robot = -0.379  # Handle the case where cx is exactly 320

    if cy > 240:
        cy_robot = -((cy * 0.00025) - 0.174)
    elif cy < 240:
        cy_robot = ((cy * 0.00025) + 0.174)
    else:
        cy_robot = - 0.174  # Handle the case where cy is exactly 240

    # Debugging: Print the results
    print(f"cx_robot: {cx_robot}, cy_robot: {cy_robot}")

    # Target position
    target_pose1 = Pose()
    orientation = tf.transformations.quaternion_from_euler(3.114, 0.052, -2.319)

    target_pose1.orientation.x = orientation[0]
    target_pose1.orientation.y = orientation[1]
    target_pose1.orientation.z = orientation[2]
    target_pose1.orientation.w = orientation[3]

    # Use the cx and cy values here
    target_pose1.position.x = -0.374
    target_pose1.position.y = -0.179
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
