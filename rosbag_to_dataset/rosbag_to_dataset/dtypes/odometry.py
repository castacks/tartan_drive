import rospy
import numpy as np

from nav_msgs.msg import Odometry

from rosbag_to_dataset.dtypes.base import Dtype

class OdometryConvert(Dtype):
    """
    Convert an odometry message into a 13d vec.
    """
    def __init__(self, zero_position=False, use_vel=True):
        self.zero_position = zero_position
        self.initial_position = None if self.zero_position else np.zeros(3)
        self.use_vel = use_vel

    def N(self):
        return 13 if self.use_vel else 7

    def rosmsg_type(self):
        return Odometry

    def ros_to_numpy(self, msg):
        if self.initial_position is None:
            self.initial_position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])

        p = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        q = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        pdot = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
        qdot = [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]
        res = np.array(p + q + pdot + qdot)
        res[:3] -= self.initial_position

        return res if self.use_vel else res[:7]

if __name__ == "__main__":
    c = OdometryConvert()
    msg = Odometry()

    print(c.ros_to_numpy(msg))
