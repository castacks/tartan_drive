import rospy
import numpy as np

from geometry_msgs.msg import PoseStamped

from rosbag_to_dataset.dtypes.base import Dtype

class PoseConvert(Dtype):
    """
    Convert a pose message into a 7d vec.
    """
    def __init__(self, zero_position=False, use_vel=True):
        self.zero_position = zero_position
        self.initial_position = None if self.zero_position else np.zeros(3)
        self.use_vel = use_vel

    def N(self):
        return 7

    def rosmsg_type(self):
        return PoseStamped

    def ros_to_numpy(self, msg):
        if self.initial_position is None:
            self.initial_position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

        p = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        q = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        res = np.array(p + q)
        res[:3] -= self.initial_position

        return res if self.use_vel else res[:7]

if __name__ == "__main__":
    c = OdometryConvert()
    msg = Odometry()

    print(c.ros_to_numpy(msg))
