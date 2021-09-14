import rospy
import numpy as np

from geometry_msgs.msg import TwistStamped

from rosbag_to_dataset.dtypes.base import Dtype

class TwistConvert(Dtype):
    """
    There's a bit of overloading here. This converter either:
    1. Converts a twist message into a 6D velocity/orientation
    2. Converts a twist message into a 2D throttle/steer (using linear.x, angular.z)
    """
    def __init__(self, mode='state'):
        """
        Args:
            mode: One of {'state', 'action'}. How to interpret the twist command.
        """
        assert mode in {'state', 'action'}, "Expected mode to be one of ['state', 'action']. Got {}".format(mode)
        self.mode = mode

    def N(self):
        return 2 if self.mode == 'action' else 6

    def rosmsg_type(self):
        return TwistStamped

    def ros_to_numpy(self, msg):
        if self.mode == 'state':
            vx = msg.twist.linear.x
            vy = msg.twist.linear.y
            vz = msg.twist.linear.z
            wx = msg.twist.angular.x
            wy = msg.twist.angular.y
            wz = msg.twist.angular.z
            return np.array([vx, vy, vz, wx, wy, wz])
        elif self.mode == 'action':
            throttle = msg.twist.linear.x
            steer = msg.twist.angular.z
            return np.array([throttle, steer])

if __name__ == "__main__":
    c1 = TwistConvert('state')
    c2 = TwistConvert('action')
    msg = TwistStamped()

    print(c1.ros_to_numpy(msg))
    print(c2.ros_to_numpy(msg))
