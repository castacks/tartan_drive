import rospy
import numpy as np

from sensor_msgs.msg import Imu

from rosbag_to_dataset.dtypes.base import Dtype

class ImuConvert(Dtype):
    """
    """
    def __init__(self, orientation=False, angular_velocity=True, linear_acceleration=True, time_series=False):
        """
        Args:
            orientation: T/F. Whether to convert orientation
            angular_velocity: T/F. Whether to convert orientation
            linear_acceleration: T/F. Whether to convert orientation
            time_series: T/F. Whether to include all IMU data inbetween timesteps (this may be tricky to implement. May have to use as a flag for the converter.)
        """
        self.orientation = orientation
        self.angular_velocity = angular_velocity
        self.linear_acceleration = linear_acceleration
        self.time_series = time_series

    def N(self):
        return (4 if self.orientation else 0) + (3 if self.angular_velocity else 0) + (3 if self.linear_acceleration else 0)

    def rosmsg_type(self):
        return Imu

    def ros_to_numpy(self, msg):
        out = []
        if self.orientation:
            out += [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        if self.angular_velocity:
            out += [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
        if self.linear_acceleration:
            out += [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        return np.array(out)

if __name__ == "__main__":
    c1 = ImuConvert(True, True, True)
    print(c1.N())
    c2 = ImuConvert(True, True, False)
    print(c2.N())
    c3 = ImuConvert(True, False, True)
    print(c3.N())
    c4 = ImuConvert(True, False, False)
    print(c4.N())
    c5 = ImuConvert(False, True, True)
    print(c5.N())
    c6 = ImuConvert(False, True, False)
    print(c6.N())
    c7 = ImuConvert(False, False, True)
    print(c7.N())
    c8 = ImuConvert(False, False, False)
    print(c8.N())

    msg = Imu()
    for c in [c1, c2, c3, c4, c5, c6, c7, c8]:
        print(c.ros_to_numpy(msg))
