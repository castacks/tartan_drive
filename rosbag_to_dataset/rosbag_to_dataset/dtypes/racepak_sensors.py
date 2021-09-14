import rospy
import numpy as np

from racepak.msg import rp_controls, rp_shock_sensors, rp_wheel_encoders

from rosbag_to_dataset.dtypes.base import Dtype

class RPControlsConvert(Dtype):
    """
    Convert racepak controls into data. Either return the raw values, or a boolean of whether they're being presed or not. (both have to be as float.)
    """
    def __init__(self, mode='intervention', intervention_threshold=50.0):
        """
        Args:
            mode: One of ['identity', 'intervention']. Whether to give the raw throttle/brake commands, or a binary var on whether the throttle/brake was manually used.
            intervention_threshold: If in intervention mode, let intervention be true if 
        """
        self.mode = mode
        self.intervention_threshold = intervention_threshold

    def N(self):
        return 1 if self.mode == 'intervention' else 2

    def rosmsg_type(self):
        return rp_controls

    def ros_to_numpy(self, msg):
        if self.mode == 'intervention':
            if msg.throttle > self.intervention_threshold or msg.brake > self.intervention_threshold:
                return np.array([1.0])
            else:
                return np.array([0.0])
        else:
            out = np.array([msg.throttle, msg.brake])
            return out

class RPWheelEncodersConvert(Dtype):
    """
    Convert shock travel to data (really simple).
    """
    def __init__(self):
        pass

    def N(self):
        return 4

    def rosmsg_type(self):
        return rp_wheel_encoders

    def ros_to_numpy(self, msg):
        return np.array([msg.front_left, msg.front_right, msg.rear_left, msg.rear_right])

class RPShockSensorsConvert(Dtype):
    """
    Convert shock travel to data (really simple).
    """
    def __init__(self):
        pass

    def N(self):
        return 4

    def rosmsg_type(self):
        return rp_shock_sensors

    def ros_to_numpy(self, msg):
        return np.array([msg.front_left, msg.front_right, msg.rear_left, msg.rear_right])

if __name__ == "__main__":
    controls = rp_controls()
    controls_cvt = RPControlsConvert(mode='intervention')

    print(controls)
    print(controls_cvt.ros_to_numpy(controls))
    controls.brake = 100.
    print(controls)
    print(controls_cvt.ros_to_numpy(controls))

    msg = rp_shock_sensors()
    msg.front_left = 100.
    msg.front_right = 101.
    msg.rear_left = 102.
    msg.rear_right = 103.
    shock_cvt = RPShockSensorsConvert()
    print(msg)
    print(shock_cvt.ros_to_numpy(msg))
