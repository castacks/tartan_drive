import rospy
import numpy as np
import cv2

from grid_map_msgs.msg import GridMap

from rosbag_to_dataset.dtypes.base import Dtype

class GridMapConvert(Dtype):
    """
    Handle GridMap msgs (very similar to images)
    """
    def __init__(self, channels, output_resolution, empty_value=None):
        """
        Args:
            channels: The names of the channels to stack into an image.
            output_resolution: The size to rescale the image to
            empty_value: The value to look for if no data available at that point. Fill with 99th percentile value of data.
        """
        self.channels = channels
        self.output_resolution = output_resolution
        self.empty_value = empty_value

    def N(self):
        return [len(self.channels)] + self.output_resolution

    def rosmsg_type(self):
        return GridMap

    def ros_to_numpy(self, msg):
#        assert isinstance(msg, self.rosmsg_type()), "Got {}, expected {}".format(type(msg), self.rosmsg_type())
        data_out = []

        for channel in self.channels:
            idx = msg.layers.index(channel)
            layer = msg.data[idx]
            height = layer.layout.dim[0].size
            width = layer.layout.dim[1].size
            data = np.array(list(layer.data), dtype=np.float32) #Why was hte data a tuple?
            data = data.reshape(height, width)
            mask = np.isclose(data, self.empty_value)
            fill_value = np.percentile(data[~mask], 99)
            data[mask] = fill_value
            mask = mask.astype(np.float32)

            data = cv2.resize(data, dsize=(self.output_resolution[0], self.output_resolution[1]), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, dsize=(self.output_resolution[0], self.output_resolution[1]), interpolation=cv2.INTER_AREA)
            
            data_out.append(data)

        data_out = np.stack(data_out, axis=0)

        return data_out

if __name__ == "__main__":
    c = ImageConvert(nchannels=1, output_resolution=[32, 32])
    msg = Image(width=64, height=64, data=np.arange(64**2).astype(np.uint8))

    print(c.ros_to_numpy(msg))
