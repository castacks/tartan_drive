import rospy
import numpy as np
import cv2

from sensor_msgs.msg import Image

from rosbag_to_dataset.dtypes.base import Dtype

class ImageConvert(Dtype):
    """
    For image, we'll rescale and 
    """
    def __init__(self, nchannels, output_resolution, empty_value=None, aggregate='none'):
        """
        Args:
            nchannels: The number of channels in the image
            output_resolution: The size to rescale the image to
            aggregate: One of {'none', 'bigendian', 'littleendian'}. Whether to leave the number of channels alone, or to combine with MSB left-to-right or right-to-left respectively.
            empty_value: A value signifying no data. Replace with the 99th percentile value to make learning simpler.
        """
        self.nchannels = nchannels
        self.output_resolution = output_resolution
        self.aggregate = aggregate
        self.empty_value = empty_value

    def N(self):
        return [self.nchannels] + self.output_resolution

    def rosmsg_type(self):
        return Image

    def ros_to_numpy(self, msg):
#        assert isinstance(msg, self.rosmsg_type()), "Got {}, expected {}".format(type(msg), self.rosmsg_type())

        is_rgb = ('8' in msg.encoding)
        if is_rgb:
            data = np.frombuffer(msg.data, dtype=np.uint8).copy()
        else:
            data = np.frombuffer(msg.data, dtype=np.float32).copy()

        data = data.reshape(msg.height, msg.width, self.nchannels)

        if self.empty_value:
            mask = np.isclose(abs(data), self.empty_value)
            fill_value = np.percentile(data[~mask], 99)
            data[mask] = fill_value

        data = cv2.resize(data, dsize=(self.output_resolution[0], self.output_resolution[1]), interpolation=cv2.INTER_AREA)

        if self.aggregate == 'littleendian':
            data = sum([data[:, :, i] * (256**i) for i in range(self.nchannels)])
        elif self.aggregate == 'bigendian':
            data = sum([data[:, :, -(i+1)] * (256**i) for i in range(self.nchannels)])

        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=0)
        else:
            data = np.moveaxis(data, 2, 0) #Switch to channels-first

        if is_rgb:
            data = data.astype(np.float32) / (255. if self.aggregate == 'none' else 255.**self.nchannels)

        return data

if __name__ == "__main__":
    c = ImageConvert(nchannels=1, output_resolution=[32, 32])
    msg = Image(width=64, height=64, data=np.arange(64**2).astype(np.uint8))

    print(c.ros_to_numpy(msg))
