import rospy
import numpy as np
import cv2

from sensor_msgs.msg import Image

from rosbag_to_dataset.dtypes.base import Dtype

class ImageConvert(Dtype):
    """
    For image, we'll rescale and 
    """
    def __init__(self, nchannels, output_resolution, aggregate='none'):
        """
        Args:
            nchannels: The number of channels in the image
            output_resolution: The size to rescale the image to
            aggregate: One of {'none', 'bigendian', 'littleendian'}. Whether to leave the number of channels alone, or to combine with MSB left-to-right or roght-to-left respectively.
        """
        self.nchannels = nchannels
        self.output_resolution = output_resolution
        self.aggregate = aggregate

    def N(self):
        return [self.nchannels] + self.output_resolution

    def rosmsg_type(self):
        return Image

    def ros_to_numpy(self, msg):
#        assert isinstance(msg, self.rosmsg_type()), "Got {}, expected {}".format(type(msg), self.rosmsg_type())

        data = np.frombuffer(msg.data, dtype=np.uint8)
        data = data.reshape(msg.height, msg.width, self.nchannels)
        data = cv2.resize(data, dsize=(self.output_resolution[0], self.output_resolution[1]), interpolation=cv2.INTER_AREA)

        if self.aggregate == 'littleendian':
            data = sum([data[:, :, i] * (256**i) for i in range(self.nchannels)])
        elif self.aggregate == 'bigendian':
            data = sum([data[:, :, -(i+1)] * (256**i) for i in range(self.nchannels)])

        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=0)
        else:
            data = np.moveaxis(data, 2, 0) #Switch to channels-first

        data = data.astype(np.float32) / (255. if self.aggregate == 'none' else 255.**self.nchannels)

        return data

if __name__ == "__main__":
    c = ImageConvert(nchannels=1, output_resolution=[32, 32])
    msg = Image(width=64, height=64, data=np.arange(64**2).astype(np.uint8))

    print(c.ros_to_numpy(msg))
