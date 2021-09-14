"""
Generic template for what we need from each type for this class.
"""

import abc

class Dtype:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def N(self):
        """
        Get the shape of the output
        """
        pass

    @abc.abstractmethod
    def rosmsg_type(self):
        """
        Get the type of the rosmsg we should be reading
        Note that certain dtypes may have multiplt rosmsgs. I'm not sire how to handle that cleanly yet.
        """
        pass

    @abc.abstractmethod
    def ros_to_numpy(self, msg):
        """
        Convert an instance of the rsomsg to a numpy array.
        """
        pass
