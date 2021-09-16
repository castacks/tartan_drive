# TartanDrive: A Large-Scale Dataset for Learning Off-Road Dynamics Models

![atv_terrain](https://user-images.githubusercontent.com/23179345/133315890-9cbb982f-4ac5-4640-88b3-319c10a2d43a.png)

## Dependencies

[ros (tested for melodic)](http://wiki.ros.org/melodic)

[racepak_msgs](https://github.com/castacks/physics_atv_racepak/tree/c543d85b802cd6cf64008eda0d60dc76fbafc914) 

[gridmap_msgs](https://github.com/ANYbotics/grid_map)

[rosbag_to_dataset](https://github.com/striest/rosbag_to_dataset)

## Installation

### Install ros
Follow the instructions on the [ros website](http://wiki.ros.org/melodic)

### Install gridmap_msgs

```
sudo apt-get install ros-$ROS_DISTRO-grid-map
```

### Install racepak_msgs
We use a custom message type for our shock travel, wheel RPM and pedal position data. The message descriptions can be found [here](https://github.com/castacks/physics_atv_racepak/tree/c543d85b802cd6cf64008eda0d60dc76fbafc914). In order to install them, you need to do the following:

1. clone the repo into the ROS install's src folder
~~~
cd <ROS base dir>/src
git clone https://github.com/castacks/physics_atv_racepak.git
~~~
2. Build the messages and source
~~~
cd ../
catkin_make
source devel/setup.bash
~~~
3. Verify that messages are built
~~~
rosmsg show racepak/rp_controls
~~~

### Install rosbag_to_dataset
rosbag_to_dataset should be installed as a Python 3 package. From the base directory of this repo:

~~~
cd rosbag_to_dataset
pip3 install .
~~~

### Install rospy for python3

~~~
pip3 install rospy
~~~

## Usage

### How to generate (torch) training data from bags
In order to generate torch training trajectories, one can use the ```multi_convert_bag.py``` script in ```rosbag_to_dataset/scripts```. The script takes the following arguments (also viewable via ```python3 multi_convert_bag.py -h```): 

|Argument|Description|
| ------ | --------- |
| ```--bag_dir``` | The location of the directory containing all of the bag files |
| ```--save_to``` | The location to save the resulting torch trajectories |
| ```--use_stamps``` | Whther to use the mesage stamps for time or the stamp given in the rosbag message. Setting to True is recommended. |
| ```--torch``` | Whether to save the trajectory as a numpy or torch file. Setting to True is recommended. |
| ```--zero_pose_init``` | Whether to use the raw GPS state, or to initialize each trajectory to start at (0, 0, 0). Note that this does not rotate the trajectory. Setting to True is recommended. |
| ```--config_spec```| Path to the YAML file that defines how the dataset is to be generated. To recreate the dataset used to train the models in the paper, use ```../specs/2021_atv_2.yaml```. |

#### Description of the Config Spec YAML
The YAML file to parse datasets is generally a dictionary of ```{observation, action, dt}```, where observations and actions are themselves dictionaries of options.
1. ```dt:<rate>``` gives the time (in s) between each step in the dataset. ```dt: 0.1``` is recommended.
2. ```action``` is a dictionary of the following form:
```
action:
  <topic>:
    type:<rosmsg type>
    options: <dict of topic-specific options>
```
3. ```observation``` is also a dictionary of topics to be included in the dataset:
```
observation:
  <topic>:
    type: <rosmsg type>
    remap: <name to call this topic in the dataset. Omit to use the rostopic name>
    N_per_step: <For time-series data, stack this many messages into a single step>
    options: <topic-specific options>
```

#### Supported rostopics
1. ```AckermannDriveStamped```
2. ```Float64```
3. ```GridMap```
4. ```Image```
5. ```IMU```
6. ```Odometry```
7. ```PoseStamped```
8. ```TwistStamped```

### Useful Scripts

## Data Description

| Modality  | Dimension  | Topic | Frequency |
| --------- | ---------- | ----- | --------- |
| State     | 7          | /odometry/filtered_odom | 50Hz |
| Action    | 2          | /cmd | 100Hz |
| RGB Image | 1024 x 512 | /multisense/left/image_rect_color | 20Hz |
| RGB Map   | 501 x 501  | /local_rgb_map | 20Hz |
| Heightmap | 501 x 501  | /local_height_map | 20Hz |
| IMU       | 6      | /multisense/imu/imu_data | 200Hz
| Shock Pos | 4      | /shock_pos | 50Hz |
| Wheel RPM | 4      | /wheel_rpm | 50Hz |
| Pedals    | 2      | /controls  | 50Hz |
