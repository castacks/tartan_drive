# TartanDrive: A Large-Scale Dataset for Learning Off-Road Dynamics Models

![atv_terrain](https://user-images.githubusercontent.com/23179345/133315890-9cbb982f-4ac5-4640-88b3-319c10a2d43a.png)

## Download the Data
Data are available via ```azcopy copy https://tartandrive.blob.core.windows.net/dataset-icra22/<FILENAME>.tar.gz```. The list of files is given in ```azfiles.txt```. Each file is a compressed folder of several rosbags. Each file is roughly 100GB.

## Dependencies

[ros (tested for melodic)](http://wiki.ros.org/melodic)

## Installation

### Install ros
Follow the instructions on the [ros website](http://wiki.ros.org/melodic)

### Install racepak_msgs
We use a custom message type for our shock travel, wheel RPM and pedal position data. The message descriptions can be found [here](https://github.com/castacks/physics_atv_racepak/tree/c543d85b802cd6cf64008eda0d60dc76fbafc914). In order to install them, you need to do the following:

1. from this directory ROS workspace's src folder
~~~
cp -r physics_atv_racepak <ROS workspace>/src
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


## How to generate (torch) training data from bags
Visualization, evaluation and training require that the datasets be stored using torch. In order to generate torch training trajectories, one can use the ```multi_convert_bag.py``` script in ```dataset```. The script takes the following arguments (also viewable via ```python3 multi_convert_bag.py -h```): 

|Argument|Description|
| ------ | --------- |
| ```--bag_dir``` | The location of the directory containing all of the bag files |
| ```--save_to``` | The location to save the resulting torch trajectories |
| ```--use_stamps``` | Whther to use the mesage stamps for time or the stamp given in the rosbag message. Setting to True is recommended. |
| ```--torch``` | Whether to save the trajectory as a numpy or torch file. Setting to True is recommended. |
| ```--zero_pose_init``` | Whether to use the raw GPS state, or to initialize each trajectory to start at (0, 0, 0). Note that this does not rotate the trajectory. Setting to True is recommended. |
| ```--config_spec```| Path to the YAML file that defines how the dataset is to be generated. To recreate the dataset used to train the models in the paper, use ```../specs/2021_atv_2.yaml```. |

### Description of the Config Spec YAML
The YAML file to parse datasets is generally a dictionary of ```{observation, action, dt}```, where observations and actions are themselves dictionaries of options. We provide ```specs/2021_atv_2.yaml``` as a default.
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

### Supported rostopics
1. ```AckermannDriveStamped```
2. ```Float64```
3. ```Image```
4. ```IMU```
5. ```Odometry```
6. ```PoseStamped```
7. ```TwistStamped```

## Visualize the Data
We provide a script ```visualization/visualize_traj.py``` that generates a video of the trajectory. Usage: ```python3 visualize_traj.py --traj_fp <path to torch traj>```.
  
## Visualize Model Predictions
We provide a script ```prob_world_model_viz.py``` that produces a video of model predictions over the entire trajectory. Run ```python3 prob_world_model_viz.py -h``` for usage. Alternatively, ```evaluation/eval_prob_world_model.py``` can produce single-frame prediction visualizations.

## Reproduce Paper Results
### Create Train/Test Split
The train/test split used in the paper can be created by running ```partition_dataset.py --dataset_fp <path to dir of all torch trajs> --save_to <where to move data>```. This will create folders ```train```, ```test-easy``` and ```test-hard``` in the location specified.

### Evaluate World Model
The script ```eval_prob_world_model``` in ```evaluation```can be used to generate RMSE of a model over a directory of torch trajectories. Usage: ```eval_prob_world_model.py -h```.

## Perform Data T-SNE Clustering
We provide the script used to produce the t-SNE plots in our motivational experiment in ```visualization```. Usage: ```python3 tsne_cluster -h```
