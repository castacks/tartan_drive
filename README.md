# TartanDrive: A Large-Scale Dataset for Learning Off-Road Dynamics Models

![atv_terrain](https://user-images.githubusercontent.com/23179345/133315890-9cbb982f-4ac5-4640-88b3-319c10a2d43a.png)

## Dependencies

[racepak_msgs](https://github.com/castacks/physics_atv_racepak/tree/c543d85b802cd6cf64008eda0d60dc76fbafc914) 

[rosbag_to_dataset](https://github.com/striest/rosbag_to_dataset)

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
