# physics_atv_racepak
ROS Publisher for Racepak G2X Datalogger for Yamaha ATV Project

## Racepak Configuration
This Racepak setup is unique to the Yamaha ATV it is being used on and has been configured accordingly. To modify any of the default parameters use DataLink II v4.8.12 in conjuction with *G2X_Yamaha_Config.rcg* on a Windows PC. Open this configuration in DataLink II, modify, and SEND configuration. There is a license file needed for using DataLink II, this is kept with the project supplies.
See DataLink II manual on drive for more info on using the program.

## Cloning Package
Clone this package to a folder named `racepak` in your ROS install's `src` folder.
```
cd (ROS Base Directory)/src
mkdir racepak
git clone https://github.com/castacks/physics_atv_racepak.git ./racepak
```

## Dependencies and Setup
Python 3

[Pyserial](https://pypi.org/project/pyserial/)
If you face issues with Pyserial after setup, check the following:
* Check the port is specified correctly, default is `'/dev/ttyUSB0'`
* https://stackoverflow.com/questions/27858041/oserror-errno-13-permission-denied-dev-ttyacm0-using-pyserial-from-pyth
* Edit all files mentioned and add user to dialout group

## Usage
With roscore already running, start the background service script:
```
cd (ROS Base Directory)
rosrun racepak racepak_handler.py
```
You should now see the following topics present in ROS:
* controls
* shock_pos
* wheel_rpm

Each has a custom msg to encapsulate the appropriate data.

`controls` contains Throttle and Brake peddle linear displacement sensors. Both are float32 values representing displacement in inches.

`shock_pos` constains front_left, front_right, rear_left, and rear_right linear displacement sensors on the shocks. All are float32 values representing displacement in inches.

`wheel_rpm` constains front_left, front_right, rear_left, and rear_right rotation sensors. All are float32 values representing instantaneous rotations per minute.
