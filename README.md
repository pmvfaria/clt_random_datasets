# randgen\_omni\_dataset

This ROS package generates random datasets in a simulation environment, for multiple robots and targets with random motion. Currently, it is designed to work with the OMNI dataset, but can be adapted to different configurations.

## Brief Description

You need to test your localization/tracking algorithm for more robots than you have access to, so you desire a simulation environment. However, many of the existing simulators are time-consuming to setup. This package provides a simple to use simulation environment, which displays internal states in rviz-friendly formats.

Due to using multiple ROS nodes, it is also multi-threading friendly.

## Modules

The following modules are available in this dataset generator:

* **Robot module** - provides the interface between odometry and landmark/target observations, pose publishing, etc. It is the main module
* **Odometry module** - provides new odometry readings at a constant frequency, to which the robot module subscribes in order to advance its pose. A probabilistic noise model is used, similar to the model described in Probabilistic Robotics. The odometry changes states (from WalkForward to Rotate) through a ROS service
* **Ball(Target) module** - Simulates a ball with gravity, hovering and pulling, just like if a person would be holding the ball by a string and moving it around
* **Walls module** - Sets up a field with 4 walls
* **Landmarks module** - Loads a configuration file and sets up static landmarks in the stage. Publishes a ROS topic with the landmarks positions
* **OMNI\_custom module** - Transforms the ROS/rviz standard formats in the other modules to the OMNI dataset message format.

## Requisites

[ROS](https://www.ros.org). Tested with ROS Indigo and Kinetic. Visualization needs [rviz](https://wiki.ros.org/rviz)

## How to Use

To generate sample data, follow these steps:

1. Clone and build this package with the *read\_omni\_dataset* submodule: `git clone --recursive https://github.com/guilhermelawless/randgen_omni_dataset.git && catkin_make`
2. Execute one of the following scripts with `python <script>`:
  * If you don't need recording to a rosbag, use the *config/**create\_launch\_file*** script, followed by the number of robots you desire
    * This script creates the new.launch file in the launch directory
    * You can specify optional parameters. Use the --help option to learn about these
    
  * If you need recording to a rosbag, use the *config/**create\_launch\_record*** script, followed by the number of robots you desire
    * This script creates the new.launch file in the launch directory
    * You should use the --help option to learn how to specify the many optional parameters
3. Run: `roslaunch randgen_omni_dataset new.launch` and the simulation will begin
4. (optional) if you want to visualize the simulation, run rviz and load the configuration *rviz.rviz* in the config directory (up to 10 robots, but customizable)

## Contribute

The simulator is designed in modules, so you are welcome to make your own modules, suggest modifications, add more customizability.

At the moment, the target simulation is quite rough, and contributions to this module are appreciated.

## Using with PF-UCLT

After recording to a rosbag file, you can use [pfuclt\_omni\_dataset](https://github.com/guilhermelawless/pfuclt_omni_dataset) to try localization and target tracking.

To accomplish this, when executing the create\_launch\_file script, it will generate a file in your the pfuclt package launch folder to work with it. By default, it will be named new.launch, so you can execute it with: `roslaunch pfuclt_omni_dataset new.launch`

The dataset generation is very CPU intensive for a higher number of robots, therefore it is advised to use the create\_launch\_record script and appropriate optional parameters to generate a rosbag of the whole dataset. The generated launch file will automatically record the dataset. The generated launch file for PF-UCLT will automatically launch the rosbag player when it is used.

If necessary when performing PF-UCLT, use the rate argument of the launch file to set different rosbag playing rates.

Use the rviz config file pfuclt\_omni\_dataset/config/omni_sim.rviz to visualize the algorithm in rviz. **Important**: you need to execute rviz only after setting the /use_sim_time parameter to true, which is done by calling the launch file above.

You will know everything is working if the PF-UCLT node outputs an odometry frequency of 33Hz (by default) almost consistently. If it goes below this value, the algorithm is causing odometry readings to be delayed, so it can't keep up and you should lower the rosbag playing rate.
