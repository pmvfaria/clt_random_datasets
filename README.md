# clt\_random\_datasets

This ROS package generates random datasets in a simulation environment, for multiple robots and targets with random motion.

## Brief Description

You need to test your localization/tracking algorithm for more robots than you have access to, so you desire a simulation environment. However, many of the existing simulators are time-consuming to setup. This package provides a simple to use simulation environment, which displays internal states in rviz-friendly formats for visualization, as well as custom msg formats with more information.

Due to using multiple ROS nodes, it scales very well across CPU cores. The computational complexity is O(m n^2), where **m** is the number of targets, and **n** is the number of robots. This is because each robot checks for collisions with all other robots.

## Modules

The following modules are available in this dataset generator:

* **Robot module** - provides odometry, landmark/target observations, checks collisions, etc. It is the main module.
* **Ball(Target) module** - Simulates a ball with gravity, hovering, pulling, and collisions with robots, as if a person would be holding the ball by a string and moving it around
* **Walls module** - Sets up a field with 4 walls surrounding it
* **Landmarks module** - Loads a configuration file and sets up static landmarks in the stage. Publishes a ROS topic with the landmarks positions

## Requisites

* [ROS](https://www.ros.org). Tested with ROS Indigo and Kinetic.
* [rviz](https://wiki.ros.org/rviz) for visualization.
* [clt_msgs](https://github.com/pmvfaria/clt_msgs)

## How to Use

To generate sample data, follow these steps:

1. Clone and build this package and *clt\_msgs* package:
`git clone https://github.com/pmvfaria/clt_msgs`
`git clone https://github.com/pmvfaria/clt_random_datasets && catkin_make`
2. Execute one of the following scripts with `python <script>`:
  * If you don't need recording to a rosbag, use the *config/**create\_launch\_file*** script, followed by the number of robots you desire
    * This script creates the new.launch file in the launch directory
    * You can specify optional parameters. Use the --help option to learn about these
    
  * If you need recording to a rosbag, use the *config/**create\_launch\_record*** script, followed by the number of robots you desire
    * This script creates the new.launch file in the launch directory
    * You should use the --help option to learn how to specify the many optional parameters
3. Run: `roslaunch clt_random_datasets new.launch` and the simulation will begin
4. (optional) if you want to visualize the simulation, run rviz and load the configuration *rviz.rviz* in the config directory (up to 10 robots, but customizable)

## Contribute

The simulator is designed in modules, so you are welcome to make your own modules, suggest modifications, add more customizability.

At the moment, the target simulation is quite rough, and contributions to this module are appreciated.

## Using with PF-UCLT

After recording to a rosbag file, you can use [pfuclt](https://github.com/pmvfaria/pfuclt) to try localization and target tracking.

To accomplish this, when executing the create\_launch\_file script, it will generate a file in your the pfuclt package launch folder to work with it. By default, it will be named new.launch, so you can execute it with: `roslaunch pfuclt new.launch`

The dataset generation is very CPU intensive for a higher number of robots, therefore it is advised to use the create\_launch\_record script and appropriate optional parameters to generate a rosbag of the whole dataset. The generated launch file will automatically record the dataset. The generated launch file for PF-UCLT will automatically launch the rosbag player when it is used.

If necessary when performing PF-UCLT+, use the rate argument of the launch file to set different rosbag playing rates.

Use the rviz config file pfuclt/config/pfuclt_datasets.rviz to visualize the algorithm in rviz. **Important**: you need to execute rviz only after setting the /use_sim_time parameter to true, which is done by calling the launch file above.

You will know everything is working if the PF-UCLT node outputs an odometry frequency of 33Hz (by default) almost consistently. If it goes below this value, the algorithm is causing odometry readings to be delayed, so it can't keep up and you should lower the rosbag playing rate.
