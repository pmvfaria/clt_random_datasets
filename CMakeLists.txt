cmake_minimum_required(VERSION 2.8.3)
project(clt_random_datasets)

find_package(catkin REQUIRED COMPONENTS
  rospy
  tf
)

find_package(clt_msgs REQUIRED)

catkin_python_setup()

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(
	PROGRAMS
		scripts/robot
		scripts/ball
		scripts/world
	DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
