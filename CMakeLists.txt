cmake_minimum_required(VERSION 2.8.3)
project(clt_random_datasets)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  message_generation
  tf
)

find_package(clt_msgs REQUIRED)

catkin_python_setup()

add_message_files(
    DIRECTORY msg
    FILES CustomOdometry.msg RobotObservation.msg RobotObservationArray.msg
)

add_service_files(
    DIRECTORY srv
    FILES SendString.srv
)

generate_messages(
    DEPENDENCIES
    std_msgs
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(
	PROGRAMS
		scripts/robot
		scripts/ball
		scripts/odometry
		scripts/world
		scripts/omni_custom
	DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

