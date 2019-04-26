cmake_minimum_required(VERSION 2.8.3)
project(biotac_force_net)

find_package(catkin REQUIRED COMPONENTS
  biotac_sensors
  geometry_msgs
  rospy
  message_generation
  )
add_service_files(
  FILES
  RecordData.srv
  StoreData.srv
 )
 generate_messages(
   DEPENDENCIES   std_msgs
 )


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES biotac_force_net
#  CATKIN_DEPENDS biotac_sensors geometry_msgs rospy
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
