cmake_minimum_required(VERSION 3.16)
project(ros2_ik_cpp)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(pinocchio REQUIRED)
find_package(urdfdom_headers REQUIRED)

add_executable(ik_node src/ik_node.cpp)
ament_target_dependencies(ik_node rclcpp sensor_msgs geometry_msgs pinocchio)

# Link Eigen and pinocchio targets to ensure include directories and libraries are available
# Use plain signature to match previous usage by ament_target_dependencies
target_link_libraries(ik_node Eigen3::Eigen pinocchio::pinocchio)

install(TARGETS ik_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
