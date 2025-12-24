cmake_minimum_required(VERSION 3.8)
project(je_software)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED IMPORTED_TARGET libzmq)

find_package(nlohmann_json REQUIRED)
find_package(Eigen3 REQUIRED)

ament_python_install_package(${PROJECT_NAME})
############# je_robot_node ############# 
add_executable(je_robot_node je_software/je_robot_node.cpp)
target_compile_features(je_robot_node PUBLIC cxx_std_17)
ament_target_dependencies(je_robot_node rclcpp sensor_msgs geometry_msgs)
target_link_libraries(je_robot_node PkgConfig::ZMQ nlohmann_json::nlohmann_json Eigen3::Eigen)

install(TARGETS je_robot_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)
#######################################

############# cli_target_publisher #############
add_executable(cli_target_publisher je_software/je_cli_target_publisher.cpp)
ament_target_dependencies(cli_target_publisher rclcpp sensor_msgs geometry_msgs)

install(TARGETS cli_target_publisher
  DESTINATION lib/${PROJECT_NAME}
)
#######################################

ament_package()




