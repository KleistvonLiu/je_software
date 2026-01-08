cmake_minimum_required(VERSION 3.8)
project(je_software LANGUAGES C CXX)

# ----------------------------
# Build type (single-config generators like Ninja/Make)
# ----------------------------
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

# ----------------------------
# Language standard (package-wide)
# ----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ----------------------------
# Warnings
# ----------------------------
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ----------------------------
# Performance options (apply to all targets in this package)
# ----------------------------
option(JE_ENABLE_LTO    "Enable -flto for Release/RelWithDebInfo" OFF)
option(JE_ENABLE_NATIVE "Enable -march=native/-mtune=native for Release/RelWithDebInfo" OFF)

set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O2 -g -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE        "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "${CMAKE_CXX_FLAGS_MINSIZEREL} -Os -DNDEBUG")

set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -ffunction-sections -fdata-sections")
set(CMAKE_CXX_FLAGS_RELEASE        "${CMAKE_CXX_FLAGS_RELEASE} -ffunction-sections -fdata-sections")
set(CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO} -Wl,--gc-sections -Wl,--as-needed")
set(CMAKE_SHARED_LINKER_FLAGS_RELEASE        "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} -Wl,--gc-sections -Wl,--as-needed")
set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO    "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} -Wl,--gc-sections -Wl,--as-needed")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE           "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -Wl,--gc-sections -Wl,--as-needed")

if(JE_ENABLE_LTO)
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -flto")
  set(CMAKE_CXX_FLAGS_RELEASE        "${CMAKE_CXX_FLAGS_RELEASE} -flto")
  set(CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO} -flto")
  set(CMAKE_SHARED_LINKER_FLAGS_RELEASE        "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} -flto")
  set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO    "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} -flto")
  set(CMAKE_EXE_LINKER_FLAGS_RELEASE           "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -flto")
endif()

if(JE_ENABLE_NATIVE)
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -march=native -mtune=native")
  set(CMAKE_CXX_FLAGS_RELEASE        "${CMAKE_CXX_FLAGS_RELEASE} -march=native -mtune=native")
endif()

# ----------------------------
# Dependencies
# ----------------------------
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

find_package(rosidl_default_generators REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED IMPORTED_TARGET libzmq)

find_package(nlohmann_json REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# ----------------------------
# Python install (Manual, to avoid target collision with rosidl)
# ----------------------------
# We manually install the python package because ament_python_install_package(${PROJECT_NAME})
# conflicts with rosidl_generate_interfaces(${PROJECT_NAME} ...) which creates the same targets.

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_path('purelib', vars={'base': ''}).lstrip('/'))"
  OUTPUT_VARIABLE PYTHON_INSTALL_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

install(DIRECTORY ${PROJECT_NAME}
  DESTINATION ${PYTHON_INSTALL_DIR}
)

install(PROGRAMS
  scripts/agilex_robot
  scripts/camera_node
  scripts/manager_node
  scripts/replay_node
  scripts/tactile_sensor
  scripts/test_publisher
  DESTINATION lib/${PROJECT_NAME}
)

# ----------------------------
# ROS interfaces
# ----------------------------
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/EndEffectorCommand.msg"
  SKIP_GROUP_MEMBERSHIP_CHECK
)

# Humble recommended: get typesupport target and link it explicitly
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")


# ----------------------------
# Targets
# ----------------------------

add_executable(je_robot_node
  je_software/je_robot_node.cpp
)
ament_target_dependencies(je_robot_node
  rclcpp
  sensor_msgs
  geometry_msgs
)
# IMPORTANT: use plain signature consistently (ament_target_dependencies uses plain internally)
target_link_libraries(je_robot_node
  PkgConfig::ZMQ
  nlohmann_json::nlohmann_json
  Eigen3::Eigen
  ${cpp_typesupport_target}
)

add_executable(cli_target_publisher
  je_software/je_cli_target_publisher.cpp
)
ament_target_dependencies(cli_target_publisher
  rclcpp
  sensor_msgs
  geometry_msgs
)
# If cli_target_publisher includes/uses EndEffectorCommand.msg, uncomment:
# target_link_libraries(cli_target_publisher
#   ${cpp_typesupport_target}
# )

add_executable(end_effector_cli
  je_software/je_end_effector_cli.cpp
)
ament_target_dependencies(end_effector_cli
  rclcpp
  geometry_msgs
)
target_link_libraries(end_effector_cli
  ${cpp_typesupport_target}
)

# ----------------------------
# Install
# ----------------------------
install(TARGETS
  je_robot_node
  cli_target_publisher
  end_effector_cli
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
