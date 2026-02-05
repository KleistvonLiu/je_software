# 
# 默认（RelWithDebInfo，无 LTO，无 native）
# colcon build --packages-select je_software

# Release + 开 LTO
# colcon build --packages-select je_software --cmake-args \
#   -DCMAKE_BUILD_TYPE=Release \
#   -DJE_ENABLE_LTO=ON

# RelWithDebInfo + native（仅建议本机部署）
# colcon build --packages-select je_software --cmake-args \
#   -DCMAKE_BUILD_TYPE=RelWithDebInfo \
#   -DJE_ENABLE_NATIVE=ON

# 
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
option(JE_USE_CONDA_INCLUDES "Append CONDA_PREFIX/include to extra include dirs" OFF)

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
find_package(common REQUIRED)

find_package(rosidl_default_generators REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED IMPORTED_TARGET libzmq)

find_path(JE_CPPZMQ_INCLUDE_DIR zmq.hpp
  PATH_SUFFIXES include
  PATHS ${CMAKE_PREFIX_PATH} /usr
)

set(JE_ZMQ_INCLUDE_DIRS "${ZMQ_INCLUDE_DIRS}" "${JE_CPPZMQ_INCLUDE_DIR}")
if(DEFINED ENV{CONDA_PREFIX})
  list(APPEND JE_ZMQ_INCLUDE_DIRS "$ENV{CONDA_PREFIX}/include")
endif()
list(REMOVE_DUPLICATES JE_ZMQ_INCLUDE_DIRS)

find_package(nlohmann_json REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(pinocchio REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Prefer system spdlog/fmt to match ROS rcl_logging_spdlog and avoid ABI conflicts
set(_JE_SYSTEM_SPDLOG_DIR "/usr/lib/x86_64-linux-gnu/cmake/spdlog")
set(_JE_SYSTEM_FMT_DIR "/usr/lib/x86_64-linux-gnu/cmake/fmt")
if(EXISTS "${_JE_SYSTEM_SPDLOG_DIR}/spdlogConfig.cmake")
  find_package(fmt REQUIRED CONFIG PATHS "${_JE_SYSTEM_FMT_DIR}" NO_DEFAULT_PATH)
  find_package(spdlog REQUIRED CONFIG PATHS "${_JE_SYSTEM_SPDLOG_DIR}" NO_DEFAULT_PATH)
else()
  find_package(fmt REQUIRED)
  find_package(spdlog REQUIRED)
endif()

set(JE_EXTRA_INCLUDE_DIRS "")
if(JE_USE_CONDA_INCLUDES AND DEFINED ENV{CONDA_PREFIX})
  list(APPEND JE_EXTRA_INCLUDE_DIRS "$ENV{CONDA_PREFIX}/include")
endif()
if(CMAKE_PREFIX_PATH)
  foreach(_je_prefix IN LISTS CMAKE_PREFIX_PATH)
    if(IS_DIRECTORY "${_je_prefix}/include")
      list(APPEND JE_EXTRA_INCLUDE_DIRS "${_je_prefix}/include")
    endif()
  endforeach()
endif()
if(TARGET pinocchio::pinocchio)
  get_target_property(_PINOCCHIO_INCLUDES pinocchio::pinocchio INTERFACE_INCLUDE_DIRECTORIES)
  if(_PINOCCHIO_INCLUDES)
    list(APPEND JE_EXTRA_INCLUDE_DIRS ${_PINOCCHIO_INCLUDES})
  endif()
endif()
if(TARGET nlohmann_json::nlohmann_json)
  get_target_property(_NLOHMANN_INCLUDES nlohmann_json::nlohmann_json INTERFACE_INCLUDE_DIRECTORIES)
  if(_NLOHMANN_INCLUDES)
    list(APPEND JE_EXTRA_INCLUDE_DIRS ${_NLOHMANN_INCLUDES})
  endif()
endif()

find_path(JE_PINOCCHIO_INCLUDE_DIR pinocchio/multibody/model.hpp
  PATH_SUFFIXES include
  PATHS ${CMAKE_PREFIX_PATH}
)
if(JE_PINOCCHIO_INCLUDE_DIR)
  list(APPEND JE_EXTRA_INCLUDE_DIRS ${JE_PINOCCHIO_INCLUDE_DIR})
endif()

find_path(JE_NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp
  PATH_SUFFIXES include
  PATHS ${CMAKE_PREFIX_PATH}
)
if(JE_NLOHMANN_JSON_INCLUDE_DIR)
  list(APPEND JE_EXTRA_INCLUDE_DIRS ${JE_NLOHMANN_JSON_INCLUDE_DIR})
endif()

if(JE_CPPZMQ_INCLUDE_DIR)
  list(APPEND JE_EXTRA_INCLUDE_DIRS ${JE_CPPZMQ_INCLUDE_DIR})
endif()

list(REMOVE_DUPLICATES JE_EXTRA_INCLUDE_DIRS)

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
  scripts/dynamixel_init_joint_state
  scripts/plot_dynamixel_positions
  scripts/joint_rate_monitor
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
  "msg/EndEffectorCommandLR.msg"
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
  common
)
target_include_directories(je_robot_node
  PRIVATE
    ${ZMQ_INCLUDE_DIRS}
    ${JE_CPPZMQ_INCLUDE_DIR}
    ${JE_EXTRA_INCLUDE_DIRS}
)
if(JE_CPPZMQ_INCLUDE_DIR)
  target_compile_options(je_robot_node PRIVATE "-I${JE_CPPZMQ_INCLUDE_DIR}")
endif()
# IMPORTANT: use plain signature consistently (ament_target_dependencies uses plain internally)
target_link_libraries(je_robot_node
  PkgConfig::ZMQ
  nlohmann_json::nlohmann_json
  Eigen3::Eigen
  ik_solver
  spdlog::spdlog
  fmt::fmt
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
target_link_libraries(cli_target_publisher
  spdlog::spdlog
  fmt::fmt
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
  common
)
target_link_libraries(end_effector_cli
  ${cpp_typesupport_target}
  spdlog::spdlog
  fmt::fmt
)

add_executable(jsonl_replayer_node
  je_software/jsonl_replayer_node.cpp
)
ament_target_dependencies(jsonl_replayer_node
  rclcpp
  sensor_msgs
  geometry_msgs
  common
)
target_include_directories(jsonl_replayer_node
  PRIVATE
    ${common_INCLUDE_DIRS}
)
target_link_libraries(jsonl_replayer_node
  nlohmann_json::nlohmann_json
  spdlog::spdlog
  fmt::fmt
)

# ik_solver library (from utils)
add_library(ik_solver STATIC
  utils/src/ik_solver.cpp
)

target_include_directories(ik_solver
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/utils/include
         ${JE_EXTRA_INCLUDE_DIRS}
)

target_compile_features(ik_solver PUBLIC cxx_std_17)

# Link Pinocchio and Eigen (support both modern imported target and legacy variable)
if (TARGET pinocchio::pinocchio)
  target_link_libraries(ik_solver PUBLIC pinocchio::pinocchio Eigen3::Eigen)
else()
  target_link_libraries(ik_solver PUBLIC ${pinocchio_LIBRARIES} Eigen3::Eigen)
endif()

# Install library and headers so local builds can find them
install(TARGETS ik_solver ARCHIVE DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/utils/include/ DESTINATION include)

# ----------------------------
# Install
# ----------------------------
install(TARGETS
  je_robot_node
  cli_target_publisher
  end_effector_cli
  jsonl_replayer_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
