#include "je_zmq_system_hardware.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <string>
#include <thread>
#include <utility>

#include "hardware_interface/handle.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "rclcpp/rclcpp.hpp"

namespace je_software
{

using json = nlohmann::json;
using namespace std::chrono_literals;

hardware_interface::CallbackReturn JeZmqSystemHardware::on_init(
  const hardware_interface::HardwareInfo & info)
{
  if (hardware_interface::SystemInterface::on_init(info) != hardware_interface::CallbackReturn::SUCCESS)
  {
    return hardware_interface::CallbackReturn::ERROR;
  }

  joint_names_.clear();
  joint_names_.reserve(info_.joints.size());
  joint_name_to_index_.clear();

  for (const auto & joint : info_.joints)
  {
    if (joint.command_interfaces.size() != 1 ||
      joint.command_interfaces[0].name != hardware_interface::HW_IF_POSITION)
    {
      RCLCPP_FATAL(
        rclcpp::get_logger("JeZmqSystemHardware"),
        "Joint '%s' must expose exactly one command interface: position",
        joint.name.c_str());
      return hardware_interface::CallbackReturn::ERROR;
    }

    if (joint.state_interfaces.empty())
    {
      RCLCPP_FATAL(
        rclcpp::get_logger("JeZmqSystemHardware"),
        "Joint '%s' must expose at least one state interface",
        joint.name.c_str());
      return hardware_interface::CallbackReturn::ERROR;
    }

    joint_name_to_index_[joint.name] = joint_names_.size();
    joint_names_.push_back(joint.name);
  }

  const auto n = joint_names_.size();
  hw_positions_.assign(n, std::numeric_limits<double>::quiet_NaN());
  hw_velocities_.assign(n, 0.0);
  hw_efforts_.assign(n, 0.0);
  hw_commands_.assign(n, std::numeric_limits<double>::quiet_NaN());

  latest_positions_.assign(n, 0.0);
  latest_velocities_.assign(n, 0.0);
  latest_efforts_.assign(n, 0.0);

  auto get_string_param = [&](const std::string & key, std::string & target)
  {
    auto it = info_.hardware_parameters.find(key);
    if (it != info_.hardware_parameters.end())
    {
      target = it->second;
    }
  };

  auto get_int_param = [&](const std::string & key, int & target)
  {
    auto it = info_.hardware_parameters.find(key);
    if (it != info_.hardware_parameters.end())
    {
      try
      {
        target = std::stoi(it->second);
      }
      catch (...) {}
    }
  };

  auto get_double_param = [&](const std::string & key, double & target)
  {
    auto it = info_.hardware_parameters.find(key);
    if (it != info_.hardware_parameters.end())
    {
      try
      {
        target = std::stod(it->second);
      }
      catch (...) {}
    }
  };

  get_string_param("robot_ip", robot_ip_);
  get_int_param("pub_port", pub_port_);
  get_int_param("sub_port", sub_port_);
  get_double_param("command_dt", command_dt_);
  get_string_param("gripper_sub_topic", gripper_sub_topic_);

  RCLCPP_INFO(
    rclcpp::get_logger("JeZmqSystemHardware"),
    "Configured JE hardware transport: robot_ip=%s, pub_port=%d, sub_port=%d, command_dt=%.4f, gripper_sub_topic=%s",
    robot_ip_.c_str(), pub_port_, sub_port_, command_dt_, gripper_sub_topic_.c_str());

  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn JeZmqSystemHardware::on_configure(
  const rclcpp_lifecycle::State &)
{
  std::fill(hw_positions_.begin(), hw_positions_.end(), 0.0);
  std::fill(hw_velocities_.begin(), hw_velocities_.end(), 0.0);
  std::fill(hw_efforts_.begin(), hw_efforts_.end(), 0.0);
  std::fill(hw_commands_.begin(), hw_commands_.end(), 0.0);

  std::fill(latest_positions_.begin(), latest_positions_.end(), 0.0);
  std::fill(latest_velocities_.begin(), latest_velocities_.end(), 0.0);
  std::fill(latest_efforts_.begin(), latest_efforts_.end(), 0.0);
  has_valid_state_ = false;

  try
  {
    context_ = std::make_unique<zmq::context_t>(1);
    publisher_ = std::make_unique<zmq::socket_t>(*context_, zmq::socket_type::pub);
    subscriber_ = std::make_unique<zmq::socket_t>(*context_, zmq::socket_type::sub);

    // Hardware plugin as publisher: binds on pub_port to send commands
    // Hardware plugin as subscriber: connects to robot's state port to receive state
    const std::string pub_bind_addr = "tcp://*:" + std::to_string(pub_port_);
    const std::string sub_connect_addr = "tcp://" + robot_ip_ + ":" + std::to_string(sub_port_);

    publisher_->set(zmq::sockopt::sndhwm, 0);
    publisher_->set(zmq::sockopt::immediate, 1);
    publisher_->bind(pub_bind_addr);

    subscriber_->set(zmq::sockopt::subscribe, "State ");
    subscriber_->set(zmq::sockopt::conflate, 1);
    subscriber_->connect(sub_connect_addr);

    RCLCPP_INFO(
      rclcpp::get_logger("JeZmqSystemHardware"),
      "ZMQ configured: pub_bind=%s, sub_connect=%s",
      pub_bind_addr.c_str(), sub_connect_addr.c_str());
  }
  catch (const std::exception & ex)
  {
    RCLCPP_ERROR(
      rclcpp::get_logger("JeZmqSystemHardware"),
      "Failed to configure ZMQ transport: %s", ex.what());
    return hardware_interface::CallbackReturn::ERROR;
  }

  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn JeZmqSystemHardware::on_activate(
  const rclcpp_lifecycle::State &)
{
  global_time_ = 0.0;

  try
  {
    gripper_node_ = std::make_shared<rclcpp::Node>("je_zmq_system_hardware_gripper_bridge");
    sub_gripper_cmd_ = gripper_node_->create_subscription<je_software::msg::EndEffectorCommandLR>(
      gripper_sub_topic_,
      rclcpp::QoS(10).reliable(),
      std::bind(&JeZmqSystemHardware::gripper_cmd_callback, this, std::placeholders::_1));

    gripper_executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
    gripper_executor_->add_node(gripper_node_);
    gripper_thread_running_.store(true);
    gripper_thread_ = std::thread(&JeZmqSystemHardware::gripper_spin_loop, this);

    RCLCPP_INFO(
      rclcpp::get_logger("JeZmqSystemHardware"),
      "Subscribed gripper command topic: %s",
      gripper_sub_topic_.c_str());
  }
  catch (const std::exception & ex)
  {
    RCLCPP_ERROR(
      rclcpp::get_logger("JeZmqSystemHardware"),
      "Failed to initialize gripper subscriber: %s", ex.what());
    return hardware_interface::CallbackReturn::ERROR;
  }

  // Initialize ZMQ connection before starting state thread
  // Add a small delay to allow ZMQ sockets to fully bind/connect
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  try
  {
    robot_switch(true);
  }
  catch (const std::exception & ex)
  {
    RCLCPP_WARN(
      rclcpp::get_logger("JeZmqSystemHardware"),
      "Failed to send switch command on activate: %s", ex.what());
  }

  // Small delay after switch command to let robot backend process
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  state_thread_running_.store(true);
  state_thread_ = std::thread(&JeZmqSystemHardware::state_receive_loop, this);

  // Wait for first valid state before considering hardware ready
  // Increased timeout to 15 seconds to allow robot backend to establish connection
  int wait_count = 0;
  int max_wait = 1500;  // 15 seconds at 10ms intervals
  while (!has_valid_state_ && wait_count < max_wait)
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    wait_count++;
    
    // Log progress every 100 iterations (1 second)
    if (wait_count % 100 == 0)
    {
      RCLCPP_INFO(
        rclcpp::get_logger("JeZmqSystemHardware"),
        "Waiting for initial state from ZMQ hardware... (%d/%d seconds)",
        wait_count / 100, max_wait / 100);
    }
  }

  if (!has_valid_state_)
  {
    RCLCPP_ERROR(
      rclcpp::get_logger("JeZmqSystemHardware"),
      "Timeout waiting for initial state from ZMQ hardware after %d seconds",
      max_wait / 100);
    return hardware_interface::CallbackReturn::ERROR;
  }

  // Now initialize hw_positions and hw_commands with the actual state
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    hw_positions_ = latest_positions_;
  }
  hw_commands_ = hw_positions_;

  RCLCPP_INFO(
    rclcpp::get_logger("JeZmqSystemHardware"),
    "Hardware activated. Initial positions: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]",
    hw_positions_[0], hw_positions_[1], hw_positions_[2], hw_positions_[3],
    hw_positions_[4], hw_positions_[5], hw_positions_[6]);

  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn JeZmqSystemHardware::on_deactivate(
  const rclcpp_lifecycle::State &)
{
  gripper_thread_running_.store(false);
  if (gripper_executor_)
  {
    gripper_executor_->cancel();
  }
  if (gripper_thread_.joinable())
  {
    gripper_thread_.join();
  }
  if (gripper_executor_ && gripper_node_)
  {
    gripper_executor_->remove_node(gripper_node_);
  }
  sub_gripper_cmd_.reset();
  gripper_executor_.reset();
  gripper_node_.reset();

  state_thread_running_.store(false);

  if (context_)
  {
    try
    {
      context_->shutdown();
    }
    catch (...) {}
  }

  if (state_thread_.joinable())
  {
    state_thread_.join();
  }

  try
  {
    robot_switch(false);
  }
  catch (...) {}

  subscriber_.reset();
  publisher_.reset();
  context_.reset();

  return hardware_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface> JeZmqSystemHardware::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> state_interfaces;
  state_interfaces.reserve(joint_names_.size() * 3);

  for (size_t i = 0; i < joint_names_.size(); ++i)
  {
    state_interfaces.emplace_back(
      joint_names_[i], hardware_interface::HW_IF_POSITION, &hw_positions_[i]);
    state_interfaces.emplace_back(
      joint_names_[i], hardware_interface::HW_IF_VELOCITY, &hw_velocities_[i]);
    state_interfaces.emplace_back(
      joint_names_[i], hardware_interface::HW_IF_EFFORT, &hw_efforts_[i]);
  }

  RCLCPP_INFO(
    rclcpp::get_logger("JeZmqSystemHardware"),
    "[export_state_interfaces] Exported %zu state interfaces for %zu joints",
    state_interfaces.size(), joint_names_.size());

  return state_interfaces;
}

std::vector<hardware_interface::CommandInterface> JeZmqSystemHardware::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> command_interfaces;
  command_interfaces.reserve(joint_names_.size());

  for (size_t i = 0; i < joint_names_.size(); ++i)
  {
    command_interfaces.emplace_back(
      joint_names_[i], hardware_interface::HW_IF_POSITION, &hw_commands_[i]);
  }

  return command_interfaces;
}

hardware_interface::return_type JeZmqSystemHardware::read(
  const rclcpp::Time &,
  const rclcpp::Duration &)
{
  std::lock_guard<std::mutex> lock(state_mutex_);
  static int read_count = 0;
  static bool first_valid = true;
  
  if (has_valid_state_)
  {
    // Copy from ZMQ state to hardware interfaces
    for (size_t i = 0; i < hw_positions_.size(); ++i)
    {
      hw_positions_[i] = latest_positions_[i];
      hw_velocities_[i] = latest_velocities_[i];
      hw_efforts_[i] = latest_efforts_[i];
    }
    
    if (first_valid)
    {
      RCLCPP_INFO(
        rclcpp::get_logger("JeZmqSystemHardware"),
        "[read] First valid state received - hardware interfaces active");
      first_valid = false;
    }
    
    // Log less frequently (every 2000 reads ≈ 20 seconds at 100Hz)
    if (++read_count % 2000 == 0)
    {
      RCLCPP_DEBUG(
        rclcpp::get_logger("JeZmqSystemHardware"),
        "[read] Read #%d: positions=[%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
        read_count,
        hw_positions_[0], hw_positions_[1], hw_positions_[2], hw_positions_[3],
        hw_positions_[4], hw_positions_[5], hw_positions_[6]);
    }
  }
  else
  {
    static bool warned = false;
    if (!warned)
    {
      RCLCPP_WARN(
        rclcpp::get_logger("JeZmqSystemHardware"),
        "[read] Waiting for valid ZMQ state from hardware...");
      warned = true;
    }
  }
  
  return hardware_interface::return_type::OK;
}

hardware_interface::return_type JeZmqSystemHardware::write(
  const rclcpp::Time &,
  const rclcpp::Duration &)
{
  if (!publisher_)
  {
    return hardware_interface::return_type::ERROR;
  }

  global_time_ += command_dt_;

  json payload;
  payload["Robot0"]["time"] = global_time_;
  payload["Robot0"]["joint"] = hw_commands_;  // Use lowercase "joint" to match je_robot_node.cpp

  append_gripper_from_command(payload, 0);

  try
  {
    // Match je_robot_node.cpp format: prefix "Joint " + JSON payload
    const std::string message = "Joint " + payload.dump();
    // Send with default blocking behavior to ensure delivery
    auto result = publisher_->send(zmq::buffer(message));
    if (!result)
    {
      RCLCPP_WARN(
        rclcpp::get_logger("JeZmqSystemHardware"),
        "[write] Failed to send message (result is false)");
    }
    
    // Log sent commands less frequently (every 500 writes to avoid log spam)
    static int write_count = 0;
    if (++write_count % 500 == 0)
    {
      RCLCPP_INFO(
        rclcpp::get_logger("JeZmqSystemHardware"),
        "[write] Write #%d: Sent message (length=%zu): %s",
        write_count, message.length(), message.c_str());
    }
  }
  catch (const std::exception & ex)
  {
    RCLCPP_ERROR(
      rclcpp::get_logger("JeZmqSystemHardware"),
      "Failed to send joint command: %s", ex.what());
    return hardware_interface::return_type::ERROR;
  }

  return hardware_interface::return_type::OK;
}

void JeZmqSystemHardware::state_receive_loop()
{
  RCLCPP_INFO(rclcpp::get_logger("JeZmqSystemHardware"), "state_receive_loop started");
  int msg_count = 0;
  bool first_message_logged = false;
  
  while (state_thread_running_.load())
  {
    if (!subscriber_)
    {
      break;
    }

    zmq::message_t msg;
    try
    {
      auto res = subscriber_->recv(msg, zmq::recv_flags::none);
      if (!res)
      {
        continue;
      }
    }
    catch (const zmq::error_t & ex)
    {
      if (ex.num() == ETERM || ex.num() == EAGAIN)
      {
        continue;
      }
      RCLCPP_ERROR(
        rclcpp::get_logger("JeZmqSystemHardware"),
        "ZMQ receive error: %s", ex.what());
      continue;
    }
    catch (const std::exception & ex)
    {
      RCLCPP_ERROR(
        rclcpp::get_logger("JeZmqSystemHardware"),
        "State receive exception: %s", ex.what());
      continue;
    }

    try
    {
      const std::string state_message(static_cast<char *>(msg.data()), msg.size());
      const auto pos = state_message.find(' ');
      if (pos == std::string::npos)
      {
        continue;
      }

      const std::string topic = state_message.substr(0, pos);
      if (topic != "State")
      {
        continue;
      }

      const auto state_json = json::parse(state_message.substr(pos + 1));

  // Log first message received for diagnostics
      if (!first_message_logged)
      {
        RCLCPP_INFO(
          rclcpp::get_logger("JeZmqSystemHardware"),
          "[state_receive_loop] FIRST MESSAGE RECEIVED. Message size: %zu bytes. Full JSON:\n%s",
          state_message.size(),
          state_json.dump(2).c_str());
        first_message_logged = true;
      }

      // Log every 500 messages with message size and raw data to detect if messages are changing
      static int size_log_count = 0;
      if (++size_log_count % 500 == 0)
      {
        RCLCPP_INFO(
          rclcpp::get_logger("JeZmqSystemHardware"),
          "[state_receive_loop] Message #%d: size=%zu bytes. Payload: %s",
          size_log_count,
          state_message.size(),
          state_message.substr(0, std::min(size_t(200), state_message.size())).c_str());
      }

      std::vector<double> positions(joint_names_.size(), 0.0);
      std::vector<double> velocities(joint_names_.size(), 0.0);
      std::vector<double> efforts(joint_names_.size(), 0.0);

      if (!parse_joint_state_from_json(state_json, positions, velocities, efforts))
      {
  // Log failed parsing every 500 messages to help diagnose format issues
        static int parse_fail_count = 0;
  if (++parse_fail_count % 500 == 0)
        {
          // Dump the entire JSON to see what we're actually receiving
          RCLCPP_WARN(
            rclcpp::get_logger("JeZmqSystemHardware"),
            "[state_receive_loop] Failed to parse state JSON (fail count=%d). Full message:\n%s",
            parse_fail_count,
            state_message.c_str());
        }
        continue;
      }

      msg_count++;
      if (msg_count % 500 == 0)  // Log every 500 messages to reduce spam
      {
        RCLCPP_INFO(
          rclcpp::get_logger("JeZmqSystemHardware"),
          "[state_receive_loop] Received %d messages. Parsed positions: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]",
          msg_count,
          positions[0], positions[1], positions[2], positions[3], 
          positions[4], positions[5], positions[6]);
      }

      {
        std::lock_guard<std::mutex> lock(state_mutex_);
  // Log before updating
  if (msg_count % 500 == 0)
        {
          RCLCPP_INFO(
            rclcpp::get_logger("JeZmqSystemHardware"),
            "[state_receive_loop] Updating latest_positions. Before: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]",
            latest_positions_[0], latest_positions_[1], latest_positions_[2], latest_positions_[3],
            latest_positions_[4], latest_positions_[5], latest_positions_[6]);
        }
        
        latest_positions_ = std::move(positions);
        latest_velocities_ = std::move(velocities);
        latest_efforts_ = std::move(efforts);
        has_valid_state_ = true;
        
  // Log after updating
  if (msg_count % 500 == 0)
        {
          RCLCPP_INFO(
            rclcpp::get_logger("JeZmqSystemHardware"),
            "[state_receive_loop] Updated latest_positions. After: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]",
            latest_positions_[0], latest_positions_[1], latest_positions_[2], latest_positions_[3],
            latest_positions_[4], latest_positions_[5], latest_positions_[6]);
        }
      }
    }
    catch (const std::exception & ex)
    {
      RCLCPP_ERROR(
        rclcpp::get_logger("JeZmqSystemHardware"),
        "Failed parsing state JSON: %s", ex.what());
    }
  }
  
  RCLCPP_INFO(rclcpp::get_logger("JeZmqSystemHardware"), "state_receive_loop stopped (total messages: %d)", msg_count);
}

void JeZmqSystemHardware::gripper_spin_loop()
{
  while (gripper_thread_running_.load())
  {
    if (!gripper_executor_)
    {
      break;
    }
    gripper_executor_->spin_once(50ms);
  }
}

void JeZmqSystemHardware::handle_gripper_cmd(
  const je_software::msg::EndEffectorCommand & msg,
  int robot_index)
{
  if (msg.mode != je_software::msg::EndEffectorCommand::MODE_POSITION &&
    msg.mode != je_software::msg::EndEffectorCommand::MODE_PRESET &&
    msg.mode != je_software::msg::EndEffectorCommand::MODE_TORQUE)
  {
    RCLCPP_WARN(
      rclcpp::get_logger("JeZmqSystemHardware"),
      "Invalid end effector mode: %d", msg.mode);
    return;
  }

  std::string command = msg.command;
  std::transform(command.begin(), command.end(), command.begin(), ::tolower);

  if (msg.mode == je_software::msg::EndEffectorCommand::MODE_TORQUE)
  {
    if (
      command != je_software::msg::EndEffectorCommand::CMD_OPEN &&
      command != je_software::msg::EndEffectorCommand::CMD_CLOSE)
    {
      RCLCPP_WARN(
        rclcpp::get_logger("JeZmqSystemHardware"),
        "Invalid torque gripper command: '%s'", command.c_str());
      return;
    }

    if (!std::isfinite(msg.torque) || msg.torque <= 0.0)
    {
      RCLCPP_WARN(
        rclcpp::get_logger("JeZmqSystemHardware"),
        "Invalid torque value: %.4f", msg.torque);
      return;
    }
  }

  std::lock_guard<std::mutex> lock(gripper_mutex_);
  auto & target = (robot_index == 1) ? gripper_cmd_right_ : gripper_cmd_left_;
  target.mode = msg.mode;
  target.position = msg.position;
  target.preset = msg.preset;
  target.command = command;
  target.torque = msg.torque;
  target.received = true;
}

void JeZmqSystemHardware::gripper_cmd_callback(
  const je_software::msg::EndEffectorCommandLR::SharedPtr msg)
{
  if (!msg)
  {
    return;
  }

  if (msg->left_valid)
  {
    handle_gripper_cmd(msg->left, 0);
  }
  if (msg->right_valid)
  {
    handle_gripper_cmd(msg->right, 1);
  }
}

const JeZmqSystemHardware::GripperCommand & JeZmqSystemHardware::get_gripper_command(int robot_index) const
{
  return (robot_index == 1) ? gripper_cmd_right_ : gripper_cmd_left_;
}

void JeZmqSystemHardware::append_gripper_from_command(nlohmann::json & payload, int robot_index) const
{
  std::lock_guard<std::mutex> lock(gripper_mutex_);
  const auto & gripper_cmd = get_gripper_command(robot_index);
  if (!gripper_cmd.received)
  {
    return;
  }

  auto & robot = payload[(robot_index == 1) ? "Robot1" : "Robot0"];
  
  // 扭矩模式: 放在 gripper_piper 下，匹配控制器的期望格式
  if (gripper_cmd.mode == je_software::msg::EndEffectorCommand::MODE_TORQUE)
  {
    robot["gripper_piper"]["command"] = gripper_cmd.command;
    robot["gripper_piper"]["torque"] = gripper_cmd.torque;
  }
  else
  {
    // 其他模式: 放在 EndEffector 对象下
    nlohmann::json ee;
    ee["mode"] = gripper_cmd.mode;
    if (gripper_cmd.mode == je_software::msg::EndEffectorCommand::MODE_POSITION)
    {
      ee["position"] = gripper_cmd.position;
    }
    else if (gripper_cmd.mode == je_software::msg::EndEffectorCommand::MODE_PRESET)
    {
      ee["preset"] = gripper_cmd.preset;
    }
    robot["EndEffector"] = ee;
  }
}

bool JeZmqSystemHardware::parse_joint_state_from_json(
  const nlohmann::json & state_json,
  std::vector<double> & positions,
  std::vector<double> & velocities,
  std::vector<double> & efforts) const
{
  if (!state_json.contains("Robot0") || !state_json["Robot0"].is_object())
  {
    return false;
  }

  const auto & robot = state_json["Robot0"];
  
  // Try both "joint" (lowercase, new format from je_robot_node) and "Joint" (uppercase, legacy)
  std::vector<double> joints;
  if (robot.contains("joint"))
  {
    joints = robot["joint"].get<std::vector<double>>();
  }
  else if (robot.contains("Joint"))
  {
    joints = robot["Joint"].get<std::vector<double>>();
  }
  else
  {
    return false;
  }
  
  if (joints.size() < positions.size())
  {
    return false;
  }

  for (size_t i = 0; i < positions.size(); ++i)
  {
    positions[i] = joints[i];
  }

  if (robot.contains("JointVelocity"))
  {
    const auto vels = robot["JointVelocity"].get<std::vector<double>>();
    for (size_t i = 0; i < velocities.size() && i < vels.size(); ++i)
    {
      velocities[i] = vels[i];
    }
  }

  if (robot.contains("JointSensorTorque"))
  {
    const auto torques = robot["JointSensorTorque"].get<std::vector<double>>();
    for (size_t i = 0; i < efforts.size() && i < torques.size(); ++i)
    {
      efforts[i] = torques[i];
    }
  }

  return true;
}

void JeZmqSystemHardware::robot_switch(bool value)
{
  if (!publisher_)
  {
    return;
  }

  json cmd_switch;
  cmd_switch["Switch"] = value;
  const std::string payload = "Switch " + cmd_switch.dump();
  publisher_->send(zmq::buffer(payload), zmq::send_flags::none);
}

}  // namespace je_software

PLUGINLIB_EXPORT_CLASS(je_software::JeZmqSystemHardware, hardware_interface::SystemInterface)
