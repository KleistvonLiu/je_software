#include "je_zmq_system_hardware.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <string>
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

  RCLCPP_INFO(
    rclcpp::get_logger("JeZmqSystemHardware"),
    "Configured JE hardware transport: robot_ip=%s, pub_port=%d, sub_port=%d, command_dt=%.4f",
    robot_ip_.c_str(), pub_port_, sub_port_, command_dt_);

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
  hw_commands_ = hw_positions_;
  global_time_ = 0.0;

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

  state_thread_running_.store(true);
  state_thread_ = std::thread(&JeZmqSystemHardware::state_receive_loop, this);

  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn JeZmqSystemHardware::on_deactivate(
  const rclcpp_lifecycle::State &)
{
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
  if (has_valid_state_)
  {
    hw_positions_ = latest_positions_;
    hw_velocities_ = latest_velocities_;
    hw_efforts_ = latest_efforts_;
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
  payload["Robot0"]["joint"] = hw_commands_;

  try
  {
    const std::string message = "Joint " + payload.dump();
    publisher_->send(zmq::buffer(message), zmq::send_flags::none);
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

      std::vector<double> positions(joint_names_.size(), 0.0);
      std::vector<double> velocities(joint_names_.size(), 0.0);
      std::vector<double> efforts(joint_names_.size(), 0.0);

      if (!parse_joint_state_from_json(state_json, positions, velocities, efforts))
      {
        continue;
      }

      {
        std::lock_guard<std::mutex> lock(state_mutex_);
        latest_positions_ = std::move(positions);
        latest_velocities_ = std::move(velocities);
        latest_efforts_ = std::move(efforts);
        has_valid_state_ = true;
      }
    }
    catch (const std::exception & ex)
    {
      RCLCPP_ERROR(
        rclcpp::get_logger("JeZmqSystemHardware"),
        "Failed parsing state JSON: %s", ex.what());
    }
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
  if (!robot.contains("Joint"))
  {
    return false;
  }

  const auto joints = robot["Joint"].get<std::vector<double>>();
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
