#pragma once

#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "rclcpp/macros.hpp"
#include "rclcpp_lifecycle/state.hpp"
#include <zmq.hpp>
#include <nlohmann/json.hpp>

namespace je_software
{

class JeZmqSystemHardware : public hardware_interface::SystemInterface
{
public:
  RCLCPP_SHARED_PTR_DEFINITIONS(JeZmqSystemHardware)

  hardware_interface::CallbackReturn on_init(const hardware_interface::HardwareInfo & info) override;

  hardware_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;

  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  hardware_interface::return_type read(
    const rclcpp::Time & time,
    const rclcpp::Duration & period) override;

  hardware_interface::return_type write(
    const rclcpp::Time & time,
    const rclcpp::Duration & period) override;

private:
  void state_receive_loop();
  bool parse_joint_state_from_json(const nlohmann::json & state_json, std::vector<double> & positions,
                                   std::vector<double> & velocities, std::vector<double> & efforts) const;
  void robot_switch(bool value);

  std::vector<std::string> joint_names_;
  std::vector<double> hw_positions_;
  std::vector<double> hw_velocities_;
  std::vector<double> hw_efforts_;
  std::vector<double> hw_commands_;

  std::unordered_map<std::string, std::size_t> joint_name_to_index_;

  std::mutex state_mutex_;
  std::vector<double> latest_positions_;
  std::vector<double> latest_velocities_;
  std::vector<double> latest_efforts_;
  bool has_valid_state_{false};

  std::string robot_ip_{"192.168.0.99"};
  int pub_port_{8001};
  int sub_port_{8000};
  double command_dt_{0.02};
  double global_time_{0.0};

  std::atomic_bool state_thread_running_{false};
  std::thread state_thread_;

  std::unique_ptr<zmq::context_t> context_;
  std::unique_ptr<zmq::socket_t> publisher_;
  std::unique_ptr<zmq::socket_t> subscriber_;
};

}  // namespace je_software
