#pragma once

#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <memory>

#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "rclcpp/executors/single_threaded_executor.hpp"
#include "rclcpp/macros.hpp"
#include "rclcpp/node.hpp"
#include "rclcpp_lifecycle/state.hpp"
#include "je_software/msg/end_effector_command.hpp"
#include "je_software/msg/end_effector_command_lr.hpp"
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
  struct GripperCommand
  {
    int mode{je_software::msg::EndEffectorCommand::MODE_POSITION};
    double position{0.0};
    int preset{0};
    std::string command;
    double torque{0.0};
    bool received{false};
  };

  void state_receive_loop();
  bool parse_joint_state_from_json(const nlohmann::json & state_json, std::vector<double> & positions,
                                   std::vector<double> & velocities, std::vector<double> & efforts) const;
  void robot_switch(bool value);
  void gripper_spin_loop();
  void gripper_cmd_callback(const je_software::msg::EndEffectorCommandLR::SharedPtr msg);
  void handle_gripper_cmd(const je_software::msg::EndEffectorCommand & msg, int robot_index);
  bool map_joint_to_robot_slot(
    const std::string & joint_name,
    int & robot_index,
    std::size_t & slot_index) const;
  const GripperCommand & get_gripper_command(int robot_index) const;
  void append_gripper_from_command(nlohmann::json & payload, int robot_index) const;

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

  mutable std::mutex gripper_mutex_;
  std::string gripper_sub_topic_{"/end_effector_cmd_lr"};
  GripperCommand gripper_cmd_left_{};
  GripperCommand gripper_cmd_right_{};
  std::shared_ptr<rclcpp::Node> gripper_node_;
  rclcpp::Subscription<je_software::msg::EndEffectorCommandLR>::SharedPtr sub_gripper_cmd_;
  std::shared_ptr<rclcpp::executors::SingleThreadedExecutor> gripper_executor_;
  std::atomic_bool gripper_thread_running_{false};
  std::thread gripper_thread_;

  std::unique_ptr<zmq::context_t> context_;
  std::unique_ptr<zmq::socket_t> publisher_;
  std::unique_ptr<zmq::socket_t> subscriber_;
};

}  // namespace je_software
