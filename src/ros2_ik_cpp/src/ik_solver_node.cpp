#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>

#include "ros2_ik_cpp/ik_solver.hpp"
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>

#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <mutex>
#include <algorithm>
#include <rcutils/logging.h>

using namespace std::chrono_literals;
using ros2_ik_cpp::IkSolver;
using Params = ros2_ik_cpp::IkSolver::Params;
using Eigen::VectorXd;
using pinocchio::Model;
using pinocchio::Data;
using pinocchio::SE3;

class IkSolverNode : public rclcpp::Node {
public:
  IkSolverNode(): Node("ik_solver_node") {
    // declare the full set of planning parameters (mirror ik_node defaults)
    int planning_dof = this->declare_parameter<int>("planning.dof", 7);
    urdf_path_ = this->declare_parameter<std::string>("planning.urdf_path", "");
    std::string tip_link = this->declare_parameter<std::string>("planning.tip_link", "Link7");
    double gripper_offset_z = this->declare_parameter<double>("planning.gripper_offset_z", 0.35);

    std::string input_type = this->declare_parameter<std::string>("planning.input_type", "xyz");
    std::string ik_solver_type = this->declare_parameter<std::string>("planning.ik_solver_type", "official_3d");

    int max_iters = this->declare_parameter<int>("planning.ik_max_iterations", 200);
    double eps = this->declare_parameter<double>("planning.ik_epsilon", 1e-4);
    double ik_epsilon_relaxed_3d = this->declare_parameter<double>("planning.ik_epsilon_relaxed_3d", 0.005);
    double ik_epsilon_relaxed_6d = this->declare_parameter<double>("planning.ik_epsilon_relaxed_6d", 0.01);
    double ik_damping_3d = this->declare_parameter<double>("planning.ik_damping_3d", 1e-12);
    double ik_damping_6d = this->declare_parameter<double>("planning.ik_damping_6d", 1e-6);
    double ik_step_size = this->declare_parameter<double>("planning.ik_step_size", 0.1);

    bool use_svd_damped = this->declare_parameter<bool>("planning.use_svd_damped", true);
    double ik_svd_damping = this->declare_parameter<double>("planning.ik_svd_damping", 1e-6);
    double max_delta = this->declare_parameter<double>("planning.max_delta", 0.03);
    double ik_svd_damping_min = this->declare_parameter<double>("planning.ik_svd_damping_min", 1e-12);
    double ik_svd_damping_max = this->declare_parameter<double>("planning.ik_svd_damping_max", 1e6);
    double ik_svd_damping_reduce_factor = this->declare_parameter<double>("planning.ik_svd_damping_reduce_factor", 0.1);
    double ik_svd_damping_increase_factor = this->declare_parameter<double>("planning.ik_svd_damping_increase_factor", 10.0);
    double ik_svd_trunc_tol = this->declare_parameter<double>("planning.ik_svd_trunc_tol", 1e-6);
    double ik_svd_min_rel_reduction = this->declare_parameter<double>("planning.ik_svd_min_rel_reduction", 1e-8);

    int numeric_fallback_after_rejects = this->declare_parameter<int>("planning.numeric_fallback_after_rejects", 3);
    int numeric_fallback_duration = this->declare_parameter<int>("planning.numeric_fallback_duration", 10);

    double joint4_penalty_threshold = this->declare_parameter<double>("planning.joint4_penalty_threshold", 0.05);
    double nullspace_penalty_scale = this->declare_parameter<double>("planning.nullspace_penalty_scale", 1e-4);

    std::vector<double> joint_limits_min = this->declare_parameter<std::vector<double>>("planning.joint_limits_min", std::vector<double>());
    std::vector<double> joint_limits_max = this->declare_parameter<std::vector<double>>("planning.joint_limits_max", std::vector<double>());

    double pos_weight = this->declare_parameter<double>("planning.pos_weight", 1.0);
    double ang_weight = this->declare_parameter<double>("planning.ang_weight", 1.0);

    // ik_delta scales (declare here so they can be set via params file)
    ik_delta_linear_scale_ = this->declare_parameter<double>("planning.ik_delta_linear_scale", 0.01);
    ik_delta_angular_scale_ = this->declare_parameter<double>("planning.ik_delta_angular_scale", 0.02);

    bool use_numeric_jacobian = this->declare_parameter<bool>("planning.use_numeric_jacobian", false);

    // optional logging parameters
    bool log_to_file = this->declare_parameter<bool>("planning.log_to_file", false);
    std::string log_file = this->declare_parameter<std::string>("planning.log_file", std::string());
    std::string nullspace_log_file = this->declare_parameter<std::string>("planning.nullspace_log_file", std::string());
    std::string log_level = this->declare_parameter<std::string>("planning.log_level", "info");
    // apply node-level log level
    int sev = RCUTILS_LOG_SEVERITY_INFO; std::string ll = log_level; std::transform(ll.begin(), ll.end(), ll.begin(), ::tolower);
    if (ll=="debug") sev = RCUTILS_LOG_SEVERITY_DEBUG; else if (ll=="warn"||ll=="warning") sev=RCUTILS_LOG_SEVERITY_WARN; else if (ll=="error") sev=RCUTILS_LOG_SEVERITY_ERROR; else if (ll=="fatal") sev=RCUTILS_LOG_SEVERITY_FATAL;
    rcutils_logging_set_logger_level(this->get_logger().get_name(), sev);

    // populate solver Params from declared parameters
    Params p;
    p.max_iters = max_iters;
    p.eps = eps;
    p.eps_relaxed_3d = ik_epsilon_relaxed_3d;
    p.eps_relaxed_6d = ik_epsilon_relaxed_6d;
    p.dt = ik_step_size;
    p.use_svd_damped = use_svd_damped;
    p.ik_svd_damping = ik_svd_damping;
    p.ik_svd_damping_min = ik_svd_damping_min;
    p.ik_svd_damping_max = ik_svd_damping_max;
    p.ik_svd_damping_reduce_factor = ik_svd_damping_reduce_factor;
    p.ik_svd_damping_increase_factor = ik_svd_damping_increase_factor;
    p.ik_svd_trunc_tol = ik_svd_trunc_tol;
    p.ik_svd_min_rel_reduction = ik_svd_min_rel_reduction;
    p.nullspace_penalty_scale = nullspace_penalty_scale;
    p.joint4_penalty_threshold = joint4_penalty_threshold;
    p.pos_weight = pos_weight;
    p.ang_weight = ang_weight;
    p.max_delta = max_delta;
    p.use_numeric_jacobian = use_numeric_jacobian;
    p.numeric_fallback_after_rejects = numeric_fallback_after_rejects;
    p.numeric_fallback_duration = numeric_fallback_duration;
    p.log_to_file = log_to_file;
    p.log_file = log_file;
    p.nullspace_log_file = nullspace_log_file;
    if (!joint_limits_min.empty() && joint_limits_min.size()>0) p.joint_limits_min = joint_limits_min;
    if (!joint_limits_max.empty() && joint_limits_max.size()>0) p.joint_limits_max = joint_limits_max;

    // Load URDF and build Pinocchio model before creating the solver
    if (urdf_path_.empty()) {
      RCLCPP_ERROR(this->get_logger(), "planning.urdf_path param required");
      throw std::runtime_error("urdf_path missing");
    }
    std::string urdf = readFileToString(urdf_path_);
    if (urdf.empty()) {
      RCLCPP_ERROR(this->get_logger(), "Failed to read URDF: %s", urdf_path_.c_str());
      throw std::runtime_error("URDF read failed");
    }
    try {
      pinocchio::urdf::buildModelFromXML(urdf, model_);
    } catch (const std::exception &e) {
      RCLCPP_ERROR(this->get_logger(), "Pinocchio URDF parse failed: %s", e.what());
      throw;
    }
    data_ = Data(model_);
    tip_frame_id_ = model_.getFrameId(tip_link);

    // cache q indices for joints we may want to target for null-space penalty
    j4_q_index_ = j5_q_index_ = j7_q_index_ = j3_q_index_ = -1;
    try { int j4_id = model_.getJointId("joint4"); if (j4_id>=0) j4_q_index_ = model_.joints[j4_id].idx_q(); } catch(...) {}
    try { int j3_id = model_.getJointId("joint3"); if (j3_id>=0) j3_q_index_ = model_.joints[j3_id].idx_q(); } catch(...) {}
    try { int j5_id = model_.getJointId("joint5"); if (j5_id>=0) j5_q_index_ = model_.joints[j5_id].idx_q(); } catch(...) {}
    try { int j7_id = model_.getJointId("joint7"); if (j7_id>=0) j7_q_index_ = model_.joints[j7_id].idx_q(); } catch(...) {}

    RCLCPP_DEBUG(this->get_logger(), "Pinocchio model loaded: nq=%d nv=%d tip_frame='%s' id=%u",
                 static_cast<int>(model_.nq), static_cast<int>(model_.nv), tip_link.c_str(), tip_frame_id_);

    // create solver now that model_ is initialized
    solver_ = std::make_shared<IkSolver>(model_);
    solver_->setParams(p);

    if (!p.joint_limits_min.empty() && p.joint_limits_min.size() == model_.nq && !p.joint_limits_max.empty() && p.joint_limits_max.size() == model_.nq) {
      VectorXd lo(model_.nq), hi(model_.nq);
      for (int i=0;i<model_.nq;++i) { lo[i]=p.joint_limits_min[i]; hi[i]=p.joint_limits_max[i]; }
      solver_->setJointLimits(lo, hi);
    }

    // create subs/pubs/timer as before, plus additional subscriptions to match ik_node
    sub_js_ = this->create_subscription<sensor_msgs::msg::JointState>("joint_states", 10, std::bind(&IkSolverNode::onJointState, this, std::placeholders::_1));
    sub_target_ = this->create_subscription<geometry_msgs::msg::PoseStamped>("target_pose", 10, std::bind(&IkSolverNode::onTargetPose, this, std::placeholders::_1));
    sub_target_end_ = this->create_subscription<geometry_msgs::msg::PoseStamped>("target_end_pose", 10, std::bind(&IkSolverNode::onTargetEndPose, this, std::placeholders::_1));
    sub_delta_ = this->create_subscription<geometry_msgs::msg::Twist>("ik_delta", 10, std::bind(&IkSolverNode::onIkDelta, this, std::placeholders::_1));
    pub_cmd_ = this->create_publisher<sensor_msgs::msg::JointState>("joint_command", 10);

    timer_ = this->create_wall_timer(50ms, std::bind(&IkSolverNode::spinOnce, this));
  }

private:
  static std::string readFileToString(const std::string &path) {
    std::ifstream ifs(path);
    if (!ifs) return std::string();
    std::ostringstream ss; ss << ifs.rdbuf();
    return ss.str();
  }

  void onJointState(const sensor_msgs::msg::JointState::SharedPtr msg) {
    std::lock_guard<std::mutex> lk(mutex_);
    last_js_ = *msg;

    // If we don't yet have a target pose, initialize it from current FK computed from incoming joint_states
    if (!target_received_ && !last_js_.position.empty()) {
      VectorXd q = VectorXd::Zero(model_.nq);
      for (size_t i = 0; i < last_js_.position.size() && i < (size_t)q.size(); ++i) q[i] = last_js_.position[i];
      Data data_fk(model_);
      try {
        pinocchio::forwardKinematics(model_, data_fk, q);
        pinocchio::updateFramePlacements(model_, data_fk);
        const SE3 &current_pose = data_fk.oMf[tip_frame_id_];
        Eigen::Quaterniond eq(current_pose.rotation());
        last_target_.header.stamp = this->get_clock()->now();
        last_target_.header.frame_id = "";
        last_target_.pose.position.x = current_pose.translation()[0];
        last_target_.pose.position.y = current_pose.translation()[1];
        last_target_.pose.position.z = current_pose.translation()[2];
        last_target_.pose.orientation.x = eq.x();
        last_target_.pose.orientation.y = eq.y();
        last_target_.pose.orientation.z = eq.z();
        last_target_.pose.orientation.w = eq.w();
        target_received_ = true;
      } catch (const std::exception &e) {
        RCLCPP_WARN(this->get_logger(), "Failed to compute FK to initialize target pose: %s", e.what());
      }
    }
  }

  void onTargetPose(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    std::lock_guard<std::mutex> lk(mutex_);
    last_target_ = *msg;
    target_received_ = true;
    prefer_end_pose_target_ = false;
  }

  void onTargetEndPose(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    std::lock_guard<std::mutex> lk(mutex_);
    last_target_ = *msg;
    target_received_ = true;
    prefer_end_pose_target_ = true;
    RCLCPP_DEBUG(this->get_logger(), "Received target_end_pose; will ignore ik_delta until overridden");
  }

  void onIkDelta(const geometry_msgs::msg::Twist::SharedPtr msg) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (!target_received_) return;
    if (prefer_end_pose_target_) {
      RCLCPP_DEBUG(this->get_logger(), "Ignoring ik_delta because target_end_pose is active");
      return;
    }
    // linear deltas (assumed in world frame) scaled by parameters
    last_target_.pose.position.x += msg->linear.x * ik_delta_linear_scale_;
    last_target_.pose.position.y += msg->linear.y * ik_delta_linear_scale_;
    last_target_.pose.position.z += msg->linear.z * ik_delta_linear_scale_;

    Eigen::Vector3d ang;
    ang.x() = msg->angular.x * ik_delta_angular_scale_;
    ang.y() = msg->angular.y * ik_delta_angular_scale_;
    ang.z() = msg->angular.z * ik_delta_angular_scale_;
    double angle = ang.norm();
    if (angle > 1e-12) {
      Eigen::Vector3d axis = ang / angle;
      Eigen::AngleAxisd aa(angle, axis);
      Eigen::Quaterniond q_delta(aa);
      Eigen::Quaterniond q_old(last_target_.pose.orientation.w,
                               last_target_.pose.orientation.x,
                               last_target_.pose.orientation.y,
                               last_target_.pose.orientation.z);
      Eigen::Quaterniond q_new = q_delta * q_old;
      q_new.normalize();
      last_target_.pose.orientation.x = q_new.x();
      last_target_.pose.orientation.y = q_new.y();
      last_target_.pose.orientation.z = q_new.z();
      last_target_.pose.orientation.w = q_new.w();
    }
    RCLCPP_DEBUG(this->get_logger(), "Applied ik_delta: linear=(%.4f,%.4f,%.4f) angular=(%.4f,%.4f,%.4f)",
                 msg->linear.x, msg->linear.y, msg->linear.z, msg->angular.x, msg->angular.y, msg->angular.z);
  }

  void spinOnce() {
    sensor_msgs::msg::JointState js;
    geometry_msgs::msg::PoseStamped target;
    {
      std::lock_guard<std::mutex> lk(mutex_);
      js = last_js_;
      target = last_target_;
      // target_received_ may be false; we still want to solve to hold current pose
    }
    if (js.position.empty()) return;

    VectorXd q_init = VectorXd::Zero(model_.nq);
    for (size_t i=0;i<js.position.size() && i<q_init.size(); ++i) q_init[i] = js.position[i];

    if (!target_received_) {
      // No external target: use current end-effector pose (from joints) as the target
      Eigen::VectorXd curv = solver_->forwardKinematics(q_init);
      geometry_msgs::msg::PoseStamped cur_msg;
      cur_msg.header.stamp = this->now();
      // curv: [tx,ty,tz, qx,qy,qz] (quat w omitted), reconstruct w
      double qx = curv.size() > 3 ? curv(3) : 0.0;
      double qy = curv.size() > 4 ? curv(4) : 0.0;
      double qz = curv.size() > 5 ? curv(5) : 0.0;
      double qw2 = 1.0 - (qx*qx + qy*qy + qz*qz);
      double qw = (qw2 > 0.0) ? std::sqrt(qw2) : 0.0;
      Eigen::Quaterniond qrot(qw, qx, qy, qz);
      qrot.normalize();
      cur_msg.pose.orientation.w = qrot.w();
      cur_msg.pose.orientation.x = qrot.x();
      cur_msg.pose.orientation.y = qrot.y();
      cur_msg.pose.orientation.z = qrot.z();
      cur_msg.pose.position.x = curv.size() > 0 ? curv(0) : 0.0;
      cur_msg.pose.position.y = curv.size() > 1 ? curv(1) : 0.0;
      cur_msg.pose.position.z = curv.size() > 2 ? curv(2) : 0.0;
      target = cur_msg;
      // update last_target_ so subsequent loops keep using this pose until a new one arrives
      {
        std::lock_guard<std::mutex> lk(mutex_);
        last_target_ = target;
      }
    }

    SE3 target_se3(Eigen::Quaterniond(target.pose.orientation.w,
                                      target.pose.orientation.x,
                                      target.pose.orientation.y,
                                      target.pose.orientation.z),
                   Eigen::Vector3d(target.pose.position.x,
                                   target.pose.position.y,
                                   target.pose.position.z));

    auto t_spin_start = std::chrono::steady_clock::now();
    auto res = solver_->solve(target_se3, q_init, 100);
    auto t_spin_end = std::chrono::steady_clock::now();
    double spin_elapsed_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_spin_end - t_spin_start).count();

    const char *status_str = "失败";
    if (res.status == 1) status_str = "精确成功";
    else if (res.status == 2) status_str = "放宽成功";

    RCLCPP_INFO(this->get_logger(), "ik_solver_node: solver elapsed: %.3f ms, final_err: %.6g, iterations: %d, status: %s",
                res.elapsed_ms, res.final_error, res.iterations, status_str);
    RCLCPP_INFO(this->get_logger(), "spinOnce elapsed: %.3f ms [结果: %s]", spin_elapsed_ms, status_str);

    if (res.success) {
      sensor_msgs::msg::JointState out;
      out.header.stamp = this->now();
      out.name = js.name;
      out.position.resize(res.q.size());
      for (int i=0;i<res.q.size() && i<(int)out.position.size(); ++i) out.position[i] = res.q[i];
      pub_cmd_->publish(out);
    } else {
      RCLCPP_WARN(this->get_logger(), "IK solver failed: %s", res.diagnostic.c_str());
    }
  }

  std::mutex mutex_;
  sensor_msgs::msg::JointState last_js_;
  geometry_msgs::msg::PoseStamped last_target_;
  bool target_received_{false};
  bool prefer_end_pose_target_{false};
  double ik_delta_linear_scale_{0.01};
  double ik_delta_angular_scale_{0.02};

  Model model_;
  Data data_;
  std::shared_ptr<IkSolver> solver_;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_js_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_target_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_target_end_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr sub_delta_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_cmd_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::string urdf_path_;
  unsigned int tip_frame_id_{0};
  int j4_q_index_{-1};
  int j5_q_index_{-1};
  int j7_q_index_{-1};
  int j3_q_index_{-1};
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<IkSolverNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
