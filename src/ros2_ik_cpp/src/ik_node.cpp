#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <mutex>
#include <string>
#include <limits>
#include <algorithm>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using pinocchio::Model;
using pinocchio::Data;
using pinocchio::SE3;

class IkNode : public rclcpp::Node {
public:
  IkNode() : Node("ik_node") {
    // Read ROS2 parameters under the namespace 'planning' (use --params-file to load)
    planning_dof_ = this->declare_parameter<int>("planning.dof", 5);
    urdf_path_ = this->declare_parameter<std::string>("planning.urdf_path", "");
    tip_frame_name_ = this->declare_parameter<std::string>("planning.tip_link", "Link17");
    gripper_offset_z_ = this->declare_parameter<double>("planning.gripper_offset_z", 0.35);

    input_type_ = this->declare_parameter<std::string>("planning.input_type", "xyz");
    ik_solver_type_ = this->declare_parameter<std::string>("planning.ik_solver_type", "official_3d");

    max_iters_ = this->declare_parameter<int>("planning.ik_max_iterations", 1000);
    eps_ = this->declare_parameter<double>("planning.ik_epsilon", 1e-4);
    ik_epsilon_relaxed_3d_ = this->declare_parameter<double>("planning.ik_epsilon_relaxed_3d", 0.005);
    ik_epsilon_relaxed_6d_ = this->declare_parameter<double>("planning.ik_epsilon_relaxed_6d", 0.01);
    ik_damping_3d_ = this->declare_parameter<double>("planning.ik_damping_3d", 1e-12);
    damp_ = this->declare_parameter<double>("planning.ik_damping_6d", 1e-6);
    dt_ = this->declare_parameter<double>("planning.ik_step_size", 0.1);

    // Enable SVD-damped pseudo-inverse and per-step clamping
    use_svd_damped_ = this->declare_parameter<bool>("planning.use_svd_damped", true);
    ik_svd_damping_ = this->declare_parameter<double>("planning.ik_svd_damping", 1e-6);
    max_delta_ = this->declare_parameter<double>("planning.max_delta", 0.03); // rad, per-iteration max joint change

    max_velocity_ = this->declare_parameter<double>("planning.max_velocity", 1.0);
    max_acceleration_ = this->declare_parameter<double>("planning.max_acceleration", 2.0);
    max_jerk_ = this->declare_parameter<double>("planning.max_jerk", 5.0);
    control_frequency_ = this->declare_parameter<double>("planning.control_frequency", 50.0);

    joint_limits_min_ = this->declare_parameter<std::vector<double>>("planning.joint_limits_min", std::vector<double>{-3.14, -2.0, -2.5, -3.14, -3.14});
    joint_limits_max_ = this->declare_parameter<std::vector<double>>("planning.joint_limits_max", std::vector<double>{3.14, 2.0, 2.5, 3.14, 3.14});

    // Optional: file to write IK diagnostics (set planning.log_file to enable)
    std::string log_file = this->declare_parameter<std::string>("planning.log_file", std::string());
    if (!log_file.empty()) {
      log_ofs_.open(log_file, std::ios::out | std::ios::app);
      if (!log_ofs_) {
        RCLCPP_WARN(this->get_logger(), "Failed to open IK diagnostics log file: %s", log_file.c_str());
      } else {
        log_to_file_ = true;
        RCLCPP_INFO(this->get_logger(), "IK diagnostics will be written to: %s", log_file.c_str());
      }
    }

    if (urdf_path_.empty()) {
      RCLCPP_ERROR(this->get_logger(), "urdf_path param required");
      throw std::runtime_error("urdf_path missing");
    }

    std::string urdf = readFileToString(urdf_path_);
    if (urdf.empty()) {
      RCLCPP_ERROR(this->get_logger(), "Failed to read URDF: %s", urdf_path_.c_str());
      throw std::runtime_error("URDF read failed");
    }

    // build model from URDF string
    try {
      pinocchio::urdf::buildModelFromXML(urdf, model_);
    } catch (const std::exception &e) {
      RCLCPP_ERROR(this->get_logger(), "Pinocchio URDF parse failed: %s", e.what());
      throw;
    }
    data_ = Data(model_);
    tip_frame_id_ = model_.getFrameId(tip_frame_name_);

    // Diagnostic: print model sizes and tip frame info
    RCLCPP_INFO(this->get_logger(), "Pinocchio model loaded: nq=%d nv=%d tip_frame='%s' id=%u",
                static_cast<int>(model_.nq), static_cast<int>(model_.nv), tip_frame_name_.c_str(), tip_frame_id_);

    sub_js_ = this->create_subscription<sensor_msgs::msg::JointState>(
        "joint_states", 10, std::bind(&IkNode::onJointState, this, std::placeholders::_1));
    sub_target_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "target_pose", 10, std::bind(&IkNode::onTargetPose, this, std::placeholders::_1));
    // new: subscription to receive small pose deltas (e.g. from a teleop key node)
    ik_delta_scale_linear_ = this->declare_parameter<double>("planning.ik_delta_linear_scale", 0.01);
    ik_delta_scale_angular_ = this->declare_parameter<double>("planning.ik_delta_angular_scale", 0.02); // radians per unit
    sub_delta_ = this->create_subscription<geometry_msgs::msg::Twist>(
        "ik_delta", 10, std::bind(&IkNode::onIkDelta, this, std::placeholders::_1));
    pub_cmd_ = this->create_publisher<sensor_msgs::msg::JointState>("joint_command", 10);

    timer_ = this->create_wall_timer(std::chrono::milliseconds(50), std::bind(&IkNode::spinOnce, this));

    RCLCPP_INFO(this->get_logger(), "ik_node initialized, tip_frame=%s", tip_frame_name_.c_str());
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
      Data data(model_);
      try {
        pinocchio::forwardKinematics(model_, data, q);
        pinocchio::updateFramePlacements(model_, data);
        const SE3 &current_pose = data.oMf[tip_frame_id_];
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

        // Print FK result (position, quaternion and RPY)
        Eigen::Vector3d rpy = current_pose.rotation().eulerAngles(0, 1, 2);
        RCLCPP_INFO(this->get_logger(), "FK init pose: pos=(%.4f, %.4f, %.4f) quat=(%.4f, %.4f, %.4f, %.4f) rpy=(%.4f, %.4f, %.4f)",
                    current_pose.translation()[0], current_pose.translation()[1], current_pose.translation()[2],
                    eq.x(), eq.y(), eq.z(), eq.w(), rpy[0], rpy[1], rpy[2]);

      } catch (const std::exception &e) {
        RCLCPP_WARN(this->get_logger(), "Failed to compute FK to initialize target pose: %s", e.what());
      }
    }
  }

  void onTargetPose(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    std::lock_guard<std::mutex> lk(mutex_);
    last_target_ = *msg;
    target_received_ = true;
  }

  // Apply incremental pose delta messages to the internally tracked target pose
  void onIkDelta(const geometry_msgs::msg::Twist::SharedPtr msg) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (!target_received_) return;

    // linear deltas (assumed in world frame) scaled by parameter
    last_target_.pose.position.x += msg->linear.x * ik_delta_scale_linear_;
    last_target_.pose.position.y += msg->linear.y * ik_delta_scale_linear_;
    last_target_.pose.position.z += msg->linear.z * ik_delta_scale_linear_;

    // angular deltas (small rotation vector) scaled and applied to orientation
    Eigen::Vector3d ang;
    ang.x() = msg->angular.x * ik_delta_scale_angular_;
    ang.y() = msg->angular.y * ik_delta_scale_angular_;
    ang.z() = msg->angular.z * ik_delta_scale_angular_;
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
  }

  void spinOnce() {
    sensor_msgs::msg::JointState js;
    geometry_msgs::msg::PoseStamped target;
    {
      std::lock_guard<std::mutex> lk(mutex_);
      if (!target_received_ || last_js_.name.empty()) return;
      js = last_js_;
      target = last_target_;
    }

    VectorXd q_init = VectorXd::Zero(model_.nq);
    for (size_t i = 0; i < js.position.size() && i < (size_t)q_init.size(); ++i) {
      q_init[i] = js.position[i];
    }

    // Print current FK (debug)
    try {
      Data data_fk(model_);
      pinocchio::forwardKinematics(model_, data_fk, q_init);
      pinocchio::updateFramePlacements(model_, data_fk);
      const SE3 &cur_fk = data_fk.oMf[tip_frame_id_];
      Eigen::Quaterniond qcur(cur_fk.rotation());
      Eigen::Vector3d rpy_cur = cur_fk.rotation().eulerAngles(0,1,2);
      // Reduced verbosity: use DEBUG so normal runs are not spammed
      RCLCPP_DEBUG(this->get_logger(), "FK current pose: pos=(%.4f, %.4f, %.4f) quat=(%.4f, %.4f, %.4f, %.4f) rpy=(%.4f, %.4f, %.4f)",
                   cur_fk.translation()[0], cur_fk.translation()[1], cur_fk.translation()[2],
                   qcur.x(), qcur.y(), qcur.z(), qcur.w(), rpy_cur[0], rpy_cur[1], rpy_cur[2]);
    } catch (const std::exception &e) {
      RCLCPP_WARN(this->get_logger(), "FK debug print failed: %s", e.what());
    }

    SE3 target_se3(Eigen::Quaterniond(target.pose.orientation.w,
                                      target.pose.orientation.x,
                                      target.pose.orientation.y,
                                      target.pose.orientation.z),
                   Eigen::Vector3d(target.pose.position.x,
                                   target.pose.position.y,
                                   target.pose.position.z));

    VectorXd q_sol;
    bool ok = solveIK6D(target_se3, q_init, q_sol);
    if (ok) {
      sensor_msgs::msg::JointState out;
      out.header.stamp = this->get_clock()->now();
      out.name = js.name;
      out.position.resize(js.name.size());
      for (size_t i = 0; i < js.name.size() && i < (size_t)q_sol.size(); ++i) out.position[i] = q_sol[i];
      pub_cmd_->publish(out);
      // Reduced verbosity: log at DEBUG to avoid console spam during continuous solves
      RCLCPP_DEBUG(this->get_logger(), "IK succeeded, published joint_command");
    } else {
      RCLCPP_WARN(this->get_logger(), "IK failed or did not converge");
    }
  }

  bool solveIK6D(const SE3 &target_pose, const VectorXd &q_init, VectorXd &q_solution) {
    q_solution = q_init;
    const double DT = dt_;
    const double damp = damp_;

    Data data(model_);
    MatrixXd J = MatrixXd::Zero(6, model_.nv);
    Eigen::Matrix<double,6,1> err6;
    VectorXd v = VectorXd::Zero(model_.nv);

    double best_error = 1e300;
    VectorXd best_q = q_init;

    bool printed_initial = false;

    for (int it = 0; it < max_iters_; ++it) {
      pinocchio::forwardKinematics(model_, data, q_solution);
      pinocchio::updateFramePlacements(model_, data);
      const SE3 &current_pose = data.oMf[tip_frame_id_];
      // compute 6D error: linear (target - current) and angular (axis-angle from current to target)
      Eigen::Vector3d pos_cur = current_pose.translation();
      Eigen::Vector3d pos_tgt = target_pose.translation();
      Eigen::Vector3d pos_err = pos_tgt - pos_cur;
      Eigen::Quaterniond qcur(current_pose.rotation());
      Eigen::Quaterniond qtgt(target_pose.rotation());
      Eigen::Quaterniond qerr = qtgt * qcur.conjugate();
      // ensure normalized
      qerr.normalize();
      Eigen::AngleAxisd aa(qerr);
      Eigen::Vector3d ang_err = Eigen::Vector3d::Zero();
      double angle = aa.angle();
      if (std::isfinite(angle) && std::abs(angle) > 1e-12) ang_err = aa.axis() * angle;
      err6.template head<3>() = pos_err;
      err6.template tail<3>() = ang_err;
      double cur_err = err6.norm();

      // Print initial err6 once and per-iteration diagnostics at DEBUG level
      if (!printed_initial) {
        RCLCPP_INFO(this->get_logger(), "IK initial err6 = [%.6f, %.6f, %.6f, %.6f, %.6f, %.6f] (norm=%.6f)",
                    err6(0), err6(1), err6(2), err6(3), err6(4), err6(5), cur_err);
        if (log_to_file_ && log_ofs_) {
          log_ofs_ << "ITER 0 INIT_ERR6 " << err6.transpose() << " NORM " << cur_err << std::endl;
        }
        printed_initial = true;
      }

      RCLCPP_DEBUG(this->get_logger(), "IK iter=%d cur_err=%.8f err6=[%.6f %.6f %.6f | %.6f %.6f %.6f]", it, cur_err,
                   err6(0), err6(1), err6(2), err6(3), err6(4), err6(5));

      if (cur_err < best_error) { best_error = cur_err; best_q = q_solution; }
      if (cur_err < eps_) { return true; }

      // Ensure Pinocchio has computed joint Jacobians for the current configuration
      pinocchio::computeJointJacobians(model_, data, q_solution);
      // updateFramePlacements already called above; now get the frame Jacobian
      J = pinocchio::getFrameJacobian(model_, data, tip_frame_id_, pinocchio::LOCAL);

      // If Jacobian appears all-zero, warn once — indicates model/joint mismatch or fixed frame
      static bool jacobian_zero_warned = false;
      if (!jacobian_zero_warned && J.norm() == 0.0) {
        RCLCPP_WARN(this->get_logger(), "Jacobian is zero matrix. Check model joint mapping, tip frame, and that computeJointJacobians was called.");
        jacobian_zero_warned = true;
      }

      // compute singular values and condition estimate for diagnosis
      Eigen::JacobiSVD<MatrixXd> svd(J, Eigen::ComputeThinU | Eigen::ComputeThinV);
      Eigen::VectorXd s = svd.singularValues();
      int sn = (int)s.size();
      double cond = std::numeric_limits<double>::infinity();
      if (sn > 0 && s(sn-1) > 0) cond = s(0) / s(sn-1);
      // build a compact string of singular values (up to 6)
      std::ostringstream ss;
      for (int i = 0; i < sn; ++i) {
        if (i) ss << ", ";
        ss << s(i);
      }
      RCLCPP_DEBUG(this->get_logger(), "Jacobian singular values (%d) = %s cond=%.3e", sn, ss.str().c_str(), cond);
      if (log_to_file_ && log_ofs_) {
        log_ofs_ << "ITER " << it << " SVD ";
        for (int ii = 0; ii < sn; ++ii) { if (ii) log_ofs_ << ","; log_ofs_ << s(ii); }
        log_ofs_ << " COND " << cond << " ERR " << cur_err << std::endl;
      }

      // Compute joint-space velocity v using SVD-damped pseudo-inverse if enabled
      if (use_svd_damped_ && sn > 0) {
        // matrixU is 6 x r, matrixV is nv x r, singularValues length r
        Eigen::MatrixXd U = svd.matrixU();
        Eigen::MatrixXd V = svd.matrixV();
        Eigen::VectorXd sval = svd.singularValues();
        int r = (int)sval.size();
        double smax = (r > 0) ? sval(0) : 0.0;
        // Tikhonov-style damping scaled by largest singular value
        double lambda = ik_svd_damping_ * smax * smax;

        // project error into singular-vector basis
        Eigen::VectorXd Ut_err = U.transpose() * err6;
        Eigen::VectorXd scaled = Eigen::VectorXd::Zero(r);
        for (int ii = 0; ii < r; ++ii) {
          double si = sval(ii);
          double coeff = (si * 1.0) / (si*si + lambda);
          scaled(ii) = coeff * Ut_err(ii);
        }
        v = - V * scaled; // nv x 1

        if (log_to_file_ && log_ofs_) {
          log_ofs_ << "ITER " << it << " LAMBDA " << lambda << " MAX_DELTA " << max_delta_ << std::endl;
        }
      } else {
        // fallback to previous JJt-based solve if SVD is disabled or rank==0
        MatrixXd JJt = J * J.transpose();
        JJt.diagonal().array() += damp;
        v = -J.transpose() * JJt.ldlt().solve(err6);
      }

      // Apply per-joint per-iteration clamping on delta = v * DT
      double max_abs_dq = 0.0;
      for (int qi = 0; qi < (int)q_solution.size() && qi < (int)v.size(); ++qi) {
        double dq = v[qi] * DT;
        if (!std::isfinite(dq)) dq = 0.0;
        if (dq > max_delta_) dq = max_delta_;
        else if (dq < -max_delta_) dq = -max_delta_;
        q_solution[qi] += dq;
        max_abs_dq = std::max(max_abs_dq, std::abs(dq));
      }
      RCLCPP_DEBUG(this->get_logger(), "IK iter=%d applied max_abs_dq=%.6f (rad)", it, max_abs_dq);
    }

    if (best_error < eps_ * 100.0) { q_solution = best_q; if (log_to_file_ && log_ofs_) log_ofs_ << "IK_CONVERGED BEST_ERR " << best_error << std::endl; return true; }
    if (log_to_file_ && log_ofs_) log_ofs_ << "IK_FAILED BEST_ERR " << best_error << std::endl;
    return false;
  }

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_js_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_target_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr sub_delta_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_cmd_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::mutex mutex_;
  sensor_msgs::msg::JointState last_js_;
  geometry_msgs::msg::PoseStamped last_target_;
  bool target_received_{false};
  double ik_delta_scale_linear_{0.01};
  double ik_delta_scale_angular_{0.02};

  Model model_;
  Data data_;
  std::string urdf_path_;
  std::string tip_frame_name_;
  unsigned int tip_frame_id_{0};

  // Planning parameters
  int max_iters_;
  double eps_;
  double dt_;
  double damp_;
  // new planning params
  int planning_dof_{5};
  double gripper_offset_z_{0.35};
  std::string input_type_{"xyz"};
  std::string ik_solver_type_{"official_3d"};
  double ik_epsilon_relaxed_3d_{0.005};
  double ik_epsilon_relaxed_6d_{0.01};
  double ik_damping_3d_{1e-12};
  double max_velocity_{1.0};
  double max_acceleration_{2.0};
  double max_jerk_{5.0};
  double control_frequency_{50.0};
  std::vector<double> joint_limits_min_;
  std::vector<double> joint_limits_max_;
  // Optional diagnostics file
  std::ofstream log_ofs_;
  bool log_to_file_{false};
  // SVD damped pseudo-inverse / clamp parameters
  bool use_svd_damped_{true};
  double ik_svd_damping_{1e-6};
  double max_delta_{0.03};
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<IkNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
