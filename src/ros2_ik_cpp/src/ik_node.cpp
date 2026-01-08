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
#include <chrono>

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
    // remember initial damping so we can reset it at the start of each solve
    initial_ik_svd_damping_ = ik_svd_damping_;
    max_delta_ = this->declare_parameter<double>("planning.max_delta", 0.03); // rad, per-iteration max joint change
    // remember initial max_delta so we can reset it per solve
    initial_max_delta_ = max_delta_;

    // LM-style damping bounds and factors
    ik_svd_damping_min_ = this->declare_parameter<double>("planning.ik_svd_damping_min", 1e-12);
    ik_svd_damping_max_ = this->declare_parameter<double>("planning.ik_svd_damping_max", 1e6);
    ik_svd_damping_reduce_factor_ = this->declare_parameter<double>("planning.ik_svd_damping_reduce_factor", 0.1);
    ik_svd_damping_increase_factor_ = this->declare_parameter<double>("planning.ik_svd_damping_increase_factor", 10.0);
    ik_svd_truncation_tol_ = this->declare_parameter<double>("planning.ik_svd_trunc_tol", 1e-6); // relative to smax
    ik_svd_min_relative_reduction_ = this->declare_parameter<double>("planning.ik_svd_min_rel_reduction", 1e-8);

    // Hybrid numeric-J fallback parameters (trigger numeric-J after several LS rejects)
    numeric_fallback_after_rejects_ = this->declare_parameter<int>("planning.numeric_fallback_after_rejects", 3);
    numeric_fallback_duration_ = this->declare_parameter<int>("planning.numeric_fallback_duration", 10);
    debug_log_predictions_ = this->declare_parameter<bool>("planning.debug_log_predictions", true);

    // targeted null-space penalty parameters: if joint4 is near zero and nullspace has joint5/joint7 opposite signs,
    // add a small penalty along that null direction to avoid the degenerate combination.
    joint4_penalty_threshold_ = this->declare_parameter<double>("planning.joint4_penalty_threshold", 0.05); // radians
    nullspace_penalty_scale_ = this->declare_parameter<double>("planning.nullspace_penalty_scale", 1e-4);

    max_velocity_ = this->declare_parameter<double>("planning.max_velocity", 1.0);
    max_acceleration_ = this->declare_parameter<double>("planning.max_acceleration", 2.0);
    max_jerk_ = this->declare_parameter<double>("planning.max_jerk", 5.0);
    control_frequency_ = this->declare_parameter<double>("planning.control_frequency", 50.0);

    joint_limits_min_ = this->declare_parameter<std::vector<double>>("planning.joint_limits_min", std::vector<double>{-2.96, -2.18, -2.96, -3.13, -2.96, -1.83, -1.83});
    joint_limits_max_ = this->declare_parameter<std::vector<double>>("planning.joint_limits_max", std::vector<double>{2.96, 2.18, 2.96, 0, 2.96, 1.83, 1.83});

    // Optional: file to write IK diagnostics (controlled by planning.log_to_file and planning.log_file)
    bool log_to_file_param = this->declare_parameter<bool>("planning.log_to_file", false);
    std::string log_file = this->declare_parameter<std::string>("planning.log_file", std::string());
    if (log_to_file_param && !log_file.empty()) {
      log_ofs_.open(log_file, std::ios::out | std::ios::app);
      if (!log_ofs_) {
        RCLCPP_WARN(this->get_logger(), "Failed to open IK diagnostics log file: %s", log_file.c_str());
      } else {
        log_to_file_ = true;
        RCLCPP_DEBUG(this->get_logger(), "IK diagnostics will be written to: %s", log_file.c_str());
      }
    } else {
      log_to_file_ = false;
      RCLCPP_DEBUG(this->get_logger(), "IK diagnostics logging disabled by planning.log_to_file or empty planning.log_file");
    }

    // Optional: separate file to record per-iteration q_solution and singular values
    std::string svd_q_log_file = this->declare_parameter<std::string>("planning.svd_q_log_file", std::string());
    if (log_to_file_param && !svd_q_log_file.empty()) {
      svd_q_ofs_.open(svd_q_log_file, std::ios::out | std::ios::app);
      if (!svd_q_ofs_) {
        RCLCPP_WARN(this->get_logger(), "Failed to open SVD/Q diagnostics file: %s", svd_q_log_file.c_str());
      } else {
        svd_q_log_to_file_ = true;
        RCLCPP_DEBUG(this->get_logger(), "SVD/Q diagnostics will be written to: %s", svd_q_log_file.c_str());
      }
    } else {
      svd_q_log_to_file_ = false;
    }

    // Optional: separate diagnostics log file to capture verbose IK/joint/publish events
    std::string diagnostics_log_file = this->declare_parameter<std::string>("planning.diagnostics_log_file", std::string());
    if (log_to_file_param && !diagnostics_log_file.empty()) {
      diag_ofs_.open(diagnostics_log_file, std::ios::out | std::ios::app);
      if (!diag_ofs_) {
        RCLCPP_WARN(this->get_logger(), "Failed to open diagnostics log file: %s", diagnostics_log_file.c_str());
      } else {
        diag_log_to_file_ = true;
        RCLCPP_DEBUG(this->get_logger(), "Diagnostics will be written to: %s", diagnostics_log_file.c_str());
      }
    } else {
      diag_log_to_file_ = false;
    }

    // Optional: separate file to record null-space penalty trigger events (to avoid DEBUG spam)
    std::string nullspace_log_file = this->declare_parameter<std::string>("planning.nullspace_log_file", std::string());
    if (log_to_file_param && !nullspace_log_file.empty()) {
      null_ns_ofs_.open(nullspace_log_file, std::ios::out | std::ios::app);
      if (!null_ns_ofs_) {
        RCLCPP_WARN(this->get_logger(), "Failed to open null-space penalty log file: %s", nullspace_log_file.c_str());
      } else {
        null_ns_log_to_file_ = true;
        RCLCPP_DEBUG(this->get_logger(), "Null-space penalty events will be written to: %s", nullspace_log_file.c_str());
      }
    } else {
      null_ns_log_to_file_ = false;
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

    // cache q indices for joints we may want to target for null-space penalty (joint4, joint5, joint7)
    // also include joint3 index for diagnostics / alternative penalty pair
    j4_q_index_ = j5_q_index_ = j7_q_index_ = j3_q_index_ = -1;
    try {
      int j4_id = model_.getJointId("joint4");
      if (j4_id >= 0 && j4_id < (int)model_.joints.size()) {
        int idx = model_.joints[j4_id].idx_q();
        if (idx >= 0) j4_q_index_ = idx;
      }
    } catch(...) { }
    try {
      int j3_id = model_.getJointId("joint3");
      if (j3_id >= 0 && j3_id < (int)model_.joints.size()) {
        int idx = model_.joints[j3_id].idx_q();
        if (idx >= 0) j3_q_index_ = idx;
      }
    } catch(...) { }
    try {
      int j5_id = model_.getJointId("joint5");
      if (j5_id >= 0 && j5_id < (int)model_.joints.size()) {
        int idx = model_.joints[j5_id].idx_q();
        if (idx >= 0) j5_q_index_ = idx;
      }
    } catch(...) { }
    try {
      int j7_id = model_.getJointId("joint7");
      if (j7_id >= 0 && j7_id < (int)model_.joints.size()) {
        int idx = model_.joints[j7_id].idx_q();
        if (idx >= 0) j7_q_index_ = idx;
      }
    } catch(...) { }

    // Diagnostic: print model sizes and tip frame info
    RCLCPP_DEBUG(this->get_logger(), "Pinocchio model loaded: nq=%d nv=%d tip_frame='%s' id=%u",
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

    RCLCPP_DEBUG(this->get_logger(), "ik_node initialized, tip_frame=%s", tip_frame_name_.c_str());
  }

private:
  static std::string readFileToString(const std::string &path) {
    std::ifstream ifs(path);
    if (!ifs) return std::string();
    std::ostringstream ss; ss << ifs.rdbuf();
    return ss.str();
  }

  // Clamp an Eigen vector of joint positions to the configured joint_limits arrays
  void clampToJointLimits(Eigen::VectorXd &q) {
    if (joint_limits_min_.size() == (size_t)q.size() && joint_limits_max_.size() == (size_t)q.size()) {
      for (int i = 0; i < q.size(); ++i) {
        double lo = joint_limits_min_[i];
        double hi = joint_limits_max_[i];
        if (lo >= hi) continue; // skip malformed limits
        if (q[i] < lo) q[i] = lo;
        if (q[i] > hi) q[i] = hi;
      }
    }
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
        RCLCPP_DEBUG(this->get_logger(), "FK init pose: pos=(%.4f, %.4f, %.4f) quat=(%.4f, %.4f, %.4f, %.4f) rpy=(%.4f, %.4f, %.4f)",
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

    // start timing for this spinOnce invocation (measure compute + IK solve + publish)
    auto __spin_start = std::chrono::steady_clock::now();

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
      if (log_to_file_ && log_ofs_) {
        log_ofs_ << "FK_CUR pos " << cur_fk.translation().transpose() << " quat " << qcur.coeffs().transpose() << " rpy " << rpy_cur.transpose() << std::endl;
      }
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
      // clamp final solution to joint limits before publishing
      clampToJointLimits(q_sol);

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

    // end timing and report
    auto __spin_end = std::chrono::steady_clock::now();
    double __spin_elapsed_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(__spin_end - __spin_start).count();
    const char *status_str = "失败";
    if (last_solve_status_ == 1) status_str = "精准成功";
    else if (last_solve_status_ == 2) status_str = "粗略成功";
    RCLCPP_INFO(this->get_logger(), "spinOnce elapsed: %.3f ms [结果: %s]", __spin_elapsed_ms, status_str);
    if (log_to_file_ && log_ofs_) {
      log_ofs_ << "SPINONCE_TIME_MS " << __spin_elapsed_ms << " STATUS " << status_str << std::endl;
    }
  }

  bool solveIK6D(const SE3 &target_pose, const VectorXd &q_init, VectorXd &q_solution) {
    // reset last_solve_status_ to failure by default
    last_solve_status_ = 0;
    // reset per-solve damping to the configured initial value so adjustments
    // within one solve (increase/decrease) do not leak to subsequent solves
    ik_svd_damping_ = initial_ik_svd_damping_;
    // reset adaptive max_delta and saturation counter for this solve
    max_delta_ = initial_max_delta_;
    consecutive_max_damping_rejections_ = 0;
    // hybrid fallback state for this solve
    int consecutive_ls_rejects = 0;
    int numeric_fallback_remaining = 0;
    bool numeric_force_active = false;
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
        RCLCPP_DEBUG(this->get_logger(), "IK initial err6 = [%.6f, %.6f, %.6f, %.6f, %.6f, %.6f] (norm=%.6f)",
                    err6(0), err6(1), err6(2), err6(3), err6(4), err6(5), cur_err);
        if (log_to_file_ && log_ofs_) {
          log_ofs_ << "ITER 0 INIT_ERR6 " << err6.transpose() << " NORM " << cur_err << std::endl;
        }
        printed_initial = true;
      }

      RCLCPP_DEBUG(this->get_logger(), "IK iter=%d cur_err=%.8f err6=[%.6f %.6f %.6f | %.6f %.6f %.6f]", it, cur_err,
                   err6(0), err6(1), err6(2), err6(3), err6(4), err6(5));

      if (cur_err < best_error) { best_error = cur_err; best_q = q_solution; }
      if (cur_err < eps_) { last_solve_status_ = 1; clampToJointLimits(q_solution); return true; }

      // Ensure Pinocchio has computed joint Jacobians for the current configuration
      pinocchio::computeJointJacobians(model_, data, q_solution);
      // updateFramePlacements already called above; now get the frame Jacobian
      // analytic Jacobian expressed in WORLD frame (twist: [v_world; omega_world])
      // Using WORLD makes the linear part correspond to the position error expressed in world coords
      MatrixXd J_world = pinocchio::getFrameJacobian(model_, data, tip_frame_id_, pinocchio::WORLD);

      // Map analytic Jacobian to the error representation used by this solver.
      // err6 = [pos_err; ang_err], where ang_err is rotation vector (axis * angle) = log(R_target * R_current^T).
      // Build a numeric 6x6 mapping M that maps spatial twist (v_world, omega_world)
      // to the rate of change of our error coordinates err6. This captures cross-terms
      // between translation and rotation that a simple block-diagonal invJl misses.
      ang_err = err6.template tail<3>();

      // Numeric approximation of 6x6 left-Jacobian mapping at the current end-effector pose
      Eigen::Matrix<double,6,6> M = Eigen::Matrix<double,6,6>::Zero();
      const double eps_se3 = 1e-6; // small spatial perturbation
      // current SE3 of the frame
      SE3 cur_se3 = current_pose;
      for (int k = 0; k < 6; ++k) {
        // construct small twist in world frame: first 3 = translation perturbation, last 3 = angular perturbation
        Eigen::Matrix<double,6,1> xi = Eigen::Matrix<double,6,1>::Zero();
        xi(k) = eps_se3;
        Eigen::Vector3d dv = xi.template head<3>();
        Eigen::Vector3d dw = xi.template tail<3>();

        // build perturbed pose: apply small rotation (from dw) then small translation dv (both in world frame)
        Eigen::Quaterniond q_delta;
        double ang = dw.norm();
        if (ang < 1e-12) {
          q_delta = Eigen::Quaterniond::Identity();
        } else {
          Eigen::Vector3d axis = dw / ang;
          q_delta = Eigen::Quaterniond(Eigen::AngleAxisd(ang, axis));
        }
        Eigen::Quaterniond qcur(cur_se3.rotation());
        Eigen::Quaterniond qpert = q_delta * qcur; qpert.normalize();
        Eigen::Vector3d tpert = cur_se3.translation() + dv;
        SE3 pert_se3(qpert, tpert);

        // compute err6 at perturbed pose (same mapping used elsewhere)
        Eigen::Vector3d pos_err_pert = target_pose.translation() - pert_se3.translation();
        Eigen::Quaterniond qcur_pert(pert_se3.rotation());
        Eigen::Quaterniond qerr_pert = qtgt * qcur_pert.conjugate(); qerr_pert.normalize();
        Eigen::AngleAxisd aa_pert(qerr_pert);
        Eigen::Vector3d ang_err_pert = Eigen::Vector3d::Zero();
        double angle_pert = aa_pert.angle();
        if (std::isfinite(angle_pert) && std::abs(angle_pert) > 1e-12) ang_err_pert = aa_pert.axis() * angle_pert;

        Eigen::Matrix<double,6,1> err6_pert;
        err6_pert.template head<3>() = pos_err_pert;
        err6_pert.template tail<3>() = ang_err_pert;

        // column k approximates (d err / d xi_k)
        M.col(k) = (err6_pert - err6) / eps_se3;
      }

      // analytic Jacobian that maps q_dot -> err_dot (err = target - current)
      // J_world maps q_dot -> twist expressed in world frame; so err_dot ≈ M * twist_world
      MatrixXd J_err = - M * J_world;

      // Optionally compute numeric Jacobian (finite-difference of err6) once for either
      // diagnostics or to use directly as the Jacobian for solving.
      const bool run_numeric_check_early = true; // always compute when requested
      MatrixXd numJ = MatrixXd::Zero(6, model_.nv);
      bool numeric_computed = false;
      if (use_numeric_jacobian_ || run_numeric_check_early) {
        const double eps_q = 1e-6;
        for (int j = 0; j < (int)model_.nv; ++j) {
          Eigen::VectorXd dq = Eigen::VectorXd::Zero(model_.nv);
          dq[j] = eps_q;
          Eigen::VectorXd q_pert = Eigen::VectorXd::Zero(q_solution.size());
          pinocchio::integrate(model_, q_solution, dq, q_pert);

          Data data_pert(model_);
          pinocchio::forwardKinematics(model_, data_pert, q_pert);
          pinocchio::updateFramePlacements(model_, data_pert);
          const SE3 &pose_pert = data_pert.oMf[tip_frame_id_];

          // compute err6 at perturbed configuration using same mapping as above
          Eigen::Vector3d pos_err_pert = target_pose.translation() - pose_pert.translation();
          Eigen::Quaterniond qcur_pert(pose_pert.rotation());
          Eigen::Quaterniond qerr_pert = qtgt * qcur_pert.conjugate();
          qerr_pert.normalize();
          Eigen::AngleAxisd aa_pert(qerr_pert);
          Eigen::Vector3d ang_err_pert = Eigen::Vector3d::Zero();
          double angle_pert = aa_pert.angle();
          if (std::isfinite(angle_pert) && std::abs(angle_pert) > 1e-12) ang_err_pert = aa_pert.axis() * angle_pert;

          Eigen::Matrix<double,6,1> err6_pert;
          err6_pert.template head<3>() = pos_err_pert;
          err6_pert.template tail<3>() = ang_err_pert;

          numJ.col(j) = (err6_pert - err6) / eps_q;
        }
        numeric_computed = true;
      }

      // Hybrid selection: allow numeric forcing when fallback is active, otherwise respect parameter
      MatrixXd J_used = ((use_numeric_jacobian_ && numeric_computed) || numeric_force_active) ? numJ : J_err;

      // If Jacobian appears all-zero, warn once — indicates model/joint mismatch or fixed frame
      static bool jacobian_zero_warned = false;
      if (!jacobian_zero_warned && J_used.norm() == 0.0) {
        RCLCPP_WARN(this->get_logger(), "Jacobian (in error coords) is zero matrix. Check model joint mapping, tip frame, and that computeJointJacobians was called.");
        jacobian_zero_warned = true;
      }

      // compute singular values and condition estimate for diagnosis
      // Apply row weights to balance position (meters) and angle (radians)
      Eigen::Matrix<double,6,6> W = Eigen::Matrix<double,6,6>::Identity();
      W(0,0) = pos_weight_; W(1,1) = pos_weight_; W(2,2) = pos_weight_;
      W(3,3) = ang_weight_; W(4,4) = ang_weight_; W(5,5) = ang_weight_;
      MatrixXd Jw = J_used;
      for (int ri = 0; ri < 6; ++ri) Jw.row(ri) *= W(ri,ri);
      Eigen::Matrix<double,6,1> errw = err6;
      errw.template head<3>() *= pos_weight_;
      errw.template tail<3>() *= ang_weight_;

      Eigen::JacobiSVD<MatrixXd> svd(Jw, Eigen::ComputeThinU | Eigen::ComputeThinV);
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
      RCLCPP_DEBUG(this->get_logger(), "Weighted Jacobian singular values (%d) = %s cond=%.3e pos_w=%.3e ang_w=%.3e", sn, ss.str().c_str(), cond, pos_weight_, ang_weight_);
      if (log_to_file_ && log_ofs_) {
        log_ofs_ << "ITER " << it << " SVD ";
        for (int ii = 0; ii < sn; ++ii) { if (ii) log_ofs_ << ","; log_ofs_ << s(ii); }
        log_ofs_ << " COND " << cond << " ERR " << cur_err << " POS_W " << pos_weight_ << " ANG_W " << ang_weight_ << std::endl;
      }

       // Compute joint-space velocity v using Levenberg-Marquardt (LM) style solve
      if (use_svd_damped_ && sn > 0) {
        // Build approximate normal equations: A = Jw^T Jw
        MatrixXd A = Jw.transpose() * Jw; // nv x nv
        Eigen::VectorXd g = Jw.transpose() * errw; // gradient (nv x 1)

        // targeted null-space penalty: only apply when joint4 is near zero and the null vector
        // has joint5 and joint7 components with opposite sign (indicating the problematic combination)
        double alpha_ns = 0.0;
        if (sn > 0) {
          Eigen::VectorXd ns_vec = svd.matrixV().col(sn-1); // smallest-singular-value null vector
          // read components (guard indices)
          double c3 = (j3_q_index_ >= 0) ? ns_vec[j3_q_index_] : 0.0;
          double c5 = (j5_q_index_ >= 0) ? ns_vec[j5_q_index_] : 0.0;
          double c7 = (j7_q_index_ >= 0) ? ns_vec[j7_q_index_] : 0.0;

          bool pair_j5j7 = (j5_q_index_ >= 0 && j7_q_index_ >= 0) && (c5 * c7 < 0.0) && (std::abs(c5) > 1e-6) && (std::abs(c7) > 1e-6);
          bool pair_j3j5 = (j3_q_index_ >= 0 && j5_q_index_ >= 0) && (c3 * c5 < 0.0) && (std::abs(c3) > 1e-6) && (std::abs(c5) > 1e-6);

          if (j4_q_index_ >= 0 && std::abs(q_solution[j4_q_index_]) < joint4_penalty_threshold_ && (pair_j5j7 || pair_j3j5)) {
            double smax = (s.size() > 0) ? s(0) : 1.0;
            alpha_ns = nullspace_penalty_scale_ * (smax * smax);
            const char *which = pair_j5j7 ? "j5/j7" : "j3/j5";
            RCLCPP_DEBUG(this->get_logger(), "Applying null-space penalty alpha=%.3e triggered_by=%s q4=%.3e ns_comp(j3,j5,j7)=(%.3e, %.3e, %.3e)", alpha_ns, which, q_solution[j4_q_index_], c3, c5, c7);
            if (null_ns_log_to_file_ && null_ns_ofs_) {
              null_ns_ofs_ << "ITER " << it << " ALPHA " << alpha_ns << " TRIG " << which << " Q4 " << q_solution[j4_q_index_]
                           << " NS " << c3 << " " << c5 << " " << c7 << std::endl;
            }
          }

          if (alpha_ns > 0.0) {
            A += alpha_ns * (ns_vec * ns_vec.transpose());
          }
        }

        // LM damping parameter stored in ik_svd_damping_. We'll treat it as absolute lambda here.
        double lambda = std::max(ik_svd_damping_min_, std::min(ik_svd_damping_, ik_svd_damping_max_));
        const int lm_max_attempts = 8;
        double predicted_reduction = -1.0;
        Eigen::VectorXd v_candidate = VectorXd::Zero(model_.nv);

        // We will try to find a lambda that yields positive predicted reduction
        for (int attempt = 0; attempt < lm_max_attempts; ++attempt) {
          // build damped system: (A + lambda I) v = -g
          MatrixXd Ad = A;
          Ad.diagonal().array() += lambda;
          Eigen::VectorXd rhs = -g;

          // solve (use LDLT for symmetric positive-definite)
          Eigen::LDLT<MatrixXd> ldlt(Ad);
          if (ldlt.info() != Eigen::Success) {
            // increase damping and retry
            lambda *= ik_svd_damping_increase_factor_;
            if (lambda > ik_svd_damping_max_) lambda = ik_svd_damping_max_;
            continue;
          }
          v_candidate = ldlt.solve(rhs);

          // predicted reduction (in weighted error norm) for full step v_candidate
          Eigen::VectorXd pred_errw = errw + Jw * v_candidate;
          double errw_norm2 = errw.squaredNorm();
          double pred_errw_norm2 = pred_errw.squaredNorm();
          predicted_reduction = 0.5 * (errw_norm2 - pred_errw_norm2);

          if (predicted_reduction > 0) {
            // found acceptable lambda
            break;
          }
          // otherwise increase damping and retry
          lambda *= ik_svd_damping_increase_factor_;
          if (lambda > ik_svd_damping_max_) { lambda = ik_svd_damping_max_; break; }
        }

        // commit chosen lambda back to adaptive damping variable (so subsequent iterations use updated value)
        double old_d = ik_svd_damping_;
        ik_svd_damping_ = std::max(ik_svd_damping_min_, std::min(lambda, ik_svd_damping_max_));

        v = v_candidate;

        if (log_to_file_ && log_ofs_) {
          log_ofs_ << "ITER " << it << " LM_LAMBDA " << ik_svd_damping_ << " PRED_REDUCTION " << predicted_reduction << "\n";
        }
        RCLCPP_DEBUG(this->get_logger(), "IK iter=%d LM lambda=%.3e pred_reduction=%.3e", it, ik_svd_damping_, predicted_reduction);

      } else {
        // fallback to previous JJt-based solve if SVD/LM is disabled or rank==0
        MatrixXd JJt = J_err * J_err.transpose();
        JJt.diagonal().array() += damp;
        // Solve for ∆q that reduces the error using analytic J_err
        v = J_err.transpose() * JJt.ldlt().solve(err6);
      }

      // compute full delta in tangent space
      Eigen::VectorXd delta_full = v * DT; // tangent-space step

      // Diagnostic / logging for numeric vs analytic Jacobian (use precomputed numJ if available)
      if (numeric_computed) {
        double nnum = numJ.norm();
        double nJ_err = J_err.norm();
        // also compute Jacobian expressed in WORLD frame for comparison
        double nJ_world = J_world.norm();
        double ndiff_err = (numJ - J_err).norm();
        double ndiff_world = (numJ - J_world).norm();
        double ndiff_sign_err = (numJ + J_err).norm();
        double ndiff_sign_world = (numJ + J_world).norm();
        double pred_err_norm = (J_err * delta_full).norm();
        double pred_err_world_norm = (J_world * delta_full).norm();
        double num_pred_err_norm = (numJ * delta_full).norm();

        if (log_to_file_ && log_ofs_) {
          log_ofs_ << "DEBUG_NUMJ norm_num=" << nnum << " norm_J_err=" << nJ_err << " norm_J_world=" << nJ_world
                   << " norm(diff=num-J_err)=" << ndiff_err << " norm(diff=num-J_world)=" << ndiff_world
                   << " norm(diff=num+J_err)=" << ndiff_sign_err << " norm(diff=num+J_world)=" << ndiff_sign_world << std::endl;
          log_ofs_ << "DEBUG_PRED pred_norm(J_err*delta)=" << pred_err_norm << " pred_norm(J_world*delta)=" << pred_err_world_norm
                   << " pred_norm(numJ*delta)=" << num_pred_err_norm << " delta_norm=" << delta_full.norm() << std::endl;
          for (int j = 0; j < std::min(6, (int)model_.nv); ++j) {
            log_ofs_ << "DEBUG_NUMJ col" << j << " num=" << numJ.col(j).transpose()
                     << " J_err=" << J_err.col(j).transpose() << " J_world=" << J_world.col(j).transpose() << std::endl;
          }
        }

        RCLCPP_DEBUG(this->get_logger(), "DEBUG_NUMJ nnum=%.6e nJ_err=%.6e nJ_world=%.6e diff_err=%.6e diff_world=%.6e pred_err=%.6e pred_world=%.6e pred_num=%.6e",
                    nnum, nJ_err, nJ_world, ndiff_err, ndiff_world, pred_err_norm, pred_err_world_norm, num_pred_err_norm);
      }

      // Line-search / backtracking along delta_full on the manifold
      bool accepted = false;
      Eigen::VectorXd q_candidate(q_solution.size());
      double best_trial_err = cur_err;
      Eigen::VectorXd best_trial_q = q_solution;
      const int max_ls = 6; // tries: alpha = 1, 1/2, 1/4, ...
      for (int ls = 0; ls < max_ls; ++ls) {
        double alpha = std::pow(0.5, ls);
        Eigen::VectorXd delta_try = delta_full * alpha;
        // clamp per-joint magnitude
        for (int i = 0; i < delta_try.size(); ++i) delta_try[i] = std::max(-max_delta_, std::min(max_delta_, delta_try[i]));

        // --- NEW: per-joint delta limiting to avoid exceeding joint limits
        // If joint limits are configured and match q_solution size, scale/clip delta_try
        if (joint_limits_min_.size() == (size_t)q_solution.size() && joint_limits_max_.size() == (size_t)q_solution.size()) {
          // compute tentative q_try by applying delta_try in tangent space (approx via direct add)
          Eigen::VectorXd q_try_est = q_solution + delta_try; // approximate candidate (works for revolute additive convention here)
          // clamp estimated q to limits and recompute delta_try = q_try_est - q_solution
          for (int i = 0; i < q_try_est.size(); ++i) {
            double lo = joint_limits_min_[i];
            double hi = joint_limits_max_[i];
            if (lo >= hi) continue; // skip malformed limit
            if (q_try_est[i] < lo) q_try_est[i] = lo;
            if (q_try_est[i] > hi) q_try_est[i] = hi;
          }
          // replace delta_try with the (possibly reduced) step that does not cross limits
          delta_try = q_try_est - q_solution;
        }
        // --- END NEW

        // integrate to get candidate configuration
        Eigen::VectorXd q_try(q_solution.size());
        pinocchio::integrate(model_, q_solution, delta_try, q_try);
        // compute error at q_try
        Data data_try(model_);
        pinocchio::forwardKinematics(model_, data_try, q_try);
        pinocchio::updateFramePlacements(model_, data_try);
        const SE3 &pose_try = data_try.oMf[tip_frame_id_];
        Eigen::Vector3d pos_err_try = target_pose.translation() - pose_try.translation();
        Eigen::Quaterniond qcur_try(pose_try.rotation());
        Eigen::Quaterniond qerr_try = qtgt * qcur_try.conjugate();
        qerr_try.normalize();
        Eigen::AngleAxisd aa_try(qerr_try);
        Eigen::Vector3d ang_err_try = Eigen::Vector3d::Zero();
        double angle_try = aa_try.angle();
        if (std::isfinite(angle_try) && std::abs(angle_try) > 1e-12) ang_err_try = aa_try.axis() * angle_try;
        Eigen::Matrix<double,6,1> err6_try;
        err6_try.template head<3>() = pos_err_try;
        err6_try.template tail<3>() = ang_err_try;
        double err_try = err6_try.norm();

        if (log_to_file_ && log_ofs_) {
          // also log linearized prediction vs actual try error for diagnosis
          double predicted_norm = -1.0;
          double predicted_reduction = -1.0;
          double actual_reduction = -1.0;
          double rho = -1.0;
          if (J_used.size() != 0) {
            // predicted err6 after applying delta_try by linearization: err6 + J_used * delta_try
            Eigen::Matrix<double,6,1> pred_err6 = err6 + J_used * delta_try;
            predicted_norm = pred_err6.norm();
            // compute predicted reduction in weighted norm using Jw
            Eigen::Matrix<double,6,1> pred_errw = errw + Jw * delta_try;
            predicted_reduction = 0.5 * (errw.squaredNorm() - pred_errw.squaredNorm());
          }
          // actual reduction (weighted)
          Eigen::Matrix<double,6,1> errw_try = err6_try;
          errw_try.template head<3>() *= pos_weight_;
          errw_try.template tail<3>() *= ang_weight_;
          actual_reduction = 0.5 * (errw.squaredNorm() - errw_try.squaredNorm());
          if (predicted_reduction > 1e-16) rho = actual_reduction / predicted_reduction; else rho = -1.0;

          log_ofs_ << "ITER " << it << " LS_TRY alpha=" << alpha << " ERR=" << err_try
                   << " PRED_NORM=" << predicted_norm
                   << " PRED_REDUCTION=" << predicted_reduction
                   << " ACT_REDUCTION=" << actual_reduction
                   << " RHO=" << rho
                   << " LAMBDA=" << ik_svd_damping_ << std::endl;
          RCLCPP_DEBUG(this->get_logger(), "ITER %d LS_TRY alpha=%.6f ERR=%.6e PRED_NORM=%.6e PRED_RED=%.6e ACT_RED=%.6e RHO=%.6e LAMBDA=%.6e",
                       it, alpha, err_try, predicted_norm, predicted_reduction, actual_reduction, rho, ik_svd_damping_);
         }

        if (err_try < cur_err) {
          accepted = true;
          q_candidate = q_try;
          best_trial_err = err_try;
          best_trial_q = q_try;
          if (log_to_file_ && log_ofs_) log_ofs_ << "ITER " << it << " LS_ACCEPT alpha=" << alpha << " ERR=" << err_try << std::endl;
          break;
        } else {
          if (err_try < best_trial_err) { best_trial_err = err_try; best_trial_q = q_try; }
        }
      }

      if (accepted) {
        // accepted -> clear LS reject counter and, if fallback active, reduce its remaining duration
        consecutive_ls_rejects = 0;
        if (numeric_force_active && numeric_fallback_remaining > 0) {
          --numeric_fallback_remaining;
          if (numeric_fallback_remaining <= 0) {
            numeric_force_active = false;
            if (log_to_file_ && log_ofs_) log_ofs_ << "ITER " << it << " NUMERIC_FALLBACK_END" << std::endl;
          }
        }
         q_solution = q_candidate;
         double max_abs_dq = 0.0;
         if (delta_full.size() > 0) max_abs_dq = (delta_full.cwiseAbs() * 1.0).maxCoeff();
         // compute component norms for diagnostics
         double pos_norm = err6.template head<3>().norm();
         double ang_norm = err6.template tail<3>().norm();
         RCLCPP_DEBUG(this->get_logger(), "IK iter=%d accepted line-search step, trial_err=%.6e max_abs_dq=%.6f pos=%.6e ang=%.6e", it, best_trial_err, max_abs_dq, pos_norm, ang_norm);

        // relative reduction
        double rel_red = (cur_err - best_trial_err) / std::max(cur_err, 1e-12);
        // Decrease damping only if significant relative reduction achieved
        if (rel_red > ik_svd_min_relative_reduction_) {
          double old_d = ik_svd_damping_;
          ik_svd_damping_ = std::max(ik_svd_damping_min_, ik_svd_damping_ * ik_svd_damping_reduce_factor_);
          if (log_to_file_ && log_ofs_) log_ofs_ << "ITER " << it << " LS_ACCEPT decrease_damping " << old_d << " -> " << ik_svd_damping_ << " rel_red=" << rel_red << std::endl;
          RCLCPP_DEBUG(this->get_logger(), "IK iter=%d decreased damping %.3e -> %.3e (rel_red=%.3e)", it, old_d, ik_svd_damping_, rel_red);
        } else {
          if (log_to_file_ && log_ofs_) log_ofs_ << "ITER " << it << " LS_ACCEPT no_significant_reduction rel_red=" << rel_red << std::endl;
        }

      } else {
        // rejection: increment LS reject counter and possibly trigger numeric fallback
        ++consecutive_ls_rejects;
        if (!numeric_force_active && numeric_computed && consecutive_ls_rejects >= numeric_fallback_after_rejects_) {
          numeric_force_active = true;
          numeric_fallback_remaining = numeric_fallback_duration_;
          if (log_to_file_ && log_ofs_) log_ofs_ << "ITER " << it << " NUMERIC_FALLBACK_START remaining=" << numeric_fallback_remaining << std::endl;
          RCLCPP_WARN(this->get_logger(), "Numeric-J fallback activated for %d iterations due to repeated LS rejects", numeric_fallback_duration_);
        }
        // If none of the alphas improved but the best trial is effectively equal (within tiny tol)
        // and produces a measurable configuration change, accept it to escape stagnation.
        double tiny_tol = 1e-10;
        double movement = (best_trial_q - q_solution).norm();
        if (best_trial_err <= cur_err + tiny_tol && movement > 1e-8) {
          q_solution = best_trial_q;
          if (log_to_file_ && log_ofs_) log_ofs_ << "ITER " << it << " LS_ACCEPT_BEST_TOL ERR=" << best_trial_err << " movement=" << movement << std::endl;
          RCLCPP_DEBUG(this->get_logger(), "IK iter=%d accepted best_trial within tol (err=%.6e movement=%.6e)", it, best_trial_err, movement);
          // modestly reduce damping to allow further progress
          double old_d = ik_svd_damping_;
          ik_svd_damping_ = std::max(ik_svd_damping_min_, ik_svd_damping_ * ik_svd_damping_reduce_factor_);
          if (log_to_file_ && log_ofs_) log_ofs_ << "ITER " << it << " LS_ACCEPT_BEST_TOL decrease_damping " << old_d << " -> " << ik_svd_damping_ << std::endl;
          // reset saturation counter
          consecutive_max_damping_rejections_ = 0;
        } else {
          // no trial reduced error -> increase damping to be more conservative next time
          double old_d = ik_svd_damping_;
          ik_svd_damping_ *= ik_svd_damping_increase_factor_;
          // clamp to maximum to avoid overflow
          if (ik_svd_damping_ > ik_svd_damping_max_) ik_svd_damping_ = ik_svd_damping_max_;

          // Track consecutive saturations of damping and adapt max_delta_ if stuck
          if (old_d >= ik_svd_damping_max_ * 0.999) {
            ++consecutive_max_damping_rejections_;
          } else {
            consecutive_max_damping_rejections_ = 0;
          }

          if (consecutive_max_damping_rejections_ >= consecutive_shrink_after_) {
            double old_max_delta = max_delta_;
            max_delta_ = std::max(max_delta_min_, max_delta_ * 0.5);
            // also try to relax damping a bit so small steps can be attempted
            double old_d2 = ik_svd_damping_;
            ik_svd_damping_ = std::max(ik_svd_damping_min_, ik_svd_damping_ * ik_svd_damping_reduce_factor_);
            consecutive_max_damping_rejections_ = 0;
            if (log_to_file_ && log_ofs_) log_ofs_ << "ITER " << it << " LS_REJECTED saturated->shrink_max_delta " << old_max_delta << " -> " << max_delta_
                                                   << " and reduce_damping " << old_d2 << " -> " << ik_svd_damping_ << std::endl;
            RCLCPP_DEBUG(this->get_logger(), "IK iter=%d damping saturated, shrink max_delta %.3e -> %.3e and reduce damping %.3e -> %.3e", it, old_max_delta, max_delta_, old_d2, ik_svd_damping_);
          } else {
            if (log_to_file_ && log_ofs_) log_ofs_ << "ITER " << it << " LS_REJECTED increase_damping " << old_d << " -> " << ik_svd_damping_ << std::endl;
            RCLCPP_DEBUG(this->get_logger(), "IK iter=%d line-search rejected, increasing ik_svd_damping %.3e -> %.3e (consec=%d)", it, old_d, ik_svd_damping_, consecutive_max_damping_rejections_);
          }
          // keep q_solution as-is (i.e., reject the step)
        }
      }
    }

    // if best_error small accept
    if (best_error < eps_ * 100.0) {
      q_solution = best_q;
      last_solve_status_ = 2; // consider coarse success
      if (log_to_file_ && log_ofs_) log_ofs_ << "IK_SUCCESS BEST_ERR " << best_error << std::endl;
      clampToJointLimits(q_solution);
      return true;
    }

    // Dual-threshold: if we exhausted iterations but final solution is within a relaxed tolerance,
    // accept it as success. Use 3D relaxed tol when input_type_=="xyz", otherwise 6D.
    {
      double relaxed_eps = (input_type_ == "xyz") ? ik_epsilon_relaxed_3d_ : ik_epsilon_relaxed_6d_;
      // evaluate final error at current q_solution
      Data data_check(model_);
      pinocchio::forwardKinematics(model_, data_check, q_solution);
      pinocchio::updateFramePlacements(model_, data_check);
      const SE3 &cur_pose = data_check.oMf[tip_frame_id_];

      Eigen::Vector3d pos_err = target_pose.translation() - cur_pose.translation();
      Eigen::Quaterniond qcur(cur_pose.rotation());
      Eigen::Quaterniond qtgt(target_pose.rotation());
      Eigen::Quaterniond qerr = qtgt * qcur.conjugate();
      qerr.normalize();
      Eigen::AngleAxisd aa(qerr);
      Eigen::Vector3d ang_err = Eigen::Vector3d::Zero();
      double angle = aa.angle();
      if (std::isfinite(angle) && std::abs(angle) > 1e-12) ang_err = aa.axis() * angle;

      Eigen::Matrix<double,6,1> final_err6;
      final_err6.template head<3>() = pos_err;
      final_err6.template tail<3>() = ang_err;
      double final_err = final_err6.norm();

      if (final_err <= relaxed_eps) {
        // accept as success under relaxed threshold
        last_solve_status_ = 2;
        if (log_to_file_ && log_ofs_) log_ofs_ << "IK_SUCCESS RELAXED final_err=" << final_err << " relaxed_eps=" << relaxed_eps << std::endl;
        RCLCPP_DEBUG(this->get_logger(), "IK accepted by relaxed tolerance (err=%.6e <= relaxed=%.6e)", final_err, relaxed_eps);
        clampToJointLimits(q_solution);
        return true;
      }
    }

    last_solve_status_ = 0;
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
  // Last solve status: 0 = failed, 1 = precise success, 2 = relaxed/coarse success
  int last_solve_status_{0};
  // Additional optional files for SVD/q traces and verbose diagnostics
  std::ofstream svd_q_ofs_;
  bool svd_q_log_to_file_{false};
  std::ofstream diag_ofs_;
  bool diag_log_to_file_{false};
  // SVD damped pseudo-inverse / clamp parameters
  bool use_svd_damped_{true};
  double ik_svd_damping_{1e-6};
  // store the original configured damping so we can reset per solve
  double initial_ik_svd_damping_{1e-6};
  double max_delta_{0.03};

  // LM-style damping bounds and factors
  double ik_svd_damping_min_{1e-12};
  double ik_svd_damping_max_{1e6};
  double ik_svd_damping_reduce_factor_{0.1};
  double ik_svd_damping_increase_factor_{10.0};
  double ik_svd_truncation_tol_{1e-6}; // relative to smax
  double ik_svd_min_relative_reduction_{1e-8};

  // Hybrid numeric-J fallback parameters (trigger numeric-J after several LS rejects)
  int numeric_fallback_after_rejects_{3};
  int numeric_fallback_duration_{10};
  bool debug_log_predictions_{true};

  // store initial adaptive values so each solve can start from configured state
  double initial_max_delta_{0.03};

  // Adaptive shrink when damping saturates
  int consecutive_max_damping_rejections_{0};
  int consecutive_shrink_after_{3};
  double max_delta_min_{1e-6};

  // new fallback parameters
  bool gradient_fallback_enable_ = this->declare_parameter<bool>("planning.gradient_fallback_enable", true);
  double gradient_fallback_alpha_ = this->declare_parameter<double>("planning.gradient_fallback_alpha", 0.02);
  // Option to use finite-difference Jacobian directly for the solver (diagnostic / debugging)
  bool use_numeric_jacobian_ = this->declare_parameter<bool>("planning.use_numeric_jacobian", false);
  // new parameters for balancing position vs orientation residuals
  double pos_weight_ = this->declare_parameter<double>("planning.pos_weight", 1.0);
  double ang_weight_ = this->declare_parameter<double>("planning.ang_weight", 1.0);

  // targeted null-space penalty caching (q indices for joint4,5,7) and tuning
  int j4_q_index_{-1};
  int j5_q_index_{-1};
  int j7_q_index_{-1};
  int j3_q_index_{-1};
  double joint4_penalty_threshold_{0.05};
  double nullspace_penalty_scale_{1e-4};

  // Optional: separate file to record null-space penalty trigger events (to avoid DEBUG spam)
  std::ofstream null_ns_ofs_;
  bool null_ns_log_to_file_{false};
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<IkNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
