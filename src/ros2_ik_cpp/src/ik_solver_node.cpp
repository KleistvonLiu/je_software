#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>

#include "ros2_ik_cpp/ik_solver.hpp"
// pinocchio headers are encapsulated inside ik_solver.hpp / IkSolver - do not include them directly here

#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <mutex>
#include <algorithm>
#include <rcutils/logging.h>
#include <cmath>

using namespace std::chrono_literals;
using ros2_ik_cpp::IkSolver;
using Params = ros2_ik_cpp::IkSolver::Params;
using Eigen::VectorXd;
// Avoid importing pinocchio symbols into this translation unit; use IkSolver APIs and fully-qualify pinocchio types if needed

class IkSolverNode : public rclcpp::Node {
public:
  IkSolverNode(): Node("ik_solver_node") {
    // declare the full set of planning parameters (mirror ik_node defaults)
    int planning_dof = this->declare_parameter<int>("planning.dof", 7);
    // legacy single urdf_path kept for backward compatibility
    urdf_path_ = this->declare_parameter<std::string>("planning.urdf_path", "");
    std::string urdf_path_left = this->declare_parameter<std::string>("planning.urdf_path_left", urdf_path_);
    std::string urdf_path_right = this->declare_parameter<std::string>("planning.urdf_path_right", urdf_path_);

    std::string tip_link = this->declare_parameter<std::string>("planning.tip_link", "Link7");
    std::string tip_link_left = this->declare_parameter<std::string>("planning.tip_link_left", tip_link);
    std::string tip_link_right = this->declare_parameter<std::string>("planning.tip_link_right", tip_link);

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

    // publish deadband parameters (removed - use unconditional publish like ik_node)
    // publish_pose_pos_deadband_ = this->declare_parameter<double>("planning.publish_pose_pos_deadband", 1e-2);
    // publish_pose_ang_deadband_ = this->declare_parameter<double>("planning.publish_pose_ang_deadband", 1e-3);
    // publish_deadband_ = this->declare_parameter<double>("planning.publish_deadband", 1e-4);

    // optional logging parameters
    bool log_to_file = this->declare_parameter<bool>("planning.log_to_file", false);
    std::string log_file = this->declare_parameter<std::string>("planning.log_file", std::string());
    std::string nullspace_log_file = this->declare_parameter<std::string>("planning.nullspace_log_file", std::string());
    std::string log_level = this->declare_parameter<std::string>("planning.log_level", "info");
    // apply node-level log level
    int sev = RCUTILS_LOG_SEVERITY_INFO; std::string ll = log_level; std::transform(ll.begin(), ll.end(), ll.begin(), ::tolower);
    if (ll=="debug") sev = RCUTILS_LOG_SEVERITY_DEBUG; else if (ll=="warn"||ll=="warning") sev=RCUTILS_LOG_SEVERITY_WARN; else if (ll=="error") sev=RCUTILS_LOG_SEVERITY_ERROR; else if (ll=="fatal") sev=RCUTILS_LOG_SEVERITY_FATAL;
    (void)rcutils_logging_set_logger_level(this->get_logger().get_name(), sev);

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

    // Load URDFs and build Pinocchio models for left and right arms
    if (urdf_path_left.empty()) {
      RCLCPP_ERROR(this->get_logger(), "planning.urdf_path_left param required");
      throw std::runtime_error("urdf_path_left missing");
    }
    if (urdf_path_right.empty()) {
      RCLCPP_ERROR(this->get_logger(), "planning.urdf_path_right param required");
      throw std::runtime_error("urdf_path_right missing");
    }

    std::string urdf_left = readFileToString(urdf_path_left);
    std::string urdf_right = readFileToString(urdf_path_right);
    if (urdf_left.empty()) {
      RCLCPP_ERROR(this->get_logger(), "Failed to read left URDF: %s", urdf_path_left.c_str());
      throw std::runtime_error("Left URDF read failed");
    }
    if (urdf_right.empty()) {
      RCLCPP_ERROR(this->get_logger(), "Failed to read right URDF: %s", urdf_path_right.c_str());
      throw std::runtime_error("Right URDF read failed");
    }
    // Build IkSolver instances from URDF XML (encapsulates Pinocchio inside IkSolver)
    try {
      solver_left_ = std::make_shared<IkSolver>(urdf_left, tip_link_left);
      solver_left_->setParams(p);
      solver_right_ = std::make_shared<IkSolver>(urdf_right, tip_link_right);
      solver_right_->setParams(p);
      tip_frame_id_left_ = static_cast<unsigned int>(solver_left_->getTipFrameId());
      tip_frame_id_right_ = static_cast<unsigned int>(solver_right_->getTipFrameId());
    } catch (const std::exception &e) {
      RCLCPP_ERROR(this->get_logger(), "Failed to build IkSolver from URDF: %s", e.what());
      throw;
    }

    // cache q indices for joints we may want to target for null-space penalty (left)
    j4_q_index_ = j5_q_index_ = j7_q_index_ = j3_q_index_ = -1;
    // IkSolver currently does not expose joint name -> joint id/index helpers.
    // Keep cached indices at -1. If needed, add an API to IkSolver to query joint indices by name.

    RCLCPP_INFO(this->get_logger(), "Pinocchio models loaded: nq_left=%d nv_left=%d nq_right=%d nv_right=%d tip_left='%s' tip_right='%s' tip_id_left=%u tip_id_right=%u",
                static_cast<int>(solver_left_->getNq()), static_cast<int>(solver_left_->getNv()), static_cast<int>(solver_right_->getNq()), static_cast<int>(solver_right_->getNv()), tip_link_left.c_str(), tip_link_right.c_str(), tip_frame_id_left_, tip_frame_id_right_);

    // Also write a textual init line into the log file (if enabled)
    {
      std::ostringstream oss;
      oss << "Pinocchio models loaded: nq_left=" << solver_left_->getNq() << " nv_left=" << solver_left_->getNv()
          << " nq_right=" << solver_right_->getNq() << " nv_right=" << solver_right_->getNv()
          << " tip_left='" << tip_link_left << "' tip_right='" << tip_link_right << "'"
          << " tip_id_left=" << tip_frame_id_left_ << " tip_id_right=" << tip_frame_id_right_;
      appendLog(oss.str());
    }

    // open iteration log file and register iter callback if requested (use left model dims for header)
    if (p.log_to_file && !p.log_file.empty()) {
      try {
        auto fp = std::make_shared<std::ofstream>(p.log_file, std::ios::app);
        if (!fp->good()) {
          RCLCPP_WARN(this->get_logger(), "Failed to open ik log file: %s", p.log_file.c_str());
        } else {
          // write CSV header if file empty
          fp->seekp(0, std::ios::end);
          if (fp->tellp() == 0) {
            (*fp) << "ts_ms,iter,err";
            for (int i = 0; i < solver_left_->getNq(); ++i) (*fp) << ",q_left" << i;
            for (int i = 0; i < solver_right_->getNq(); ++i) (*fp) << ",q_right" << i;
            (*fp) << "\n";
            fp->flush();
            // also write a short textual header to the log file for init/summary messages
            (*fp) << "# ik_solver textual log (init & solver summaries)\n";
            fp->flush();
          }
          iter_log_fp_ = fp;
          // initialize cached q vectors
          last_logged_q_left_.resize(solver_left_->getNq()); last_logged_q_left_.setZero();
          last_logged_q_right_.resize(solver_right_->getNq()); last_logged_q_right_.setZero();
          // left callback: update left cache and write combined row (using latest right cache)
          auto write_cb_left = [this, fp](int it, const Eigen::VectorXd &q, double err) {
            if (!fp || !fp->good()) return;
            auto now = std::chrono::system_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
            std::lock_guard<std::mutex> lk(mutex_);
            // update cached left q
            if (q.size() == last_logged_q_left_.size()) last_logged_q_left_ = q;
            (*fp) << ms << "," << it << "," << std::setprecision(9) << err;
            for (int i = 0; i < last_logged_q_left_.size(); ++i) (*fp) << "," << std::setprecision(9) << last_logged_q_left_[i];
            for (int i = 0; i < last_logged_q_right_.size(); ++i) (*fp) << "," << std::setprecision(9) << last_logged_q_right_[i];
            (*fp) << "\n";
            fp->flush();
          };
          // right callback: update right cache and write combined row (using latest left cache)
          auto write_cb_right = [this, fp](int it, const Eigen::VectorXd &q, double err) {
            if (!fp || !fp->good()) return;
            auto now = std::chrono::system_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
            std::lock_guard<std::mutex> lk(mutex_);
            if (q.size() == last_logged_q_right_.size()) last_logged_q_right_ = q;
            (*fp) << ms << "," << it << "," << std::setprecision(9) << err;
            for (int i = 0; i < last_logged_q_left_.size(); ++i) (*fp) << "," << std::setprecision(9) << last_logged_q_left_[i];
            for (int i = 0; i < last_logged_q_right_.size(); ++i) (*fp) << "," << std::setprecision(9) << last_logged_q_right_[i];
            (*fp) << "\n";
            fp->flush();
          };
          solver_left_->setIterCallback(write_cb_left);
          solver_right_->setIterCallback(write_cb_right);
        }
      } catch (const std::exception &e) {
        RCLCPP_WARN(this->get_logger(), "Exception opening ik log file '%s': %s", p.log_file.c_str(), e.what());
      }
    }

    if (!p.joint_limits_min.empty() && p.joint_limits_min.size() == solver_left_->getNq() && !p.joint_limits_max.empty() && p.joint_limits_max.size() == solver_left_->getNq()) {
      int nqL = solver_left_->getNq();
      VectorXd lo(nqL), hi(nqL);
      for (int i=0;i<nqL;++i) { lo[i]=p.joint_limits_min[i]; hi[i]=p.joint_limits_max[i]; }
      solver_left_->setJointLimits(lo, hi);
    }
    if (!p.joint_limits_min.empty() && p.joint_limits_min.size() == solver_right_->getNq() && !p.joint_limits_max.empty() && p.joint_limits_max.size() == solver_right_->getNq()) {
      int nqR = solver_right_->getNq();
      VectorXd lo(nqR), hi(nqR);
      for (int i=0;i<nqR;++i) { lo[i]=p.joint_limits_min[i]; hi[i]=p.joint_limits_max[i]; }
      solver_right_->setJointLimits(lo, hi);
    }

    // create subs/pubs/timer as before, plus additional subscriptions to match ik_node
    sub_js_ = this->create_subscription<sensor_msgs::msg::JointState>("joint_states", 10, std::bind(&IkSolverNode::onJointState, this, std::placeholders::_1));
    sub_target_ = this->create_subscription<geometry_msgs::msg::PoseStamped>("target_pose", 10, std::bind(&IkSolverNode::onTargetPose, this, std::placeholders::_1));
    // per-arm targets
    sub_target_left_ = this->create_subscription<geometry_msgs::msg::PoseStamped>("target_pose_left", 10, std::bind(&IkSolverNode::onTargetLeft, this, std::placeholders::_1));
    sub_target_right_ = this->create_subscription<geometry_msgs::msg::PoseStamped>("target_pose_right", 10, std::bind(&IkSolverNode::onTargetRight, this, std::placeholders::_1));
    sub_target_end_ = this->create_subscription<geometry_msgs::msg::PoseStamped>("target_end_pose", 10, std::bind(&IkSolverNode::onTargetEndPose, this, std::placeholders::_1));
    sub_delta_ = this->create_subscription<geometry_msgs::msg::Twist>("ik_delta", 10, std::bind(&IkSolverNode::onIkDelta, this, std::placeholders::_1));
    // joint-delta topic: applies small joint deltas to each arm and updates respective FK targets
    sub_delta_joint_ = this->create_subscription<sensor_msgs::msg::JointState>("delta_joint", 10, std::bind(&IkSolverNode::onDeltaJoint, this, std::placeholders::_1));
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

  // Append a textual log line to the configured log file (if opened). Used for init and solver summary logs.
  void appendLog(const std::string &line) {
    if (iter_log_fp_ && iter_log_fp_->good()) {
      (*iter_log_fp_) << line << std::endl;
      iter_log_fp_->flush();
    }
  }

  void onJointState(const sensor_msgs::msg::JointState::SharedPtr msg) {
    std::lock_guard<std::mutex> lk(mutex_);
    last_js_ = *msg;

    // If we don't yet have a target pose, initialize it from current FK computed from incoming joint_states
    if (!target_received_ && !last_js_.position.empty()) {
      // build left and right q from incoming joint_states (first left then right)
      VectorXd ql = VectorXd::Zero(solver_left_->getNq());
      VectorXd qr = VectorXd::Zero(solver_right_->getNq());
      for (size_t i = 0; i < last_js_.position.size() && i < (size_t)ql.size(); ++i) ql[i] = last_js_.position[i];
      for (size_t i = 0; i + ql.size() < last_js_.position.size() && (i < (size_t)qr.size()); ++i) qr[i] = last_js_.position[i + ql.size()];
      // compute FK for left
      try {
        auto poseL = solver_left_->forwardKinematicsSE3(ql);
        Eigen::Quaterniond eqL(poseL.rotation());
        geometry_msgs::msg::PoseStamped psl;
        psl.header.stamp = this->get_clock()->now(); psl.header.frame_id = "";
        psl.pose.position.x = poseL.translation()[0]; psl.pose.position.y = poseL.translation()[1]; psl.pose.position.z = poseL.translation()[2];
        psl.pose.orientation.x = eqL.x(); psl.pose.orientation.y = eqL.y(); psl.pose.orientation.z = eqL.z(); psl.pose.orientation.w = eqL.w();
        last_target_left_ = psl;
      } catch (const std::exception &e) {
        RCLCPP_WARN(this->get_logger(), "Failed to compute left FK to initialize target pose: %s", e.what());
      }
      // compute FK for right
      try {
        auto poseR = solver_right_->forwardKinematicsSE3(qr);
        Eigen::Quaterniond eqR(poseR.rotation());
        geometry_msgs::msg::PoseStamped psr;
        psr.header.stamp = this->get_clock()->now(); psr.header.frame_id = "";
        psr.pose.position.x = poseR.translation()[0]; psr.pose.position.y = poseR.translation()[1]; psr.pose.position.z = poseR.translation()[2];
        psr.pose.orientation.x = eqR.x(); psr.pose.orientation.y = eqR.y(); psr.pose.orientation.z = eqR.z(); psr.pose.orientation.w = eqR.w();
        last_target_right_ = psr;
      } catch (const std::exception &e) {
        RCLCPP_WARN(this->get_logger(), "Failed to compute right FK to initialize target pose: %s", e.what());
      }

      // Log received joint_state (names + positions) into the log file to help debug mapping/order issues
      if (iter_log_fp_ && iter_log_fp_->good()) {
        std::ostringstream ossjs;
        ossjs << "JointState_received: names=[";
        for (size_t i=0;i<last_js_.name.size();++i) { if (i) ossjs << ","; ossjs << last_js_.name[i]; }
        ossjs << "] positions=[";
        for (size_t i=0;i<last_js_.position.size();++i) { if (i) ossjs << ","; ossjs << last_js_.position[i]; }
        ossjs << "]";
        appendLog(ossjs.str());

        // also log the mapping assumption (left then right) and sizes
        std::ostringstream ossmap; ossmap << "Assumed mapping: left_nq=" << solver_left_->getNq() << " right_nq=" << solver_right_->getNq() << " incoming_count=" << last_js_.position.size();
        appendLog(ossmap.str());
      }

      // do NOT set shared target_received_ here; instead leave per-arm last_target_left_/right_ so spinOnce uses per-arm targets
      // but keep last_target_ for backward compatibility (set to left)
      last_target_ = last_target_left_;
      // Use per-arm initialized targets but keep shared-flag true (legacy behavior)
      target_received_ = true;
      target_left_received_ = true;
      target_right_received_ = true;
      explicit_target_pending_ = true;
    }
  }

  void onTargetPose(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    std::lock_guard<std::mutex> lk(mutex_);
    last_target_ = *msg;
    target_received_ = true;
    prefer_end_pose_target_ = false;
    explicit_target_pending_ = true;
  }

  // per-arm target handlers
  void onTargetLeft(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    std::lock_guard<std::mutex> lk(mutex_);
    last_target_left_ = *msg;
    target_left_received_ = true;
    // treat a per-arm explicit target as an explicit request: set shared flag and pending
    target_received_ = true;
    explicit_target_pending_ = true;
  }

  void onTargetRight(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    std::lock_guard<std::mutex> lk(mutex_);
    last_target_right_ = *msg;
    target_right_received_ = true;
    target_received_ = true;
    explicit_target_pending_ = true;
  }

  void onTargetEndPose(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    std::lock_guard<std::mutex> lk(mutex_);
    last_target_ = *msg;
    target_received_ = true;
    prefer_end_pose_target_ = true;
    RCLCPP_DEBUG(this->get_logger(), "Received target_end_pose; will ignore ik_delta until overridden");
    explicit_target_pending_ = true;
  }

  // delta joint message: applies joint deltas to current joint_states and updates per-arm FK targets
  void onDeltaJoint(const sensor_msgs::msg::JointState::SharedPtr msg) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (last_js_.position.empty()) return;
    // build current q vectors
    VectorXd q_left = VectorXd::Zero(solver_left_->getNq());
    VectorXd q_right = VectorXd::Zero(solver_right_->getNq());
    for (size_t i=0;i<last_js_.position.size() && i<q_left.size(); ++i) q_left[i] = last_js_.position[i];
    for (size_t i=0;i+q_left.size() < last_js_.position.size() && (i < q_right.size()); ++i) q_right[i] = last_js_.position[i + q_left.size()];
    // apply deltas if provided in msg.position (supports length == left, == right, or == total)
    if (!msg->position.empty()) {
      if (msg->position.size() == static_cast<size_t>(solver_left_->getNq())) {
        for (size_t i=0;i<q_left.size(); ++i) q_left[i] += msg->position[i];
      } else if (msg->position.size() == static_cast<size_t>(solver_right_->getNq())) {
        for (size_t i=0;i<q_right.size(); ++i) q_right[i] += msg->position[i];
      } else if (msg->position.size() == q_left.size() + q_right.size()) {
        for (size_t i=0;i<q_left.size(); ++i) q_left[i] += msg->position[i];
        for (size_t i=0;i<q_right.size(); ++i) q_right[i] += msg->position[q_left.size() + i];
      }
    }
    // compute FK for each and update last_target_left_/right_
    try {
      Eigen::VectorXd curl = solver_left_->forwardKinematics(q_left);
      Eigen::VectorXd curr = solver_right_->forwardKinematics(q_right);
      geometry_msgs::msg::PoseStamped pl, pr;
      pl.header.stamp = this->now(); pr.header.stamp = this->now();
      double qxL = curl.size() > 3 ? curl(3) : 0.0;
      double qyL = curl.size() > 4 ? curl(4) : 0.0;
      double qzL = curl.size() > 5 ? curl(5) : 0.0;
      double qw2L = 1.0 - (qxL*qxL + qyL*qyL + qzL*qzL);
      double qwL = (qw2L > 0.0) ? std::sqrt(qw2L) : 0.0;
      Eigen::Quaterniond qrotL(qwL, qxL, qyL, qzL);
      qrotL.normalize();
      pl.pose.orientation.w = qrotL.w(); pl.pose.orientation.x = qrotL.x(); pl.pose.orientation.y = qrotL.y(); pl.pose.orientation.z = qrotL.z();
      pl.pose.position.x = curl.size() > 0 ? curl(0) : 0.0; pl.pose.position.y = curl.size() > 1 ? curl(1) : 0.0; pl.pose.position.z = curl.size() > 2 ? curl(2) : 0.0;
      double qxR = curr.size() > 3 ? curr(3) : 0.0;
      double qyR = curr.size() > 4 ? curr(4) : 0.0;
      double qzR = curr.size() > 5 ? curr(5) : 0.0;
      double qw2R = 1.0 - (qxR*qxR + qyR*qyR + qzR*qzR);
      double qwR = (qw2R > 0.0) ? std::sqrt(qw2R) : 0.0;
      Eigen::Quaterniond qrotR(qwR, qxR, qyR, qzR);
      qrotR.normalize();
      pr.pose.orientation.w = qrotR.w(); pr.pose.orientation.x = qrotR.x(); pr.pose.orientation.y = qrotR.y(); pr.pose.orientation.z = qrotR.z();
      pr.pose.position.x = curr.size() > 0 ? curr(0) : 0.0; pr.pose.position.y = curr.size() > 1 ? curr(1) : 0.0; pr.pose.position.z = curr.size() > 2 ? curr(2) : 0.0;
      last_target_left_ = pl; last_target_right_ = pr;
      target_left_received_ = target_right_received_ = true;
    } catch (const std::exception &e) {
      RCLCPP_WARN(this->get_logger(), "delta_joint FK computation failed: %s", e.what());
    }
  }

  void onIkDelta(const geometry_msgs::msg::Twist::SharedPtr msg) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (!target_received_) return;
    if (prefer_end_pose_target_) {
      RCLCPP_DEBUG(this->get_logger(), "Ignoring ik_delta because target_end_pose is active");
      return;
    }

    // compute linear delta (world frame) and rotation delta quaternion
    double dx = msg->linear.x * ik_delta_linear_scale_;
    double dy = msg->linear.y * ik_delta_linear_scale_;
    double dz = msg->linear.z * ik_delta_linear_scale_;

    Eigen::Vector3d ang;
    ang.x() = msg->angular.x * ik_delta_angular_scale_;
    ang.y() = msg->angular.y * ik_delta_angular_scale_;
    ang.z() = msg->angular.z * ik_delta_angular_scale_;
    double angle = ang.norm();
    Eigen::Quaterniond q_delta = Eigen::Quaterniond::Identity();
    if (angle > 1e-12) {
      Eigen::Vector3d axis = ang / angle;
      Eigen::AngleAxisd aa(angle, axis);
      q_delta = Eigen::Quaterniond(aa);
    }

    auto apply_delta_to_pose = [&](geometry_msgs::msg::PoseStamped &ps) {
      ps.pose.position.x += dx;
      ps.pose.position.y += dy;
      ps.pose.position.z += dz;
      if (angle > 1e-12) {
        Eigen::Quaterniond q_old(ps.pose.orientation.w,
                                 ps.pose.orientation.x,
                                 ps.pose.orientation.y,
                                 ps.pose.orientation.z);
        Eigen::Quaterniond q_new = q_delta * q_old;
        q_new.normalize();
        ps.pose.orientation.x = q_new.x();
        ps.pose.orientation.y = q_new.y();
        ps.pose.orientation.z = q_new.z();
        ps.pose.orientation.w = q_new.w();
      }
    };

    bool applied = false;
    // If per-arm explicit targets exist, apply to them (preserve explicitness)
    if (target_left_received_) {
      apply_delta_to_pose(last_target_left_);
      applied = true;
    }
    if (target_right_received_) {
      apply_delta_to_pose(last_target_right_);
      applied = true;
    }
    // If no per-arm explicit targets, apply to shared target
    if (!applied) {
      apply_delta_to_pose(last_target_);
    }

    // mark explicit pending so updated pose will be published promptly
    explicit_target_pending_ = true;

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

    // build left and right init q vectors from incoming joint_states (first left then right)
    VectorXd q_init_left = VectorXd::Zero(solver_left_->getNq());
    VectorXd q_init_right = VectorXd::Zero(solver_right_->getNq());
    for (size_t i=0;i<js.position.size() && i<static_cast<size_t>(q_init_left.size()); ++i) q_init_left[i] = js.position[i];
    for (size_t i=0;i+q_init_left.size() < js.position.size() && (i < static_cast<size_t>(q_init_right.size())); ++i) q_init_right[i] = js.position[i + q_init_left.size()];

    if (!target_received_) {
      // No external target: compute FK via Pinocchio for each arm and use those poses as per-arm targets
      geometry_msgs::msg::PoseStamped cur_msg_left, cur_msg_right;
      cur_msg_left.header.stamp = this->now();
      cur_msg_right.header.stamp = this->now();
      // Log q_init vectors and tip frame ids to help debug zero targets
      {
        std::ostringstream ossl; ossl << "q_init_left=[";
        for (int ii=0; ii < static_cast<int>(q_init_left.size()); ++ii) { if (ii) ossl << ","; ossl << q_init_left[ii]; }
        ossl << "]";
        std::ostringstream ossr; ossr << "q_init_right=[";
        for (int ii=0; ii < static_cast<int>(q_init_right.size()); ++ii) { if (ii) ossr << ","; ossr << q_init_right[ii]; }
        ossr << "]";
        RCLCPP_INFO(this->get_logger(), "spinOnce: %s", ossl.str().c_str());
        RCLCPP_INFO(this->get_logger(), "spinOnce: %s", ossr.str().c_str());
        RCLCPP_INFO(this->get_logger(), "spinOnce: tip_frame_id_left=%u tip_frame_id_right=%u model_left_nq=%d model_right_nq=%d",
                    tip_frame_id_left_, tip_frame_id_right_, static_cast<int>(solver_left_->getNq()), static_cast<int>(solver_right_->getNq()));
        // write same init info to textual log file
        appendLog(ossl.str());
        appendLog(ossr.str());
        {
          std::ostringstream ossf; ossf << "tip_frame_id_left=" << tip_frame_id_left_ << " tip_frame_id_right=" << tip_frame_id_right_
                                        << " model_left_nq=" << solver_left_->getNq() << " model_right_nq=" << solver_right_->getNq();
          appendLog(ossf.str());
        }
        // additionally log incoming joint names (mapping info)
        if (iter_log_fp_ && iter_log_fp_->good()) {
          std::ostringstream ossn; ossn << "incoming_joint_names_count=" << js.name.size() << " names=[";
          for (size_t i=0;i<js.name.size();++i) { if (i) ossn << ","; ossn << js.name[i]; }
          ossn << "]";
          appendLog(ossn.str());
        }
      }
      try {
        const auto poseL = solver_left_->forwardKinematicsSE3(q_init_left);
        Eigen::Quaterniond qrotL(poseL.rotation()); qrotL.normalize();
        cur_msg_left.pose.orientation.w = qrotL.w(); cur_msg_left.pose.orientation.x = qrotL.x(); cur_msg_left.pose.orientation.y = qrotL.y(); cur_msg_left.pose.orientation.z = qrotL.z();
        cur_msg_left.pose.position.x = poseL.translation()[0]; cur_msg_left.pose.position.y = poseL.translation()[1]; cur_msg_left.pose.position.z = poseL.translation()[2];
        RCLCPP_INFO(this->get_logger(), "Left FK pose: trans=(%.6f,%.6f,%.6f) quat=(%.6f,%.6f,%.6f,%.6f)",
                    poseL.translation()[0], poseL.translation()[1], poseL.translation()[2], qrotL.w(), qrotL.x(), qrotL.y(), qrotL.z());
      } catch (const std::exception &e) {
        RCLCPP_WARN(this->get_logger(), "Left FK failed: %s", e.what());
      }
      try {
        const auto poseR = solver_right_->forwardKinematicsSE3(q_init_right);
        Eigen::Quaterniond qrotR(poseR.rotation()); qrotR.normalize();
        cur_msg_right.pose.orientation.w = qrotR.w(); cur_msg_right.pose.orientation.x = qrotR.x(); cur_msg_right.pose.orientation.y = qrotR.y(); cur_msg_right.pose.orientation.z = qrotR.z();
        cur_msg_right.pose.position.x = poseR.translation()[0]; cur_msg_right.pose.position.y = poseR.translation()[1]; cur_msg_right.pose.position.z = poseR.translation()[2];
        RCLCPP_INFO(this->get_logger(), "Right FK pose: trans=(%.6f,%.6f,%.6f) quat=(%.6f,%.6f,%.6f,%.6f)",
                    poseR.translation()[0], poseR.translation()[1], poseR.translation()[2], qrotR.w(), qrotR.x(), qrotR.y(), qrotR.z());
      } catch (const std::exception &e) {
        RCLCPP_WARN(this->get_logger(), "Right FK failed: %s", e.what());
      }
      // only overwrite per-arm last_target if the user hasn't provided an explicit per-arm target
      std::lock_guard<std::mutex> lk(mutex_);
      if (!target_left_received_) last_target_left_ = cur_msg_left;
      if (!target_right_received_) last_target_right_ = cur_msg_right;
      // keep last_target_ for compatibility (set to left)
      last_target_ = last_target_left_;
    }

    // build per-arm SE3 targets: prefer explicit per-arm targets, else shared target, else per-arm FK
    IkSolver::SE3 target_se3_left = IkSolver::SE3::Identity();
    IkSolver::SE3 target_se3_right = IkSolver::SE3::Identity();
    // Left precedence: explicit left target > shared target > per-arm FK-initialized left target
    if (target_left_received_) {
      Eigen::Quaterniond q(last_target_left_.pose.orientation.w,
                           last_target_left_.pose.orientation.x,
                           last_target_left_.pose.orientation.y,
                           last_target_left_.pose.orientation.z);
      q.normalize();
      target_se3_left.linear() = q.toRotationMatrix();
      target_se3_left.translation() = Eigen::Vector3d(last_target_left_.pose.position.x,
                                                      last_target_left_.pose.position.y,
                                                      last_target_left_.pose.position.z);
    } else if (target_received_) {
      Eigen::Quaterniond q(target.pose.orientation.w,
                           target.pose.orientation.x,
                           target.pose.orientation.y,
                           target.pose.orientation.z);
      q.normalize();
      target_se3_left.linear() = q.toRotationMatrix();
      target_se3_left.translation() = Eigen::Vector3d(target.pose.position.x,
                                                      target.pose.position.y,
                                                      target.pose.position.z);
    } else {
      Eigen::Quaterniond q(last_target_left_.pose.orientation.w,
                           last_target_left_.pose.orientation.x,
                           last_target_left_.pose.orientation.y,
                           last_target_left_.pose.orientation.z);
      q.normalize();
      target_se3_left.linear() = q.toRotationMatrix();
      target_se3_left.translation() = Eigen::Vector3d(last_target_left_.pose.position.x,
                                                      last_target_left_.pose.position.y,
                                                      last_target_left_.pose.position.z);
    }

    // Right precedence: explicit right target > shared target > per-arm FK-initialized right target
    if (target_right_received_) {
      Eigen::Quaterniond q(last_target_right_.pose.orientation.w,
                           last_target_right_.pose.orientation.x,
                           last_target_right_.pose.orientation.y,
                           last_target_right_.pose.orientation.z);
      q.normalize();
      target_se3_right.linear() = q.toRotationMatrix();
      target_se3_right.translation() = Eigen::Vector3d(last_target_right_.pose.position.x,
                                                       last_target_right_.pose.position.y,
                                                       last_target_right_.pose.position.z);
    } else if (target_received_) {
      Eigen::Quaterniond q(target.pose.orientation.w,
                           target.pose.orientation.x,
                           target.pose.orientation.y,
                           target.pose.orientation.z);
      q.normalize();
      target_se3_right.linear() = q.toRotationMatrix();
      target_se3_right.translation() = Eigen::Vector3d(target.pose.position.x,
                                                       target.pose.position.y,
                                                       target.pose.position.z);
    } else {
      Eigen::Quaterniond q(last_target_right_.pose.orientation.w,
                           last_target_right_.pose.orientation.x,
                           last_target_right_.pose.orientation.y,
                           last_target_right_.pose.orientation.z);
      q.normalize();
      target_se3_right.linear() = q.toRotationMatrix();
      target_se3_right.translation() = Eigen::Vector3d(last_target_right_.pose.position.x,
                                                       last_target_right_.pose.position.y,
                                                       last_target_right_.pose.position.z);
    }

    // Log target pose for debugging/trace
    RCLCPP_INFO(this->get_logger(), "Target left pos=(%.6f, %.6f, %.6f) quat=(%.6f, %.6f, %.6f, %.6f)",
                last_target_left_.pose.position.x, last_target_left_.pose.position.y, last_target_left_.pose.position.z,
                last_target_left_.pose.orientation.w, last_target_left_.pose.orientation.x, last_target_left_.pose.orientation.y, last_target_left_.pose.orientation.z);
    RCLCPP_INFO(this->get_logger(), "Target right pos=(%.6f, %.6f, %.6f) quat=(%.6f, %.6f, %.6f, %.6f)",
                last_target_right_.pose.position.x, last_target_right_.pose.position.y, last_target_right_.pose.position.z,
                last_target_right_.pose.orientation.w, last_target_right_.pose.orientation.x, last_target_right_.pose.orientation.y, last_target_right_.pose.orientation.z);

    // also append human-readable target lines into log file
    if (iter_log_fp_ && iter_log_fp_->good()) {
      std::ostringstream osstl; osstl << "TargetLeft: pos=(" << target_se3_left.translation().transpose() << ")";
      appendLog(osstl.str());
      std::ostringstream osstr; osstr << "TargetRight: pos=(" << target_se3_right.translation().transpose() << ")";
      appendLog(osstr.str());
    }

    auto t_spin_start = std::chrono::steady_clock::now();
    auto res_left = solver_left_->solve(target_se3_left, q_init_left, 100);
    auto res_right = solver_right_->solve(target_se3_right, q_init_right, 100);
    auto t_spin_end = std::chrono::steady_clock::now();
    double spin_elapsed_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_spin_end - t_spin_start).count();

    const char *status_str_left = "失败";
    if (res_left.status == 1) status_str_left = "精确成功";
    else if (res_left.status == 2) status_str_left = "放宽成功";
    const char *status_str_right = "失败";
    if (res_right.status == 1) status_str_right = "精确成功";
    else if (res_right.status == 2) status_str_right = "放宽成功";

    RCLCPP_INFO(this->get_logger(), "ik_solver_node: left elapsed: %.3f ms, final_err: %.6g, iterations: %d, status: %s",
                res_left.elapsed_ms, res_left.final_error, res_left.iterations, status_str_left);
    RCLCPP_INFO(this->get_logger(), "ik_solver_node: right elapsed: %.3f ms, final_err: %.6g, iterations: %d, status: %s",
                res_right.elapsed_ms, res_right.final_error, res_right.iterations, status_str_right);
    RCLCPP_INFO(this->get_logger(), "spinOnce elapsed: %.3f ms [left: %s right: %s]", spin_elapsed_ms, status_str_left, status_str_right);
    // write solver summary to textual log
    {
      std::ostringstream oss; oss << "SolverSummary: left_elapsed_ms=" << res_left.elapsed_ms << " left_err=" << res_left.final_error << " left_iters=" << res_left.iterations << " left_status=" << status_str_left
                                  << " | right_elapsed_ms=" << res_right.elapsed_ms << " right_err=" << res_right.final_error << " right_iters=" << res_right.iterations << " right_status=" << status_str_right
                                  << " | spin_ms=" << spin_elapsed_ms;
      appendLog(oss.str());
    }

    // compute FK at solver results and log error to target for each arm (translation norm and rotation angle)
    if (iter_log_fp_ && iter_log_fp_->good()) {
      try {
        if (res_left.q.size() == solver_left_->getNq()) {
          auto pose_resL = solver_left_->forwardKinematicsSE3(res_left.q);
          Eigen::Vector3d p_res = pose_resL.translation();
          Eigen::Matrix3d R_res = pose_resL.rotation();
          Eigen::Vector3d p_tgt = target_se3_left.translation();
          Eigen::Matrix3d R_tgt = target_se3_left.rotation();
          double trans_err = (p_res - p_tgt).norm();
          double tr = (R_res * R_tgt.transpose()).trace();
          double cosang = (tr - 1.0) / 2.0; if (cosang > 1.0) cosang = 1.0; if (cosang < -1.0) cosang = -1.0;
          double rot_err = std::acos(cosang);
          std::ostringstream oss; oss << "ResultFK left: res_pos=(" << p_res.transpose() << ") tgt_pos=(" << p_tgt.transpose() << ") trans_err=" << trans_err << " rot_err=" << rot_err;
          appendLog(oss.str());
        } else {
          appendLog("ResultFK left: no valid q to compute FK");
        }
      } catch (const std::exception &e) { std::ostringstream oss; oss << "ResultFK left exception: " << e.what(); appendLog(oss.str()); }

      try {
        if (res_right.q.size() == solver_right_->getNq()) {
          auto pose_resR = solver_right_->forwardKinematicsSE3(res_right.q);
          Eigen::Vector3d p_res = pose_resR.translation();
          Eigen::Matrix3d R_res = pose_resR.rotation();
          Eigen::Vector3d p_tgt = target_se3_right.translation();
          Eigen::Matrix3d R_tgt = target_se3_right.rotation();
          double trans_err = (p_res - p_tgt).norm();
          double tr = (R_res * R_tgt.transpose()).trace();
          double cosang = (tr - 1.0) / 2.0; if (cosang > 1.0) cosang = 1.0; if (cosang < -1.0) cosang = -1.0;
          double rot_err = std::acos(cosang);
          std::ostringstream oss; oss << "ResultFK right: res_pos=(" << p_res.transpose() << ") tgt_pos=(" << p_tgt.transpose() << ") trans_err=" << trans_err << " rot_err=" << rot_err;
          appendLog(oss.str());
        } else {
          appendLog("ResultFK right: no valid q to compute FK");
        }
      } catch (const std::exception &e) { std::ostringstream oss; oss << "ResultFK right exception: " << e.what(); appendLog(oss.str()); }
    }

    // publish combined joint_command with left then right positions only when there is an explicit target
    bool have_explicit_target = target_received_ || target_left_received_ || target_right_received_ || prefer_end_pose_target_;
    // always prepare output positions but apply deadband to decide publish
    if (res_left.success || res_right.success) {
      sensor_msgs::msg::JointState out;
      out.header.stamp = this->now();
      out.name = js.name; // expect names in same order: left then right
      size_t total_q = static_cast<size_t>(solver_left_->getNq() + solver_right_->getNq());
      out.position.resize(total_q);
      // fill left
      if (res_left.success) {
        for (int i=0;i<res_left.q.size() && i<(int)solver_left_->getNq(); ++i) out.position[i] = res_left.q[i];
      } else {
        for (int i=0;i<solver_left_->getNq(); ++i) out.position[i] = q_init_left[i];
      }
      // fill right
      if (res_right.success) {
        for (int i=0;i<res_right.q.size() && i<(int)solver_right_->getNq(); ++i) out.position[solver_left_->getNq() + i] = res_right.q[i];
      } else {
        for (int i=0;i<solver_right_->getNq(); ++i) out.position[solver_left_->getNq() + i] = q_init_right[i];
      }
      // decide publish: explicit targets are published once (or when solution differs from last published);
      // otherwise compare desired joints to current measured joints
      bool should_publish = false;
      double max_diff_js = 0.0;
      double max_diff_lastpub = 0.0;
      size_t ncompare = std::min(out.position.size(), js.position.size());
      if (ncompare == 0) {
        // no reliable current joint info -> be conservative and only publish if explicit target
        should_publish = have_explicit_target && (explicit_target_pending_ || last_published_pos_.empty());
      } else {
        // compute max diff to current joint_states
        for (size_t i = 0; i < ncompare; ++i) {
          double d = std::abs(out.position[i] - js.position[i]);
          if (d > max_diff_js) max_diff_js = d;
        }
        // compute max diff to last_published_pos_ if available
        if (!last_published_pos_.empty() && last_published_pos_.size() == out.position.size()) {
          for (size_t i = 0; i < out.position.size(); ++i) {
            double d = std::abs(out.position[i] - last_published_pos_[i]);
            if (d > max_diff_lastpub) max_diff_lastpub = d;
          }
        }
        // If explicit target: publish if pending or solution differs from last published
        if (have_explicit_target) {
          if (explicit_target_pending_ || last_published_pos_.empty() || max_diff_lastpub > publish_deadband_) {
            should_publish = true;
          } else {
            should_publish = false;
          }
          RCLCPP_DEBUG(this->get_logger(), "Explicit target: max_diff_js=%.9g max_diff_lastpub=%.9g publish_deadband=%.9g pending=%d should_publish=%d",
                       max_diff_js, max_diff_lastpub, publish_deadband_, explicit_target_pending_ ? 1 : 0, should_publish ? 1 : 0);
        } else {
          // If we don't have an explicit target but both solvers converged exactly (final_error == 0),
          // publish the solved joint_command only once (when it differs from last published),
          // to avoid repeated publishes due to small sensor/controller drift.
          if (res_left.success && res_right.success && res_left.final_error == 0.0 && res_right.final_error == 0.0) {
            // If current measured joints are already within deadband of desired, do NOT publish
            if (max_diff_js <= publish_deadband_) {
              should_publish = false;
            } else {
              // publish if we haven't published this solution before (or last_published empty)
              should_publish = last_published_pos_.empty() || (max_diff_lastpub > publish_deadband_);
            }
            RCLCPP_DEBUG(this->get_logger(), "Exact convergence: max_diff_js=%.9g max_diff_lastpub=%.9g publish_deadband=%.9g should_publish=%d",
                         max_diff_js, max_diff_lastpub, publish_deadband_, should_publish ? 1 : 0);
          } else if (!have_explicit_target) {
            // For non-exact solves, fall back to comparing desired joints to measured joints (previous behavior)
            should_publish = (max_diff_js > publish_deadband_);
            RCLCPP_DEBUG(this->get_logger(), "Non-exact solve: max_diff_js=%.9g publish_deadband=%.9g should_publish=%d",
                         max_diff_js, publish_deadband_, should_publish ? 1 : 0);
          }
        }
      }
      if (should_publish) {
        if (have_explicit_target) RCLCPP_INFO(this->get_logger(), "Publishing joint_command due to explicit target (max_diff_js=%.9g)", max_diff_js);
        else RCLCPP_INFO(this->get_logger(), "Publishing joint_command due to IK solution (max_diff_js=%.9g max_diff_lastpub=%.9g)", max_diff_js, max_diff_lastpub);
        pub_cmd_->publish(out);
        last_published_pos_.assign(out.position.begin(), out.position.end());
        // clear explicit pending flag after first publish for this explicit target
        if (have_explicit_target) explicit_target_pending_ = false;
      } else {
        RCLCPP_DEBUG(this->get_logger(), "Deadband: skipping joint_command publish (max delta to js=%.9g lastpub_diff=%.9g threshold=%.6g)", max_diff_js, max_diff_lastpub, publish_deadband_);
      }
    } else {
      RCLCPP_WARN(this->get_logger(), "Both IK solvers failed: left='%s' right='%s'", res_left.diagnostic.c_str(), res_right.diagnostic.c_str());
    }
  }

  std::mutex mutex_;
  sensor_msgs::msg::JointState last_js_;
  geometry_msgs::msg::PoseStamped last_target_;
  geometry_msgs::msg::PoseStamped last_target_left_;
  geometry_msgs::msg::PoseStamped last_target_right_;
  bool target_left_received_{false};
  bool target_right_received_{false};
  bool target_received_{false};
  bool prefer_end_pose_target_{false};
  // publishing deadband thresholds
  double ik_delta_linear_scale_{0.01};
  double ik_delta_angular_scale_{0.02};
  double publish_pose_pos_deadband_{1e-2};
  double publish_pose_ang_deadband_{1e-3};
  double publish_deadband_{1e-4};
  // last published joint positions
  std::vector<double> last_published_pos_;
  // when an explicit/shared target is received we mark it pending so we publish once
  bool explicit_target_pending_{false};

  std::shared_ptr<IkSolver> solver_left_;
  std::shared_ptr<IkSolver> solver_right_;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_js_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_target_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_target_left_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_target_right_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_target_end_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr sub_delta_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_delta_joint_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_cmd_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::string urdf_path_;
  unsigned int tip_frame_id_left_{0};
  unsigned int tip_frame_id_right_{0};
  int j4_q_index_{-1};
  int j5_q_index_{-1};
  int j7_q_index_{-1};
  int j3_q_index_{-1};

  // file logging support
  std::shared_ptr<std::ofstream> iter_log_fp_;
  // cached latest q values for writing combined left+right CSV rows
  Eigen::VectorXd last_logged_q_left_;
  Eigen::VectorXd last_logged_q_right_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<IkSolverNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
