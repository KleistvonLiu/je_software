#pragma once

#include <Eigen/Dense>
#include <functional>
#include <mutex>
#include <chrono>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <memory>

// Include Pinocchio forward declarations (typedefs) instead of forward-declaring class names
#include <pinocchio/multibody/fwd.hpp>

namespace ros2_ik_cpp {

// Public SE3 type (pose) exposed to callers. Implementations convert to pinocchio::SE3 internally.
using SE3 = Eigen::Isometry3d;

class IkSolver {
public:
  struct Params {
    int max_iters = 200;
    double eps = 1e-4;
    double eps_relaxed_3d = 5e-3;
    double eps_relaxed_6d = 1e-2;
    double ik_step_size = 1.0; // step scale (formerly dt)
    bool use_svd_damped = true;
    double ik_svd_damping = 1e-6;
    double ik_svd_damping_min = 1e-12;
    double ik_svd_damping_max = 1e6;
    double ik_svd_trunc_tol = 1e-6;
    double ik_svd_min_rel_reduction = 1e-8;
    double ik_svd_damping_reduce_factor = 0.1;
    double ik_svd_damping_increase_factor = 10.0;
    double nullspace_penalty_scale = 1e-4;
    double joint4_penalty_threshold = 0.05;
    double pos_weight = 1.0;
    double ang_weight = 1.0;
    double max_delta = 0.03;
    double max_delta_min = 1e-6;
    std::vector<double> joint_limits_min;
    std::vector<double> joint_limits_max;

    // numeric-J fallback
    bool use_numeric_jacobian = false;
    int numeric_fallback_after_rejects = 3;
    int numeric_fallback_duration = 10;

  // solver timeout (ms) used by caller
  int timeout_ms = 100;

    // logging / diagnostics (solver-local)
    bool log_to_file = false;
    std::string log_file;
    std::string nullspace_log_file;
  };

  struct Result {
    bool success = false;
    int status = 0; // 0 fail, 1 precise, 2 relaxed
    Eigen::VectorXd q;
    double final_error = 1e300;
    int iterations = 0;
    double elapsed_ms = 0.0;
    std::string diagnostic;
  };

  using IterCallback = std::function<void(int, const Eigen::VectorXd&, double)>;

  // Provide a class-local SE3 alias so callers can write IkSolver::SE3
  using SE3 = ::ros2_ik_cpp::SE3;

  // Construct from URDF XML string or URDF file path and tip frame name (builds internal Pinocchio model)
  explicit IkSolver(const std::string &urdf_xml, const std::string &tip_frame_name);
  // Construct from URDF and load solver params from YAML file path (if provided)
  IkSolver(const std::string &urdf_xml, const std::string &tip_frame_name, const std::string &yaml_path);
  // Construct entirely from YAML file path (urdf + tip + params)
  explicit IkSolver(const std::string &yaml_path);
  ~IkSolver();

  // Parameter access
  void setParams(const Params &p);
  Params getParams() const;

  // Set joint limits (length must match model_.nq)
  void setJointLimits(const Eigen::VectorXd &lo, const Eigen::VectorXd &hi);

  // Set optional iteration callback
  void setIterCallback(IterCallback cb);

  // Synchronous solve. If timeout_ms==0 -> no timeout.
  Result solve(const SE3 &target, const Eigen::VectorXd &q_init, int timeout_ms = 0);

  // Single-step iteration (advances one internal LM iteration). Returns Result with updated q
  Result step(const SE3 &target, const Eigen::VectorXd &q_current);

  // Utility read-only queries
  Eigen::VectorXd forwardKinematics(const Eigen::VectorXd &q);
  Eigen::MatrixXd getFrameJacobian(const Eigen::VectorXd &q);

  // Query model / frame info
  int getTipFrameId() const;
  std::string getTipFrameName() const;
  int getNq() const;
  int getNv() const;

  // Return tip SE3 for given joint vector (uses internal data_)
  SE3 forwardKinematicsSE3(const Eigen::VectorXd &q);

  // Load a simple YAML-style file (keys:value) to populate Params. Returns true on success.
  bool loadParamsFromFile(const std::string &path);

private:
  // non-copyable
  IkSolver(const IkSolver &) = delete;
  IkSolver &operator=(const IkSolver &) = delete;

  // internal solver (time-bounded)
  Result solveInternal(const SE3 &target, Eigen::VectorXd q_init, const std::chrono::steady_clock::time_point &deadline);

  void clampToJointLimits(Eigen::VectorXd &q);

  // Pinocchio model/data are implementation details; store via pointers to avoid exposing headers here.
  std::unique_ptr<pinocchio::Model> model_;
  std::unique_ptr<pinocchio::Data> data_; // reused across solves
  // tip frame for SE3 queries
  unsigned int tip_frame_id_{0};
  std::string tip_frame_name_; // for informational queries
  Params params_;
  mutable std::mutex mutex_;
  IterCallback iter_cb_{nullptr};

  // cached q indices (if present in URDF/joint names)
  int j4_q_index_{-1};
  int j5_q_index_{-1};
  int j7_q_index_{-1};
  int j3_q_index_{-1};
};

} // namespace ros2_ik_cpp
