#pragma once

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <Eigen/Dense>
#include <functional>
#include <mutex>
#include <chrono>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

namespace ros2_ik_cpp {

class IkSolver {
public:
  struct Params {
    int max_iters = 200;
    double eps = 1e-4;
    double eps_relaxed_3d = 5e-3;
    double eps_relaxed_6d = 1e-2;
    double dt = 0.1; // step scale
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

  // Construct with an existing Pinocchio model (copy)
  explicit IkSolver(const pinocchio::Model &model);
  ~IkSolver();

  // Parameter access
  void setParams(const Params &p);
  Params getParams() const;

  // Set joint limits (length must match model_.nq)
  void setJointLimits(const Eigen::VectorXd &lo, const Eigen::VectorXd &hi);

  // Set optional iteration callback
  void setIterCallback(IterCallback cb);

  // Synchronous solve. If timeout_ms==0 -> no timeout.
  Result solve(const pinocchio::SE3 &target, const Eigen::VectorXd &q_init, int timeout_ms = 0);

  // Single-step iteration (advances one internal LM iteration). Returns Result with updated q
  Result step(const pinocchio::SE3 &target, const Eigen::VectorXd &q_current);

  // Utility read-only queries
  Eigen::VectorXd forwardKinematics(const Eigen::VectorXd &q);
  Eigen::MatrixXd getFrameJacobian(const Eigen::VectorXd &q);

  // Load a simple YAML-style file (keys:value) to populate Params. Returns true on success.
  bool loadParamsFromFile(const std::string &path);

private:
  // non-copyable
  IkSolver(const IkSolver &) = delete;
  IkSolver &operator=(const IkSolver &) = delete;

  // internal solver (time-bounded)
  Result solveInternal(const pinocchio::SE3 &target, Eigen::VectorXd q_init, const std::chrono::steady_clock::time_point &deadline);

  void clampToJointLimits(Eigen::VectorXd &q);

  pinocchio::Model model_;
  pinocchio::Data data_; // reused across solves
  Params params_;
  mutable std::mutex mutex_;
  IterCallback iter_cb_{nullptr};

  // cached q indices (if present in URDF/joint names)
  int j4_q_index_{-1};
  int j5_q_index_{-1};
  int j7_q_index_{-1};
  int j3_q_index_{-1};
};

// Inline simple YAML-like loader implementation
inline bool IkSolver::loadParamsFromFile(const std::string &path) {
  std::ifstream ifs(path);
  if (!ifs) return false;
  std::string line;
  bool in_planning = false;
  Params p;
  while (std::getline(ifs, line)) {
    auto l = line.find_first_not_of(" \t\r\n");
    if (l==std::string::npos) continue;
    line = line.substr(l);
    if (line.rfind("planning:",0) == 0) { in_planning = true; continue; }
    if (!in_planning) continue;
    if (line.size()>0 && line[0]=='#') continue;
    auto colon = line.find(':');
    if (colon==std::string::npos) continue;
    std::string key = line.substr(0, colon);
    key.erase(key.find_last_not_of(" \t\r\n")+1);
    std::string val = line.substr(colon+1);
    auto s = val.find_first_not_of(" \t"); if (s!=std::string::npos) val = val.substr(s);
    if (key=="max_delta") p.max_delta = std::stod(val);
    else if (key=="pos_weight") p.pos_weight = std::stod(val);
    else if (key=="ang_weight") p.ang_weight = std::stod(val);
    else if (key=="use_numeric_jacobian") p.use_numeric_jacobian = (val.find("true")!=std::string::npos);
    else if (key=="ik_max_iterations") p.max_iters = std::stoi(val);
    else if (key=="ik_epsilon") p.eps = std::stod(val);
    else if (key=="nullspace_penalty_scale") p.nullspace_penalty_scale = std::stod(val);
    else if (key=="joint_limits_min") {
      size_t a = val.find('['); size_t b = val.find(']'); if (a!=std::string::npos && b!=std::string::npos && b>a) {
        std::string body = val.substr(a+1, b-a-1);
        std::vector<double> vals; std::stringstream ss(body); double x; char ch;
        while (ss >> x) { vals.push_back(x); ss >> ch; }
        p.joint_limits_min = vals;
      }
    } else if (key=="joint_limits_max") {
      size_t a = val.find('['); size_t b = val.find(']'); if (a!=std::string::npos && b!=std::string::npos && b>a) {
        std::string body = val.substr(a+1, b-a-1);
        std::vector<double> vals; std::stringstream ss(body); double x; char ch;
        while (ss >> x) { vals.push_back(x); ss >> ch; }
        p.joint_limits_max = vals;
      }
    }
  }
  setParams(p);
  if (p.joint_limits_min.size() == p.joint_limits_max.size() && p.joint_limits_min.size()>0) {
    Eigen::VectorXd lo(p.joint_limits_min.size()), hi(p.joint_limits_max.size());
    for (size_t i=0;i<p.joint_limits_min.size();++i) { lo[i]=p.joint_limits_min[i]; hi[i]=p.joint_limits_max[i]; }
    setJointLimits(lo, hi);
  }
  return true;
}

} // namespace ros2_ik_cpp
