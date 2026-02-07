#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include "ik_solver.hpp"
#include <chrono>
#include <cmath>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <stdexcept>
#include <cctype>

using namespace ros2_ik_cpp;
using Eigen::VectorXd;
using Eigen::MatrixXd;

// Helper: convert Eigen::Isometry3d (public SE3) to pinocchio::SE3
static pinocchio::SE3 toPinocchioSE3(const SE3 &s) {
  Eigen::Quaterniond q(s.rotation());
  return pinocchio::SE3(q, s.translation());
}

// Helper: convert pinocchio::SE3 to Eigen::Isometry3d
static SE3 fromPinocchioSE3(const pinocchio::SE3 &p) {
  SE3 s = SE3::Identity();
  s.linear() = p.rotation();
  s.translation() = p.translation();
  return s;
}

namespace {
struct IkYamlConfig {
  std::string urdf_path;
  std::string tip_frame;
  ros2_ik_cpp::IkSolver::Params params;
};

std::string trim_copy(const std::string &s) {
  const auto start = s.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) return "";
  const auto end = s.find_last_not_of(" \t\r\n");
  return s.substr(start, end - start + 1);
}

std::string strip_quotes(const std::string &s) {
  if (s.size() >= 2) {
    if ((s.front() == '"' && s.back() == '"') || (s.front() == '\'' && s.back() == '\'')) {
      return s.substr(1, s.size() - 2);
    }
  }
  return s;
}

bool parse_bool(const std::string &val) {
  std::string v;
  v.reserve(val.size());
  for (char c : val) v.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  v = trim_copy(v);
  return (v == "true" || v == "1" || v == "yes" || v == "on");
}

std::vector<double> parse_list(const std::string &val) {
  std::vector<double> out;
  const auto a = val.find('[');
  const auto b = val.find(']');
  if (a == std::string::npos || b == std::string::npos || b <= a) return out;
  const std::string body = val.substr(a + 1, b - a - 1);
  std::stringstream ss(body);
  double x;
  char ch;
  while (ss >> x) {
    out.push_back(x);
    ss >> ch;
  }
  return out;
}

void apply_key_value(const std::string &key,
                     const std::string &val,
                     IkYamlConfig &cfg) {
  if (key == "urdf" || key == "urdf_path" || key == "robot_urdf") {
    cfg.urdf_path = strip_quotes(val);
  } else if (key == "tip" || key == "tip_frame" || key == "tip_frame_name") {
    cfg.tip_frame = strip_quotes(val);
  } else if (key == "max_iters" || key == "ik_max_iterations") {
    cfg.params.max_iters = std::stoi(val);
  } else if (key == "eps" || key == "ik_epsilon") {
    cfg.params.eps = std::stod(val);
  } else if (key == "eps_relaxed_6d") {
    cfg.params.eps_relaxed_6d = std::stod(val);
  } else if (key == "pos_weight") {
    cfg.params.pos_weight = std::stod(val);
  } else if (key == "ang_weight") {
    cfg.params.ang_weight = std::stod(val);
  } else if (key == "use_numeric_jacobian") {
    cfg.params.use_numeric_jacobian = parse_bool(val);
  } else if (key == "use_svd_damped") {
    cfg.params.use_svd_damped = parse_bool(val);
  } else if (key == "ik_svd_damping") {
    cfg.params.ik_svd_damping = std::stod(val);
  } else if (key == "ik_svd_damping_min") {
    cfg.params.ik_svd_damping_min = std::stod(val);
  } else if (key == "ik_svd_damping_max") {
    cfg.params.ik_svd_damping_max = std::stod(val);
  } else if (key == "ik_svd_damping_reduce_factor") {
    cfg.params.ik_svd_damping_reduce_factor = std::stod(val);
  } else if (key == "ik_svd_damping_increase_factor") {
    cfg.params.ik_svd_damping_increase_factor = std::stod(val);
  } else if (key == "ik_svd_trunc_tol") {
    cfg.params.ik_svd_trunc_tol = std::stod(val);
  } else if (key == "ik_svd_min_rel_reduction") {
    cfg.params.ik_svd_min_rel_reduction = std::stod(val);
  } else if (key == "max_delta") {
    cfg.params.max_delta = std::stod(val);
  } else if (key == "max_delta_min") {
    cfg.params.max_delta_min = std::stod(val);
  } else if (key == "nullspace_penalty_scale") {
    cfg.params.nullspace_penalty_scale = std::stod(val);
  } else if (key == "joint4_penalty_threshold") {
    cfg.params.joint4_penalty_threshold = std::stod(val);
  } else if (key == "numeric_fallback_after_rejects") {
    cfg.params.numeric_fallback_after_rejects = std::stoi(val);
  } else if (key == "numeric_fallback_duration") {
    cfg.params.numeric_fallback_duration = std::stoi(val);
  } else if (key == "ik_step_size") {
    cfg.params.ik_step_size = std::stod(val);
  } else if (key == "joint_limits_min") {
    cfg.params.joint_limits_min = parse_list(val);
  } else if (key == "joint_limits_max") {
    cfg.params.joint_limits_max = parse_list(val);
  } else if (key == "timeout_ms") {
    cfg.params.timeout_ms = std::stoi(val);
  }
}

bool parse_ik_yaml(const std::string &path, IkYamlConfig &cfg) {
  std::ifstream ifs(path);
  if (!ifs) return false;
  bool in_planning = false;
  std::string line;
  while (std::getline(ifs, line)) {
    auto comment_pos = line.find('#');
    if (comment_pos != std::string::npos) {
      line = line.substr(0, comment_pos);
    }
    line = trim_copy(line);
    if (line.empty()) continue;
    if (line.rfind("planning:", 0) == 0) {
      in_planning = true;
      continue;
    }
    const auto colon = line.find(':');
    if (colon == std::string::npos) continue;
    std::string key = trim_copy(line.substr(0, colon));
    std::string val = trim_copy(line.substr(colon + 1));
    if (in_planning) {
      apply_key_value(key, val, cfg);
    } else {
      apply_key_value(key, val, cfg);
    }
  }
  return true;
}
} // namespace

static std::string load_urdf_xml(const std::string &urdf_xml_or_path) {
  if (urdf_xml_or_path.find("<robot") != std::string::npos ||
      urdf_xml_or_path.find("<?xml") != std::string::npos) {
    return urdf_xml_or_path;
  }

  std::ifstream ifs(urdf_xml_or_path);
  if (!ifs) {
    throw std::runtime_error("Failed to open URDF file: " + urdf_xml_or_path);
  }
  std::ostringstream oss;
  oss << ifs.rdbuf();
  std::string xml = oss.str();
  if (xml.empty()) {
    throw std::runtime_error("URDF file is empty: " + urdf_xml_or_path);
  }
  return xml;
}

// Construct from URDF XML string and tip frame name
IkSolver::IkSolver(const std::string &urdf_xml, const std::string &tip_frame_name)
  : model_(nullptr), data_(nullptr), tip_frame_name_(tip_frame_name) {
  try {
    model_ = std::make_unique<pinocchio::Model>();
    const std::string xml = load_urdf_xml(urdf_xml);
    pinocchio::urdf::buildModelFromXML(xml, *model_);
    data_ = std::make_unique<pinocchio::Data>(*model_);
    tip_frame_id_ = model_->getFrameId(tip_frame_name_);
  } catch (const std::exception &e) {
    throw std::runtime_error(std::string("Failed to build Pinocchio model: ") + e.what());
  }
}

IkSolver::IkSolver(const std::string &urdf_xml,
                   const std::string &tip_frame_name,
                   const std::string &yaml_path)
  : IkSolver(urdf_xml, tip_frame_name) {
  if (!yaml_path.empty()) {
    if (!loadParamsFromFile(yaml_path)) {
      throw std::runtime_error("Failed to load IK params YAML: " + yaml_path);
    }
  }
}

IkSolver::IkSolver(const std::string &yaml_path)
  : model_(nullptr), data_(nullptr), tip_frame_name_() {
  IkYamlConfig cfg;
  if (!parse_ik_yaml(yaml_path, cfg)) {
    throw std::runtime_error("Failed to read IK YAML: " + yaml_path);
  }
  if (cfg.urdf_path.empty() || cfg.tip_frame.empty()) {
    throw std::runtime_error("IK YAML missing urdf_path or tip_frame: " + yaml_path);
  }
  model_ = std::make_unique<pinocchio::Model>();
  const std::string xml = load_urdf_xml(cfg.urdf_path);
  pinocchio::urdf::buildModelFromXML(xml, *model_);
  data_ = std::make_unique<pinocchio::Data>(*model_);
  tip_frame_name_ = cfg.tip_frame;
  tip_frame_id_ = model_->getFrameId(tip_frame_name_);
  setParams(cfg.params);
  if (cfg.params.joint_limits_min.size() == cfg.params.joint_limits_max.size() &&
      !cfg.params.joint_limits_min.empty()) {
    Eigen::VectorXd lo(cfg.params.joint_limits_min.size());
    Eigen::VectorXd hi(cfg.params.joint_limits_max.size());
    for (size_t i = 0; i < cfg.params.joint_limits_min.size(); ++i) {
      lo[i] = cfg.params.joint_limits_min[i];
      hi[i] = cfg.params.joint_limits_max[i];
    }
    setJointLimits(lo, hi);
  }
}

IkSolver::~IkSolver() {}

IkSolver::SE3 IkSolver::makeSE3(double px, double py, double pz,
                                double qx, double qy, double qz, double qw) {
  SE3 se3 = SE3::Identity();
  Eigen::Quaterniond q(qw, qx, qy, qz);
  se3.linear() = q.toRotationMatrix();
  se3.translation() = Eigen::Vector3d(px, py, pz);
  return se3;
}

IkSolver::SE3 IkSolver::makeSE3(const geometry_msgs::msg::Pose &pose) {
  return makeSE3(pose.position.x, pose.position.y, pose.position.z,
                 pose.orientation.x, pose.orientation.y,
                 pose.orientation.z, pose.orientation.w);
}

IkSolver::Result IkSolver::solvePose(const geometry_msgs::msg::Pose &pose,
                                     const Eigen::VectorXd &q_init,
                                     int timeout_ms) {
  return solve(makeSE3(pose), q_init, timeout_ms);
}

void IkSolver::setParams(const Params &p) {
  std::lock_guard<std::mutex> lk(mutex_);
  params_ = p;
}

IkSolver::Params IkSolver::getParams() const {
  std::lock_guard<std::mutex> lk(mutex_);
  return params_;
}

void IkSolver::setJointLimits(const Eigen::VectorXd &lo, const Eigen::VectorXd &hi) {
  std::lock_guard<std::mutex> lk(mutex_);
  if (!model_) return;
  if ((int)lo.size() == model_->nq && (int)hi.size() == model_->nq) {
    params_.joint_limits_min.assign(lo.data(), lo.data() + lo.size());
    params_.joint_limits_max.assign(hi.data(), hi.data() + hi.size());
  }
}

void IkSolver::setIterCallback(IterCallback cb) {
  std::lock_guard<std::mutex> lk(mutex_);
  iter_cb_ = cb;
}

bool IkSolver::loadParamsFromFile(const std::string &path) {
  IkYamlConfig cfg;
  if (!parse_ik_yaml(path, cfg)) {
    return false;
  }
  setParams(cfg.params);
  if (cfg.params.joint_limits_min.size() == cfg.params.joint_limits_max.size() &&
      !cfg.params.joint_limits_min.empty()) {
    Eigen::VectorXd lo(cfg.params.joint_limits_min.size());
    Eigen::VectorXd hi(cfg.params.joint_limits_max.size());
    for (size_t i = 0; i < cfg.params.joint_limits_min.size(); ++i) {
      lo[i] = cfg.params.joint_limits_min[i];
      hi[i] = cfg.params.joint_limits_max[i];
    }
    setJointLimits(lo, hi);
  }
  return true;
}

IkSolver::Result IkSolver::solve(const SE3 &target, const Eigen::VectorXd &q_init, int timeout_ms) {
  auto deadline = (timeout_ms <= 0) ? std::chrono::steady_clock::time_point::max()
                                     : std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  Eigen::VectorXd q0 = q_init;
  return solveInternal(target, q0, deadline);
}

IkSolver::Result IkSolver::step(const SE3 &target, const Eigen::VectorXd &q_current) {
  auto dl = std::chrono::steady_clock::now() + std::chrono::milliseconds(1);
  return solveInternal(target, q_current, dl);
}

Eigen::VectorXd IkSolver::forwardKinematics(const Eigen::VectorXd &q) {
  std::lock_guard<std::mutex> lk(mutex_);
  if (!model_ || !data_) return Eigen::VectorXd();
  pinocchio::forwardKinematics(*model_, *data_, q);
  pinocchio::updateFramePlacements(*model_, *data_);
  Eigen::VectorXd out(6);
  out.head<3>() = data_->oMf.back().translation();
  Eigen::Quaterniond qd(data_->oMf.back().rotation());
  out.tail<3>() = Eigen::Vector3d(qd.x(), qd.y(), qd.z());
  return out;
}

SE3 IkSolver::forwardKinematicsSE3(const Eigen::VectorXd &q) {
  std::lock_guard<std::mutex> lk(mutex_);
  SE3 s = SE3::Identity();
  if (!model_ || !data_) return s;
  pinocchio::forwardKinematics(*model_, *data_, q);
  pinocchio::updateFramePlacements(*model_, *data_);
  const pinocchio::SE3 &pose = data_->oMf[tip_frame_id_];
  s.linear() = pose.rotation();
  s.translation() = pose.translation();
  return s;
}

Eigen::MatrixXd IkSolver::getFrameJacobian(const Eigen::VectorXd &q) {
  std::lock_guard<std::mutex> lk(mutex_);
  if (!model_ || !data_) return Eigen::MatrixXd();
  pinocchio::forwardKinematics(*model_, *data_, q);
  pinocchio::updateFramePlacements(*model_, *data_);
  const int fid = static_cast<int>(std::max<size_t>(1, model_->nframes) - 1);
  Eigen::MatrixXd J = pinocchio::getFrameJacobian(*model_, *data_, fid, pinocchio::WORLD);
  return J;
}

void IkSolver::clampToJointLimits(Eigen::VectorXd &q) {
  std::lock_guard<std::mutex> lk(mutex_);
  if (params_.joint_limits_min.size() == (size_t)q.size() && params_.joint_limits_max.size() == (size_t)q.size()) {
    for (int i = 0; i < q.size(); ++i) {
      double lo = params_.joint_limits_min[i];
      double hi = params_.joint_limits_max[i];
      if (lo >= hi) continue;
      if (q[i] < lo) q[i] = lo;
      if (q[i] > hi) q[i] = hi;
    }
  }
}

IkSolver::Result IkSolver::solveInternal(const SE3 &target_se3, Eigen::VectorXd q_init, const std::chrono::steady_clock::time_point &deadline) {
  Result res;
  auto tstart = std::chrono::steady_clock::now();

  // copy params safely
  Params p;
  {
    std::lock_guard<std::mutex> lk(mutex_);
    p = params_;
  }

  if (!model_ || !data_) {
    res.success = false; res.status = 0; res.diagnostic = "no model"; return res;
  }

  // convert public SE3 to pinocchio::SE3 for internal computations
  pinocchio::SE3 target = toPinocchioSE3(target_se3);

  const int nv = model_->nv;
  Eigen::VectorXd q_solution = q_init;
  pinocchio::Data data_local(*model_);

  double best_error = 1e300;
  Eigen::VectorXd best_q = q_solution;
  int best_iter = -1;

  bool precise_success = false;
  int precise_iter = -1;
  Eigen::VectorXd precise_q;
  double precise_error = 0.0;

  Eigen::Matrix<double,6,1> err6;

  int consecutive_ls_rejects = 0;
  int numeric_fallback_remaining = 0;
  bool numeric_force_active = false;
  int consecutive_max_damping_rejections = 0;

  // attempt to cache joint indices for j3/j4/j5/j7 if names available
  try {
    int id4 = model_->getJointId("joint4"); if (id4 >= 0 && id4 < (int)model_->joints.size()) j4_q_index_ = model_->joints[id4].idx_q();
    int id3 = model_->getJointId("joint3"); if (id3 >= 0 && id3 < (int)model_->joints.size()) j3_q_index_ = model_->joints[id3].idx_q();
    int id5 = model_->getJointId("joint5"); if (id5 >= 0 && id5 < (int)model_->joints.size()) j5_q_index_ = model_->joints[id5].idx_q();
    int id7 = model_->getJointId("joint7"); if (id7 >= 0 && id7 < (int)model_->joints.size()) j7_q_index_ = model_->joints[id7].idx_q();
  } catch(...) {}

  for (int it = 0; it < p.max_iters; ++it) {
    if (std::chrono::steady_clock::now() > deadline) { res.success=false; res.status=0; res.diagnostic="timeout"; break; }

    pinocchio::forwardKinematics(*model_, data_local, q_solution);
    pinocchio::updateFramePlacements(*model_, data_local);
    const pinocchio::SE3 &current_pose = data_local.oMf.back();
    Eigen::Vector3d pos_cur = current_pose.translation();
    Eigen::Vector3d pos_tgt = target.translation();
    Eigen::Vector3d pos_err = pos_tgt - pos_cur;
    Eigen::Quaterniond qcur(current_pose.rotation());
    Eigen::Quaterniond qtgt(target.rotation());
    Eigen::Quaterniond qerr = qtgt * qcur.conjugate(); qerr.normalize();
    Eigen::AngleAxisd aa(qerr); Eigen::Vector3d ang_err = Eigen::Vector3d::Zero();
    double angle = aa.angle(); if (std::isfinite(angle) && std::abs(angle) > 1e-12) ang_err = aa.axis() * angle;
    err6.head<3>() = pos_err; err6.tail<3>() = ang_err; double cur_err = err6.norm();

    if (cur_err < best_error) { best_error = cur_err; best_q = q_solution; best_iter = it; }
    if (cur_err < p.eps) {
      precise_success = true; precise_iter = it; precise_q = q_solution; precise_error = cur_err; break; }

    pinocchio::computeJointJacobians(*model_, data_local, q_solution);
    const int fid = static_cast<int>(std::max<size_t>(1, model_->nframes) - 1);
    MatrixXd J_world = pinocchio::getFrameJacobian(*model_, data_local, fid, pinocchio::WORLD);

    const double eps_se3 = 1e-6;
    Eigen::Matrix<double,6,6> Mmat = Eigen::Matrix<double,6,6>::Zero();
    for (int k=0;k<6;++k) {
      Eigen::Matrix<double,6,1> xi = Eigen::Matrix<double,6,1>::Zero(); xi(k)=eps_se3;
      Eigen::Vector3d dv = xi.head<3>(); Eigen::Vector3d dw = xi.tail<3>();
      Eigen::Quaterniond q_delta; double ang2 = dw.norm();
      if (ang2 < 1e-12) q_delta = Eigen::Quaterniond::Identity(); else q_delta = Eigen::Quaterniond(Eigen::AngleAxisd(ang2, dw/ang2));
      Eigen::Quaterniond qcur2(current_pose.rotation()); Eigen::Quaterniond qpert = q_delta * qcur2; qpert.normalize();
      Eigen::Vector3d tpert = current_pose.translation() + dv; pinocchio::SE3 pert_se3(qpert, tpert);
      Eigen::Vector3d pos_err_pert = target.translation() - pert_se3.translation();
      Eigen::Quaterniond qerr_pert = qtgt * Eigen::Quaterniond(pert_se3.rotation()).conjugate(); qerr_pert.normalize();
      Eigen::AngleAxisd aa_pert(qerr_pert); Eigen::Vector3d ang_err_pert = Eigen::Vector3d::Zero(); double angle_pert = aa_pert.angle();
      if (std::isfinite(angle_pert) && std::abs(angle_pert) > 1e-12) ang_err_pert = aa_pert.axis() * angle_pert;
      Eigen::Matrix<double,6,1> err6_pert; err6_pert.head<3>() = pos_err_pert; err6_pert.tail<3>() = ang_err_pert; Mmat.col(k) = (err6_pert - err6) / eps_se3;
    }

    MatrixXd J_err = - Mmat * J_world;

    MatrixXd numJ = MatrixXd::Zero(6, nv); bool numeric_computed=false;
    if (p.use_numeric_jacobian) {
      const double eps_q = 1e-6;
      for (int j=0;j<nv;++j) {
        Eigen::VectorXd dq = Eigen::VectorXd::Zero(nv);
        dq[j]=eps_q;
        Eigen::VectorXd q_pert(q_solution.size());
        pinocchio::integrate(*model_, q_solution, dq, q_pert);
        pinocchio::Data data_pert(*model_);
        pinocchio::forwardKinematics(*model_, data_pert, q_pert);
        pinocchio::updateFramePlacements(*model_, data_pert);
        const pinocchio::SE3 &pose_pert = data_pert.oMf.back();
        Eigen::Vector3d pos_err_pert = target.translation() - pose_pert.translation();
        Eigen::Quaterniond qcur_pert(pose_pert.rotation());
        Eigen::Quaterniond qerr_pert = qtgt * qcur_pert.conjugate(); qerr_pert.normalize();
        Eigen::AngleAxisd aa_pert(qerr_pert);
        Eigen::Vector3d ang_err_pert = Eigen::Vector3d::Zero();
        double angle_pert = aa_pert.angle();
        if (std::isfinite(angle_pert) && std::abs(angle_pert) > 1e-12) ang_err_pert = aa_pert.axis() * angle_pert;
        Eigen::Matrix<double,6,1> err6_pert;
        err6_pert.head<3>() = pos_err_pert;
        err6_pert.tail<3>() = ang_err_pert;
        numJ.col(j) = (err6_pert - err6) / eps_q;
      }
      numeric_computed = true;
    }

    MatrixXd J_used = ( (p.use_numeric_jacobian && numeric_computed) || numeric_force_active) ? numJ : J_err;

    Eigen::Matrix<double,6,6> W = Eigen::Matrix<double,6,6>::Identity();
    for (int ri=0;ri<3;++ri) W(ri,ri)=p.pos_weight;
    for (int ri=3;ri<6;++ri) W(ri,ri)=p.ang_weight;
    MatrixXd Jw = J_used;
    for (int ri=0; ri<6; ++ri) Jw.row(ri) *= W(ri,ri);
    Eigen::Matrix<double,6,1> errw = err6; errw.head<3>() *= p.pos_weight; errw.tail<3>() *= p.ang_weight;

    Eigen::JacobiSVD<MatrixXd> svd(Jw, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd s = svd.singularValues(); int sn = (int)s.size();
    double cond = std::numeric_limits<double>::infinity(); if (sn>0 && s(sn-1)>0) cond = s(0)/s(sn-1);

    MatrixXd A = Jw.transpose() * Jw;
    Eigen::VectorXd g = Jw.transpose() * errw;

    double alpha_ns = 0.0;
    if (sn>0) {
      Eigen::VectorXd ns_vec = svd.matrixV().col(sn-1);
      double c3 = (j3_q_index_>=0) ? ns_vec[j3_q_index_] : 0.0;
      double c5 = (j5_q_index_>=0) ? ns_vec[j5_q_index_] : 0.0;
      double c7 = (j7_q_index_>=0) ? ns_vec[j7_q_index_] : 0.0;
      bool pair_j5j7 = (j5_q_index_>=0 && j7_q_index_>=0) && (c5*c7<0.0) && (std::abs(c5)>1e-6) && (std::abs(c7)>1e-6);
      bool pair_j3j5 = (j3_q_index_>=0 && j5_q_index_>=0) && (c3*c5<0.0) && (std::abs(c3)>1e-6) && (std::abs(c5)>1e-6);
      if (j4_q_index_>=0 && std::abs(q_solution[j4_q_index_]) < p.joint4_penalty_threshold && (pair_j5j7 || pair_j3j5)) {
        double smax = (s.size()>0) ? s(0) : 1.0;
        alpha_ns = p.nullspace_penalty_scale * (smax*smax);
        A += alpha_ns * (ns_vec * ns_vec.transpose());
      }
    }

    double lambda = std::max(p.ik_svd_damping_min, std::min(p.ik_svd_damping, p.ik_svd_damping_max));
    const int lm_max_attempts = 8;
    double predicted_reduction = -1.0;
    Eigen::VectorXd v = Eigen::VectorXd::Zero(nv);
    Eigen::VectorXd v_candidate = Eigen::VectorXd::Zero(nv);
    for (int attempt=0; attempt<lm_max_attempts; ++attempt) {
      MatrixXd Ad = A;
      Ad.diagonal().array() += lambda;
      Eigen::VectorXd rhs = -g;
      Eigen::LDLT<MatrixXd> ldlt(Ad);
      if (ldlt.info() != Eigen::Success) {
        lambda *= p.ik_svd_damping_increase_factor;
        if (lambda>p.ik_svd_damping_max) lambda=p.ik_svd_damping_max;
        continue;
      }
      v_candidate = ldlt.solve(rhs);
      Eigen::VectorXd pred_errw = errw + Jw * v_candidate;
      double errw_norm2 = errw.squaredNorm();
      double pred_errw_norm2 = pred_errw.squaredNorm();
      predicted_reduction = 0.5 * (errw_norm2 - pred_errw_norm2);
      if (predicted_reduction > 0) break;
      lambda *= p.ik_svd_damping_increase_factor;
      if (lambda>p.ik_svd_damping_max) { lambda=p.ik_svd_damping_max; break; }
    }
    p.ik_svd_damping = std::max(p.ik_svd_damping_min, std::min(lambda, p.ik_svd_damping_max));
    v = v_candidate;

    Eigen::VectorXd delta_full = v * p.ik_step_size;

    bool accepted = false;
    Eigen::VectorXd q_candidate = q_solution;
    double best_trial_err = cur_err;
    Eigen::VectorXd best_trial_q = q_solution;
    std::vector<double> alphas = {1.0, 0.5, 0.25, 0.125};
    for (double alpha : alphas) {
      Eigen::VectorXd delta_try = delta_full * alpha;
      for (int i=0;i<delta_try.size();++i) delta_try[i] = std::max(-p.max_delta, std::min(p.max_delta, delta_try[i]));
      if (p.joint_limits_min.size() == (size_t)q_solution.size() && p.joint_limits_max.size() == (size_t)q_solution.size()) {
        Eigen::VectorXd q_try_est = q_solution + delta_try;
        for (int i=0;i<q_try_est.size();++i) { double lo=p.joint_limits_min[i], hi=p.joint_limits_max[i]; if (lo<hi) { if (q_try_est[i]<lo) q_try_est[i]=lo; if (q_try_est[i]>hi) q_try_est[i]=hi; } }
        delta_try = q_try_est - q_solution;
      }
      Eigen::VectorXd q_try(q_solution.size());
      pinocchio::integrate(*model_, q_solution, delta_try, q_try);
      pinocchio::forwardKinematics(*model_, data_local, q_try);
      pinocchio::updateFramePlacements(*model_, data_local);
      const pinocchio::SE3 &pose_try = data_local.oMf.back();
      Eigen::Vector3d pos_err_try = target.translation() - pose_try.translation();
      Eigen::Quaterniond qcur_try(pose_try.rotation());
      Eigen::Quaterniond qerr_try = qtgt * qcur_try.conjugate(); qerr_try.normalize();
      Eigen::AngleAxisd aa_try(qerr_try);
      Eigen::Vector3d ang_err_try = Eigen::Vector3d::Zero();
      double angle_try = aa_try.angle();
      if (std::isfinite(angle_try) && std::abs(angle_try) > 1e-12) ang_err_try = aa_try.axis() * angle_try;
      Eigen::Matrix<double,6,1> err6_try;
      err6_try.head<3>() = pos_err_try;
      err6_try.tail<3>() = ang_err_try;
      double err_try = err6_try.norm();
      if (err_try < cur_err) {
        accepted = true;
        q_candidate = q_try;
        best_trial_err = err_try;
        best_trial_q = q_try;
        break;
      } else {
        if (err_try < best_trial_err) { best_trial_err = err_try; best_trial_q = q_try; }
      }
    }

    if (accepted) {
      q_solution = q_candidate;
      consecutive_ls_rejects = 0;
      if (numeric_force_active && numeric_fallback_remaining>0) {
        --numeric_fallback_remaining;
        if (numeric_fallback_remaining<=0) numeric_force_active=false;
      }
      double rel_red = (cur_err - best_trial_err) / std::max(cur_err, 1e-12);
      if (rel_red > p.ik_svd_min_rel_reduction) {
        p.ik_svd_damping = std::max(p.ik_svd_damping_min, p.ik_svd_damping * p.ik_svd_damping_reduce_factor);
      }
    } else {
      ++consecutive_ls_rejects;
      if (!numeric_force_active && numeric_computed && consecutive_ls_rejects >= p.numeric_fallback_after_rejects) {
        numeric_force_active=true;
        numeric_fallback_remaining=p.numeric_fallback_duration;
      }
      double tiny_tol=1e-10;
      double movement = (best_trial_q - q_solution).norm();
      if (best_trial_err <= cur_err + tiny_tol && movement > 1e-8) {
        q_solution = best_trial_q;
        p.ik_svd_damping = std::max(p.ik_svd_damping_min, p.ik_svd_damping * p.ik_svd_damping_reduce_factor);
        consecutive_max_damping_rejections = 0;
      } else {
        p.ik_svd_damping *= p.ik_svd_damping_increase_factor;
        if (p.ik_svd_damping > p.ik_svd_damping_max) p.ik_svd_damping = p.ik_svd_damping_max;
        if (p.ik_svd_damping >= p.ik_svd_damping_max * 0.999) ++consecutive_max_damping_rejections; else consecutive_max_damping_rejections = 0;
        if (consecutive_max_damping_rejections >= 3) {
          double old_max_delta = p.max_delta;
          p.max_delta = std::max(p.max_delta_min, p.max_delta * 0.5);
          p.ik_svd_damping = std::max(p.ik_svd_damping_min, p.ik_svd_damping * p.ik_svd_damping_reduce_factor);
          consecutive_max_damping_rejections = 0;
        }
      }
    }

    if (iter_cb_) iter_cb_(it, q_solution, cur_err);
  }

  double elapsed = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(std::chrono::steady_clock::now() - tstart).count();
  res.elapsed_ms = elapsed;

  if (precise_success) {
    res.success = true;
    res.status = 1; // precise success
    res.q = precise_q;
    res.final_error = precise_error;
    res.iterations = precise_iter;
    return res;
  }

  res.q = best_q;
  res.final_error = best_error;
  res.iterations = (best_iter >= 0) ? best_iter : 0;
  double relaxed_threshold = p.eps_relaxed_6d;
  if (relaxed_threshold <= 0.0) relaxed_threshold = p.eps * 100.0;
  res.success = (best_error < relaxed_threshold);
  res.status = res.success ? 2 : 0;
  return res;
}

// Compute the 6D error (position + angle-axis) between target and FK at q.
// Returns norm([pos_error; angle_axis_error]).
double IkSolver::computeError(const SE3 &target, const Eigen::VectorXd &q) const {
  if (!model_) return std::numeric_limits<double>::infinity();
  // Local data to avoid mutating shared data_ and to be thread-safe.
  pinocchio::Data data_local(*model_);
  pinocchio::forwardKinematics(*model_, data_local, q);
  pinocchio::updateFramePlacements(*model_, data_local);
  const pinocchio::SE3 &current_pose = data_local.oMf[tip_frame_id_];
  Eigen::Vector3d pos_err = target.translation() - current_pose.translation();
  Eigen::Quaterniond qcur(current_pose.rotation());
  Eigen::Quaterniond qtgt(target.rotation());
  Eigen::Quaterniond qerr = qtgt * qcur.conjugate(); qerr.normalize();
  Eigen::AngleAxisd aa(qerr);
  Eigen::Vector3d ang_err = Eigen::Vector3d::Zero();
  double angle = aa.angle();
  if (std::isfinite(angle) && std::abs(angle) > 1e-12) ang_err = aa.axis() * angle;
  Eigen::Matrix<double,6,1> err6; err6.head<3>() = pos_err; err6.tail<3>() = ang_err;
  return err6.norm();
}

int IkSolver::getTipFrameId() const { return static_cast<int>(tip_frame_id_); }
std::string IkSolver::getTipFrameName() const { return tip_frame_name_; }
int IkSolver::getNq() const { return model_ ? model_->nq : 0; }
int IkSolver::getNv() const { return model_ ? model_->nv : 0; }

std::string IkSolver::makeInitLog(const geometry_msgs::msg::Pose &pose_msg, const Eigen::VectorXd &q_init, const Result &r, bool include_solution) const {
   std::ostringstream oss;
   oss.setf(std::ios::fixed);
   oss<<std::setprecision(6);
   if (!model_) {
     oss << "init_err=inf result=time_ms=" << std::fixed << std::setprecision(3) << r.elapsed_ms << " success=" << (int)r.success;
     return oss.str();
   }

   // convert ROS pose to internal SE3
   SE3 target = makeSE3(pose_msg);
 
   // Local data for FK to keep this method const-safe
   pinocchio::Data data_local(*model_);
   pinocchio::forwardKinematics(*model_, data_local, q_init);
   pinocchio::updateFramePlacements(*model_, data_local);
   const pinocchio::SE3 &frame_pose = data_local.oMf[tip_frame_id_];
   SE3 init_fk = SE3::Identity();
   init_fk.linear() = frame_pose.rotation();
   init_fk.translation() = frame_pose.translation();

   Eigen::Vector3d pos_init = init_fk.translation();
   Eigen::Quaterniond qinit(init_fk.rotation());
   // compute rpy from quaternion (Z,Y,X -> yaw,pitch,roll)
   auto quat_to_rpy_local = [](const Eigen::Quaterniond &q, double &roll, double &pitch, double &yaw) {
     Eigen::Matrix3d R = q.toRotationMatrix();
     Eigen::Vector3d ypr = R.eulerAngles(2, 1, 0);
     yaw = ypr[0]; pitch = ypr[1]; roll = ypr[2];
   };
   double r_init=0,p_init=0,y_init=0; quat_to_rpy_local(qinit, r_init, p_init, y_init);

   // target
   Eigen::Vector3d pos_tgt = target.translation();
   Eigen::Quaterniond qtgt(target.rotation());
   double r_t=0,p_t=0,y_t=0; quat_to_rpy_local(qtgt, r_t, p_t, y_t);

   // init_err
   double init_err = computeError(target, q_init);
   oss << "init_err=" << init_err << " init_fk=pos(" << pos_init.x() << "," << pos_init.y() << "," << pos_init.z() << ") rpy(" << r_init << "," << p_init << "," << y_init << ")";
   oss << " target=pos(" << pos_tgt.x() << "," << pos_tgt.y() << "," << pos_tgt.z() << ") rpy(" << r_t << "," << p_t << "," << y_t << ")";
   oss << " result=time_ms=" << std::fixed << std::setprecision(3) << r.elapsed_ms << " success=" << (int)r.success << " iters=" << r.iterations << " final_err=" << std::setprecision(6) << r.final_error;
   if (include_solution && r.q.size()>0) {
     oss << " q=[";
     for (int i=0;i<r.q.size();++i) { if (i) oss<<", "; oss<<std::fixed<<std::setprecision(6)<<r.q[i]; }
     oss << "]";
   }
   return oss.str();
 }
