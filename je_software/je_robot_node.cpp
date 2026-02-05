#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <unordered_map>
#include <array>
#include <algorithm>
#include <iostream>
#include <atomic>
#include <mutex>
#include <cerrno>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <cstdio>

// ===== NEW: logging =====
#include <fstream>
#include <filesystem>
// ========================

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "je_software/msg/end_effector_command.hpp"
#include "je_software/msg/end_effector_command_lr.hpp"
#include "common/msg/oculus_controllers.hpp"
#include "common/msg/oculus_init_joint_state.hpp"
#include "ros2_qos.hpp"
#include "ik_solver.hpp"

#include <zmq.hpp>
#include "nlohmann/json.hpp"
#include <Eigen/Dense>

using namespace std::chrono_literals;

using json = nlohmann::json; // 默认 std::map

#ifndef JE_ENABLE_PUBLISH_STATE_FREQ_LOG
#define JE_ENABLE_PUBLISH_STATE_FREQ_LOG 0
#endif

class JeRobotNode : public rclcpp::Node
{
public:
    JeRobotNode()
        : JeRobotNode(rclcpp::NodeOptions())
    {
    }

    explicit JeRobotNode(const rclcpp::NodeOptions &options)
        : Node("je_robot_node", options),
          context_(1),
          publisher_(context_, zmq::socket_type::pub),
          subscriber_(context_, zmq::socket_type::sub),
          joint_cmd_received_(false),
          state_thread_running_(false)
    {

        // ---------- 声明 & 获取参数 ----------
        this->declare_parameter<std::string>("joint_sub_topic", "/joint_cmd");
        this->declare_parameter<std::string>("end_pose_topic", "/end_pose");
        this->declare_parameter<std::string>("joint_pub_topic", "/joint_states_double_arm");
        this->declare_parameter<std::string>("oculus_controllers_topic", "/oculus_controllers");
        this->declare_parameter<std::string>("oculus_init_joint_state_topic", "/oculus_init_joint_state");
        this->declare_parameter<double>("fps", 50.0);

        // ZMQ 相关参数
        this->declare_parameter<std::string>("robot_ip", "192.168.0.99");
        this->declare_parameter<int>("pub_port", 8001);
        this->declare_parameter<int>("sub_port", 8000);

        // 下发关节指令时的插补时间（秒）
        this->declare_parameter<double>("dt", 0.014);
        this->declare_parameter<double>("dt_init", 5.0);
        this->declare_parameter<double>("oculus_joint_jump_threshold", 1.0);
        this->declare_parameter<double>("oculus_pose_jump_threshold_pos", 0.5);
        this->declare_parameter<double>("oculus_pose_jump_threshold_rpy", 0.5);

        // Gripper ROS2 command topic
        this->declare_parameter<std::string>("gripper_sub_topic", "/end_effector_cmd_lr");
        this->declare_parameter<std::string>("end_effector_mode", "external");

        // ===== NEW: state logging parameters =====
        this->declare_parameter<bool>("state_log_enable", true);
        this->declare_parameter<std::string>("state_log_dir", "./je_robot_logs");
        this->declare_parameter<std::string>("state_log_prefix", "robot_state");
        this->declare_parameter<int>("state_log_flush_every_n", 10);
        // per-solve IK console logging
        this->declare_parameter<bool>("ik_log", false);
        // ========================================

        std::string joint_sub_topic =
            this->get_parameter("joint_sub_topic").as_string();
        std::string end_pose_topic =
            this->get_parameter("end_pose_topic").as_string();
        std::string joint_pub_topic =
            this->get_parameter("joint_pub_topic").as_string();
        std::string oculus_controllers_topic =
            this->get_parameter("oculus_controllers_topic").as_string();
        std::string oculus_init_joint_state_topic =
            this->get_parameter("oculus_init_joint_state_topic").as_string();
        std::string gripper_sub_topic =
            this->get_parameter("gripper_sub_topic").as_string();
        std::string end_effector_mode =
            this->get_parameter("end_effector_mode").as_string();
        double fps = this->get_parameter("fps").as_double();

        std::string robot_ip = this->get_parameter("robot_ip").as_string();
        int pub_port = this->get_parameter("pub_port").as_int();
        int sub_port = this->get_parameter("sub_port").as_int();
        dt_ = this->get_parameter("dt").as_double();
        dt_init_ = this->get_parameter("dt_init").as_double();
        joint_jump_threshold_ = this->get_parameter("oculus_joint_jump_threshold").as_double();
        pose_jump_threshold_pos_ =
            this->get_parameter("oculus_pose_jump_threshold_pos").as_double();
        pose_jump_threshold_rpy_ =
            this->get_parameter("oculus_pose_jump_threshold_rpy").as_double();

        gripper_sub_topic_ = gripper_sub_topic;
        {
            std::transform(end_effector_mode.begin(),
                           end_effector_mode.end(),
                           end_effector_mode.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (end_effector_mode == "msg" || end_effector_mode == "message")
            {
                oculus_init_use_msg_gripper_ = true;
            }
            else if (!end_effector_mode.empty() &&
                     end_effector_mode != "external" &&
                     end_effector_mode != "command")
            {
                RCLCPP_WARN(this->get_logger(),
                            "Unknown end_effector_mode '%s', fallback to 'external'.",
                            end_effector_mode.c_str());
                oculus_init_use_msg_gripper_ = false;
            }
        }

        // IK solver parameters (per-arm, optional)
        this->declare_parameter<std::string>("ik_left_yaml_path", "");
        this->declare_parameter<std::string>("ik_right_yaml_path", "");

        std::string ik_left_yaml_path = this->get_parameter("ik_left_yaml_path").as_string();
        std::string ik_right_yaml_path = this->get_parameter("ik_right_yaml_path").as_string();

        if (!ik_left_yaml_path.empty()) {
            try {
                ik_solver_left_ = std::make_unique<ros2_ik_cpp::IkSolver>(ik_left_yaml_path);
                ik_left_timeout_ms_ = ik_solver_left_->getParams().timeout_ms;
                RCLCPP_INFO(this->get_logger(),
                            "IkSolver left initialized from YAML: %s",
                            ik_left_yaml_path.c_str());
            } catch (const std::exception &e) {
                RCLCPP_WARN(this->get_logger(), "Failed to create IkSolver left: %s", e.what());
            }
        } else {
            RCLCPP_WARN(this->get_logger(), "No ik_left_yaml_path provided; left IK disabled.");
        }

        if (!ik_right_yaml_path.empty()) {
            try {
                ik_solver_right_ = std::make_unique<ros2_ik_cpp::IkSolver>(ik_right_yaml_path);
                ik_right_timeout_ms_ = ik_solver_right_->getParams().timeout_ms;
                RCLCPP_INFO(this->get_logger(),
                            "IkSolver right initialized from YAML: %s",
                            ik_right_yaml_path.c_str());
            } catch (const std::exception &e) {
                RCLCPP_WARN(this->get_logger(), "Failed to create IkSolver right: %s", e.what());
            }
        } else {
            RCLCPP_WARN(this->get_logger(), "No ik_right_yaml_path provided; right IK disabled.");
        }

        // ===== NEW: read logging params =====
        log_enabled_ = this->get_parameter("state_log_enable").as_bool();
        log_dir_ = this->get_parameter("state_log_dir").as_string();
        log_prefix_ = this->get_parameter("state_log_prefix").as_string();
        flush_every_n_ = this->get_parameter("state_log_flush_every_n").as_int();
        if (flush_every_n_ <= 0) flush_every_n_ = 1;
        ik_log_ = this->get_parameter("ik_log").as_bool();
        // ==================================

        if (fps <= 0.0)
        {
            fps = 30.0;
        }
        if (fps > 0.0)
        {
            dt_ = 1.0 / fps;
        }

        // ---------- 初始化 ZMQ 通讯 ----------
        std::string pub_bind_addr = "tcp://*:" + std::to_string(pub_port);
        std::string sub_connect_addr = "tcp://" + robot_ip + ":" + std::to_string(sub_port);

        publisher_.set(zmq::sockopt::sndhwm, 0);
        publisher_.set(zmq::sockopt::immediate, 1);
        publisher_.bind(pub_bind_addr);

        subscriber_.set(zmq::sockopt::subscribe, "State ");
        subscriber_.set(zmq::sockopt::conflate, 1);
        subscriber_.connect(sub_connect_addr);

        RCLCPP_INFO(this->get_logger(),
                    "ZMQ pub bind: %s, sub connect: %s",
                    pub_bind_addr.c_str(), sub_connect_addr.c_str());

        // 打开机器人开关
        std::this_thread::sleep_for(500ms);
        robot_switch(true);
        std::this_thread::sleep_for(500ms);

        // 先拉一次状态，得到初始关节/位姿
        nlohmann::json initial_state = get_robot_state_blocking();
        {
            std::lock_guard<std::mutex> lk(state_mutex_);
            last_state_json_ = initial_state;
        }
        if (initial_state.is_null())
        {
            RCLCPP_WARN(this->get_logger(),
                        "Failed to get initial robot state (timeout or invalid).");
        }
        else
        {
            RCLCPP_INFO(this->get_logger(), "Got initial robot state.");
        }

        // 期望的 JointState 名称顺序：joint1..joint7
        expected_names_.reserve(7);
        for (int i = 0; i < 7; ++i)
        {
            expected_names_.push_back("joint" + std::to_string(i + 1));
        }
        current_cmd_joint_.assign(7, 0.0);

        // ---------- ROS 订阅/发布 ----------
        auto reliable_qos_shallow = common_utils::reliable_qos_shallow();
        auto reliable_qos = common_utils::reliable_qos();

        // 订阅关节目标
        sub_joint_cmd_ = this->create_subscription<sensor_msgs::msg::JointState>(
            joint_sub_topic,
            reliable_qos_shallow,
            std::bind(&JeRobotNode::joint_cmd_callback, this, std::placeholders::_1));
        RCLCPP_INFO(this->get_logger(), "[SUB] joint cmd: %s", joint_sub_topic.c_str());

        // 订阅夹爪指令（模式 + 指令值）
        sub_gripper_cmd_ = this->create_subscription<je_software::msg::EndEffectorCommandLR>(
            gripper_sub_topic_,
            reliable_qos_shallow,
            std::bind(&JeRobotNode::gripper_cmd_callback, this, std::placeholders::_1));
        RCLCPP_INFO(this->get_logger(), "[SUB] gripper cmd: %s", gripper_sub_topic_.c_str());

        // 订阅末端位姿（暂只缓存，不控制）
        sub_end_pose_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            end_pose_topic,
            reliable_qos_shallow,
            std::bind(&JeRobotNode::end_pose_callback, this, std::placeholders::_1));
        RCLCPP_INFO(this->get_logger(), "[SUB] end pose: %s", end_pose_topic.c_str());

        // 订阅 Oculus 控制器位姿（左右手）
        sub_oculus_controllers_ = this->create_subscription<common::msg::OculusControllers>(
            oculus_controllers_topic,
            reliable_qos_shallow,
            std::bind(&JeRobotNode::oculus_controllers_callback, this, std::placeholders::_1));
        RCLCPP_INFO(this->get_logger(), "[SUB] oculus controllers: %s", oculus_controllers_topic.c_str());

        // 订阅 Oculus 初始化关节指令（左右手）
        sub_oculus_init_joint_ = this->create_subscription<common::msg::OculusInitJointState>(
            oculus_init_joint_state_topic,
            reliable_qos_shallow,
            std::bind(&JeRobotNode::oculus_init_joint_state_callback, this, std::placeholders::_1));
        RCLCPP_INFO(this->get_logger(), "[SUB] oculus init joint: %s", oculus_init_joint_state_topic.c_str());

        // 发布关节状态（OculusInitJointState）
        pub_joint_state_ = this->create_publisher<common::msg::OculusInitJointState>(
            joint_pub_topic,
            reliable_qos);
        RCLCPP_INFO(this->get_logger(), "[PUB] joint state: %s", joint_pub_topic.c_str());

        // ===== NEW: open log file once (optional) =====
        if (log_enabled_)
        {
            open_log_file_if_needed();
            if (!last_state_json_.is_null())
            {
                // 记录初始状态（可选）
                write_state_to_disk(last_state_json_, this->get_clock()->now());
            }
        }
        // =============================================

        // 独立线程：循环读取状态并发布（避免阻塞 executor）
        state_thread_running_.store(true);
        state_thread_ = std::thread(&JeRobotNode::state_publish_loop, this);
    }

    ~JeRobotNode() override
    {
        state_thread_running_.store(false);

        // 关键：打断 blocking recv
        try
        {
            context_.shutdown();
        }
        catch (...)
        {
        }

        if (state_thread_.joinable())
        {
            state_thread_.join();
        }

        try
        {
            robot_switch(false);
        }
        catch (...)
        {
        }

        // ===== NEW: close log =====
        if (state_log_.is_open())
        {
            state_log_.flush();
            state_log_.close();
        }
        // =========================
    }

private:
    // ==================== ZMQ 协议封装 ====================

    void robot_switch(bool value)
    {
        nlohmann::json cmd_switch;
        cmd_switch["Switch"] = value;
        std::string payload = "Switch " + cmd_switch.dump();
        publisher_.send(zmq::buffer(payload), zmq::send_flags::none);
    }

    // 修复点：timeout/无效返回 nullptr JSON，确保外层 is_null() 判断正确
    nlohmann::json get_robot_state_blocking()
    {
        zmq::message_t msg;
        try
        {
            auto res = subscriber_.recv(msg, zmq::recv_flags::none);
            if (!res)
            {
                return nlohmann::json(nullptr); // timeout -> null
            }
        }
        catch (const zmq::error_t &e)
        {
            if (e.num() == ETERM)
            {
                return nlohmann::json(nullptr); // context shutdown -> 正常退出
            }
            if (e.num() == EAGAIN)
            {
                return nlohmann::json(nullptr); // timeout -> null
            }
            RCLCPP_ERROR(this->get_logger(), "ZMQ recv error: %s", e.what());
            return nlohmann::json(nullptr);
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "ZMQ recv error: %s", e.what());
            return nlohmann::json(nullptr);
        }

        std::string state(static_cast<char *>(msg.data()), msg.size());
        auto pos = state.find(' ');
        if (pos == std::string::npos)
        {
            RCLCPP_WARN(this->get_logger(),
                        "State message format invalid (no space separator).");
            return nlohmann::json(nullptr);
        }

        std::string topic = state.substr(0, pos);
        if (topic != "State")
        {
            return nlohmann::json(nullptr);
        }

        try
        {
            return nlohmann::json::parse(state.substr(pos + 1));
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "JSON parse error: %s", e.what());
            return nlohmann::json(nullptr);
        }
    }

    void clear_historical_data()
    {
        zmq::message_t msg;
        while (true)
        {
            auto res = subscriber_.recv(msg, zmq::recv_flags::dontwait);
            if (!res)
                break;
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
    }

    const char *robot_key(int robot_index) const
    {
        return (robot_index == 1) ? "Robot1" : "Robot0";
    }

    struct GripperCommand
    {
        int mode{je_software::msg::EndEffectorCommand::MODE_POSITION};
        double position{0.0};
        int preset{0};
        bool received{false};
    };

    const GripperCommand &get_gripper_command(int robot_index) const
    {
        return (robot_index == 1) ? gripper_cmd_right_ : gripper_cmd_left_;
    }

    GripperCommand &get_gripper_command_mutable(int robot_index)
    {
        return (robot_index == 1) ? gripper_cmd_right_ : gripper_cmd_left_;
    }

    void append_gripper_from_command(nlohmann::json &data, int robot_index)
    {
        const auto &gripper_cmd = get_gripper_command(robot_index);
        if (!gripper_cmd.received)
        {
            return;
        }
        nlohmann::json ee;
        ee["mode"] = gripper_cmd.mode;
        if (gripper_cmd.mode == je_software::msg::EndEffectorCommand::MODE_POSITION)
        {
            ee["position"] = gripper_cmd.position;
        }
        else if (gripper_cmd.mode == je_software::msg::EndEffectorCommand::MODE_PRESET)
        {
            ee["preset"] = gripper_cmd.preset;
        }
        data[robot_key(robot_index)]["EndEffector"] = ee;
    }

    void append_gripper_from_position(nlohmann::json &data, int robot_index, double position)
    {
        nlohmann::json ee;
        ee["mode"] = je_software::msg::EndEffectorCommand::MODE_POSITION;
        ee["position"] = position;
        // std::cout << "Here: published postion: " << robot_index << ", " << position << std::endl;
        data[robot_key(robot_index)]["EndEffector"] = ee;
    }

    void append_oculus_init_gripper(nlohmann::json &data, int robot_index, double position)
    {
        if (oculus_init_use_msg_gripper_)
        {
            append_gripper_from_position(data, robot_index, position);
        }
        else
        {
            append_gripper_from_command(data, robot_index);
        }
    }

    void set_robot_joint(const std::vector<double> &joint, int robot_index, double delta_time = 0.0)
    {
        if (std::abs(delta_time - 0) < 1e-5)
        {
            global_time_ += dt_;
        }
        else
        {
            global_time_ += delta_time;
        }
        nlohmann::json data;
        const char *key = robot_key(robot_index);
        data[key]["time"] = global_time_;
        data[key]["joint"] = joint;
        append_gripper_from_command(data, robot_index);

        std::ostringstream oss;
        oss.setf(std::ios::fixed);
        oss << std::setprecision(6) << "[";
        for (std::size_t i = 0; i < joint.size(); ++i)
        {
            if (i != 0)
                oss << ", ";
            oss << joint[i];
        }
        oss << "]";
        RCLCPP_INFO_THROTTLE(
            this->get_logger(), *this->get_clock(), 500,
            "Sending joint cmd (%s): %s", key, oss.str().c_str());

        std::string payload = "Joint " + data.dump();
        publisher_.send(zmq::buffer(payload));
    }

    void set_robot_cartesian(const std::vector<double> &cartesian, int robot_index, double delta_time = 0.0)
    {
        if (std::abs(delta_time - 0) < 1e-5)
        {
            global_time_ += dt_;
        }
        else
        {
            global_time_ += delta_time;
        }
        global_time_ += dt_;
        nlohmann::json data;
        const char *key = robot_key(robot_index);
        data[key]["time"] = global_time_;
        data[key]["cartesian"] = cartesian;
        append_gripper_from_command(data, robot_index);
        publisher_.send(zmq::buffer("Cartesian " + data.dump()));
    }

    // quaternion -> RPY (rad)
    static inline void quat_to_rpy(double qx, double qy, double qz, double qw,
                                   double &roll, double &pitch, double &yaw)
    {
        const double n = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
        if (n > 1e-12)
        {
            qx /= n;
            qy /= n;
            qz /= n;
            qw /= n;
        }
        else
        {
            qx = qy = qz = 0.0;
            qw = 1.0;
        }

        const double sinr_cosp = 2.0 * (qw * qx + qy * qz);
        const double cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy);
        roll = std::atan2(sinr_cosp, cosr_cosp);

        const double sinp = 2.0 * (qw * qy - qz * qx);
        if (std::abs(sinp) >= 1.0)
            pitch = std::copysign(M_PI / 2.0, sinp);
        else
            pitch = std::asin(sinp);

        const double siny_cosp = 2.0 * (qw * qz + qx * qy);
        const double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
        yaw = std::atan2(siny_cosp, cosy_cosp);
    }

    static inline void quat_to_rpy_eigen(const double qx, const double qy, const double qz, const double qw,
                                   double &roll, double &pitch, double &yaw)
    {
        Eigen::Quaterniond q(qw, qx, qy, qz);
        Eigen::Matrix3d R = q.toRotationMatrix();

        // 返回的是绕 Z, Y, X 的角（顺序与参数一致）
        Eigen::Vector3d ypr = R.eulerAngles(2, 1, 0);
        yaw   = ypr[0];
        pitch = ypr[1];
        roll  = ypr[2];
    }


    // ==================== NEW: logging helpers ====================

    static std::string now_string_local_compact()
    {
        using clock = std::chrono::system_clock;
        const auto t = clock::to_time_t(clock::now());
        std::tm tm{};
#if defined(_WIN32)
        localtime_s(&tm, &t);
#else
        localtime_r(&t, &tm);
#endif
        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
        return oss.str();
    }

    void open_log_file_if_needed()
    {
        if (!log_enabled_ || state_log_.is_open()) return;

        std::error_code ec;
        std::filesystem::create_directories(log_dir_, ec);
        if (ec)
        {
            RCLCPP_ERROR(this->get_logger(),
                         "Failed to create log dir '%s': %s",
                         log_dir_.c_str(), ec.message().c_str());
            log_enabled_ = false;
            return;
        }

        log_path_ = (std::filesystem::path(log_dir_) /
                     (log_prefix_ + "_" + now_string_local_compact() + ".jsonl")).string();

        state_log_.open(log_path_, std::ios::out | std::ios::app);
        if (!state_log_)
        {
            RCLCPP_ERROR(this->get_logger(),
                         "Failed to open log file '%s' for writing.", log_path_.c_str());
            log_enabled_ = false;
            return;
        }

        RCLCPP_INFO(this->get_logger(),
                    "State logging enabled: %s", log_path_.c_str());
    }

    void write_state_to_disk(const nlohmann::json &state_json,
                             const rclcpp::Time &stamp)
    {
        if (!log_enabled_) return;
        open_log_file_if_needed();
        if (!state_log_.is_open()) return;

        nlohmann::json out = state_json;
        out["__ros_stamp_ns"] = stamp.nanoseconds();
        out["__ros_stamp_sec"] = stamp.seconds();

        state_log_ << out.dump() << "\n";

        ++log_count_;
        if ((log_count_ % static_cast<size_t>(flush_every_n_)) == 0)
        {
            state_log_.flush();
        }
    }

    // ==================== ROS 回调 ====================

    bool fill_joint_target(const sensor_msgs::msg::JointState &msg,
                           std::vector<double> &target,
                           std::string &missing_csv)
    {
        if (msg.name.empty() || msg.position.empty())
        {
            return false;
        }

        std::unordered_map<std::string, std::size_t> name_to_idx;
        name_to_idx.reserve(msg.name.size());
        for (std::size_t i = 0; i < msg.name.size(); ++i)
        {
            name_to_idx[msg.name[i]] = i;
        }

        std::vector<std::string> missing;
        for (std::size_t i = 0; i < expected_names_.size(); ++i)
        {
            const auto &joint_name = expected_names_[i];
            auto it = name_to_idx.find(joint_name);
            if (it != name_to_idx.end() && it->second < msg.position.size())
            {
                target[i] = msg.position[it->second];
            }
            else
            {
                missing.push_back(joint_name);
            }
        }

        if (!missing.empty())
        {
            for (std::size_t i = 0; i < missing.size(); ++i)
            {
                if (i > 0)
                    missing_csv += ",";
                missing_csv += missing[i];
            }
            return false;
        }

        return true;
    }

    std::vector<double> pose_to_cartesian(const geometry_msgs::msg::Pose &pose)
    {
        const double x = pose.position.x;
        const double y = pose.position.y;
        const double z = pose.position.z;

        const double qx = pose.orientation.x;
        const double qy = pose.orientation.y;
        const double qz = pose.orientation.z;
        const double qw = pose.orientation.w;

        double roll = 0.0, pitch = 0.0, yaw = 0.0;
        quat_to_rpy_eigen(qx, qy, qz, qw, roll, pitch, yaw);

        return {x, y, z, roll, pitch, yaw};
    }

    void joint_cmd_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "joint cmd callback!");
        if (msg->name.empty() || msg->position.empty())
        {
            RCLCPP_WARN(this->get_logger(), "Received empty JointState cmd.");
            return;
        }

        RCLCPP_INFO_THROTTLE(
            this->get_logger(), *this->get_clock(), 1000,
            "Received JointState cmd: names=%zu positions=%zu",
            msg->name.size(), msg->position.size());

        std::vector<double> target(7, 0.0);
        std::string missing_csv;
        if (!fill_joint_target(*msg, target, missing_csv))
        {
            RCLCPP_WARN(this->get_logger(),
                        "Missing joints in JointState cmd: [%s]", missing_csv.c_str());
            return;
        }

        current_cmd_joint_ = target;
        joint_cmd_received_ = true;

        // ===== FIX: guard effort access =====
        const bool is_init =
            (!msg->effort.empty() && std::abs(msg->effort[0] - 666.0) < 1e-5);
        // ===================================

        if (is_init)
        {
            RCLCPP_INFO(this->get_logger(), "JeRobotNode received one initial joint posi.");
            set_robot_joint(current_cmd_joint_, 0, dt_init_);
        }
        else
        {
            set_robot_joint(current_cmd_joint_, 0);
        }
    }

    void handle_gripper_cmd(const je_software::msg::EndEffectorCommand &msg,
                            int robot_index,
                            const char *label)
    {
        if (msg.mode != je_software::msg::EndEffectorCommand::MODE_POSITION &&
            msg.mode != je_software::msg::EndEffectorCommand::MODE_PRESET)
        {
            RCLCPP_WARN(this->get_logger(), "Invalid end effector mode: %d", msg.mode);
            return;
        }

        auto &gripper_cmd = get_gripper_command_mutable(robot_index);
        gripper_cmd.mode = msg.mode;
        gripper_cmd.position = msg.position;
        gripper_cmd.preset = msg.preset;
        gripper_cmd.received = true;

        if (msg.mode == je_software::msg::EndEffectorCommand::MODE_POSITION)
        {
            RCLCPP_INFO_THROTTLE(
                this->get_logger(), *this->get_clock(), 1000,
                "Received end effector cmd (%s): mode=POSITION position=%.3f",
                label, msg.position);
        }
        else
        {
            RCLCPP_INFO_THROTTLE(
                this->get_logger(), *this->get_clock(), 1000,
                "Received end effector cmd (%s): mode=PRESET preset=%d",
                label, msg.preset);
        }
    }

    void gripper_cmd_callback(const je_software::msg::EndEffectorCommandLR::SharedPtr msg)
    {
        if (!msg)
        {
            return;
        }

        if (msg->left_valid)
        {
            handle_gripper_cmd(msg->left, 0, "left");
        }
        if (msg->right_valid)
        {
            handle_gripper_cmd(msg->right, 1, "right");
        }
    }

    void end_pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        if (!msg)
            return;
        latest_ee_pose_ = msg;
        std::vector<double> cartesian = pose_to_cartesian(msg->pose);
        set_robot_cartesian(cartesian, 0);

        static int count = 0;
        if ((++count % 50) == 0)
        {
            const double x = msg->pose.position.x;
            const double y = msg->pose.position.y;
            const double z = msg->pose.position.z;
            const double qx = msg->pose.orientation.x;
            const double qy = msg->pose.orientation.y;
            const double qz = msg->pose.orientation.z;
            const double qw = msg->pose.orientation.w;
            double roll = 0.0, pitch = 0.0, yaw = 0.0;
            quat_to_rpy_eigen(qx, qy, qz, qw, roll, pitch, yaw);
            RCLCPP_INFO(this->get_logger(),
                        "EE pose @ %s: pos(%.3f, %.3f, %.3f) rpy(rad)(%.3f, %.3f, %.3f)",
                        msg->header.frame_id.c_str(), x, y, z, roll, pitch, yaw);
        }
    }

    void oculus_controllers_callback(const common::msg::OculusControllers::SharedPtr msg)
    {
        if (!msg) return;

    bool left_ok = msg->left_valid;
    bool right_ok = msg->right_valid;
        if (!left_ok && !right_ok) return;

        global_time_ += dt_;

        // Local snapshot of last_state_json_
        nlohmann::json state_snapshot;
        {
            std::lock_guard<std::mutex> lk(state_mutex_);
            state_snapshot = last_state_json_;
        }

        // Process left
        if (left_ok) {
            if (ik_solver_left_) {
                Eigen::VectorXd q_init = Eigen::VectorXd::Zero(ik_solver_left_->getNq());
                // try to seed with last reported robot joints if available
                try {
                    if (state_snapshot.contains("Robot0") && state_snapshot["Robot0"].contains("Joint")) {
                        auto arr = state_snapshot["Robot0"]["Joint"];
                        int limit = std::min(static_cast<int>(arr.size()), static_cast<int>(q_init.size()));
                        for (int i = 0; i < limit; ++i) q_init[i] = arr[i].get<double>();
                    }
                } catch(...) {}

                ros2_ik_cpp::IkSolver::Result r;
                try { r = ik_solver_left_->solvePose(msg->left_pose, q_init, ik_left_timeout_ms_); }
                catch (const std::exception &e) { RCLCPP_WARN(this->get_logger(), "IK left threw: %s", e.what()); }

                if (ik_log_) {
                    // compute init FK and initial error
                    auto init_fk = ik_solver_left_->forwardKinematicsSE3(q_init);
                    // compute error between init_fk and target
                    Eigen::Vector3d pos_init = init_fk.translation();
                    auto target = ros2_ik_cpp::IkSolver::makeSE3(msg->left_pose);
                    Eigen::Vector3d pos_tgt = target.translation();
                    Eigen::Vector3d pos_err = pos_tgt - pos_init;
                    Eigen::Quaterniond qinit(init_fk.rotation());
                    Eigen::Quaterniond qtgt(target.rotation());
                    Eigen::Quaterniond qerr = qtgt * qinit.conjugate(); qerr.normalize();
                    Eigen::AngleAxisd aa(qerr); Eigen::Vector3d ang_err = Eigen::Vector3d::Zero();
                    double angle = aa.angle(); if (std::isfinite(angle) && std::abs(angle) > 1e-12) ang_err = aa.axis() * angle;
                    double init_err = (Eigen::Matrix<double,6,1>() << pos_err, ang_err).finished().norm();
                    // stringify poses
                    double r_init=0,p_init=0,y_init=0; quat_to_rpy_eigen(qinit.x(), qinit.y(), qinit.z(), qinit.w(), r_init, p_init, y_init);
                    double r_t=0,p_t=0,y_t=0; quat_to_rpy_eigen(qtgt.x(), qtgt.y(), qtgt.z(), qtgt.w(), r_t, p_t, y_t);
                    std::ostringstream oss_init, oss_tgt;
                    oss_init.setf(std::ios::fixed); oss_init<<std::setprecision(6)
                        <<"pos("<<pos_init.x()<<","<<pos_init.y()<<","<<pos_init.z()<<") rpy("<<r_init<<","<<p_init<<","<<y_init<<")";
                    oss_tgt.setf(std::ios::fixed); oss_tgt<<std::setprecision(6)
                        <<"pos("<<pos_tgt.x()<<","<<pos_tgt.y()<<","<<pos_tgt.z()<<") rpy("<<r_t<<","<<p_t<<","<<y_t<<")";
                    // after solve, print summary
                    RCLCPP_INFO(this->get_logger(), "[IK LEFT] init_err=%.6f init_fk=%s target=%s", init_err, oss_init.str().c_str(), oss_tgt.str().c_str());
                    RCLCPP_INFO(this->get_logger(), "[IK LEFT] result: time_ms=%.3f success=%d iters=%d final_err=%.6f",
                        r.elapsed_ms, (int)r.success, r.iterations, r.final_error);
                    if (r.success && r.q.size()>0) {
                        std::ostringstream oss_q; oss_q<<"["; for (int i=0;i<r.q.size();++i){ if(i) oss_q<<", "; oss_q<<std::fixed<<std::setprecision(6)<<r.q[i]; } oss_q<<"]";
                        RCLCPP_INFO(this->get_logger(), "[IK LEFT] published joints: %s", oss_q.str().c_str());
                    }
                }

                if (r.success && r.q.size() > 0) {
                    std::vector<double> joints(r.q.size());
                    for (int i = 0; i < r.q.size(); ++i) joints[i] = r.q[i];
                    set_robot_joint(joints, 0);
                } else {
                    // fallback: send cartesian
                    left_ok = false;
                    RCLCPP_INFO(this->get_logger(), "No IK solution for left arm. Skipping IK.");
                    // set_robot_cartesian(pose_to_cartesian(msg->left_pose), 0);
                }
            } else {
                left_ok = false;
                RCLCPP_INFO(this->get_logger(), "No IK solver for left arm, skipping IK.");
                // set_robot_cartesian(pose_to_cartesian(msg->left_pose), 0);
            }
        }

        // Process right
        if (right_ok) {
            if (ik_solver_right_) {
                Eigen::VectorXd q_init = Eigen::VectorXd::Zero(ik_solver_right_->getNq());
                try {
                    if (state_snapshot.contains("Robot1") && state_snapshot["Robot1"].contains("Joint")) {
                        auto arr = state_snapshot["Robot1"]["Joint"];
                        int limit = std::min(static_cast<int>(arr.size()), static_cast<int>(q_init.size()));
                        for (int i = 0; i < limit; ++i) q_init[i] = arr[i].get<double>();
                    }
                } catch(...) {}

                ros2_ik_cpp::IkSolver::Result r;
                try { r = ik_solver_right_->solvePose(msg->right_pose, q_init, ik_right_timeout_ms_); }
                catch (const std::exception &e) { RCLCPP_WARN(this->get_logger(), "IK right threw: %s", e.what()); }

                if (ik_log_) {
                    auto init_fk = ik_solver_right_->forwardKinematicsSE3(q_init);
                    Eigen::Vector3d pos_init = init_fk.translation();
                    auto target = ros2_ik_cpp::IkSolver::makeSE3(msg->right_pose);
                    Eigen::Vector3d pos_tgt = target.translation();
                    Eigen::Vector3d pos_err = pos_tgt - pos_init;
                    Eigen::Quaterniond qinit(init_fk.rotation());
                    Eigen::Quaterniond qtgt(target.rotation());
                    Eigen::Quaterniond qerr = qtgt * qinit.conjugate(); qerr.normalize();
                    Eigen::AngleAxisd aa(qerr); Eigen::Vector3d ang_err = Eigen::Vector3d::Zero();
                    double angle = aa.angle(); if (std::isfinite(angle) && std::abs(angle) > 1e-12) ang_err = aa.axis() * angle;
                    double init_err = (Eigen::Matrix<double,6,1>() << pos_err, ang_err).finished().norm();
                    double r_init=0,p_init=0,y_init=0; quat_to_rpy_eigen(qinit.x(), qinit.y(), qinit.z(), qinit.w(), r_init, p_init, y_init);
                    double r_t=0,p_t=0,y_t=0; quat_to_rpy_eigen(qtgt.x(), qtgt.y(), qtgt.z(), qtgt.w(), r_t, p_t, y_t);
                    std::ostringstream oss_init, oss_tgt;
                    oss_init.setf(std::ios::fixed); oss_init<<std::setprecision(6)
                        <<"pos("<<pos_init.x()<<","<<pos_init.y()<<","<<pos_init.z()<<") rpy("<<r_init<<","<<p_init<<","<<y_init<<")";
                    oss_tgt.setf(std::ios::fixed); oss_tgt<<std::setprecision(6)
                        <<"pos("<<pos_tgt.x()<<","<<pos_tgt.y()<<","<<pos_tgt.z()<<") rpy("<<r_t<<","<<p_t<<","<<y_t<<")";
                    RCLCPP_INFO(this->get_logger(), "[IK RIGHT] init_err=%.6f init_fk=%s target=%s", init_err, oss_init.str().c_str(), oss_tgt.str().c_str());
                    RCLCPP_INFO(this->get_logger(), "[IK RIGHT] result: time_ms=%.3f success=%d iters=%d final_err=%.6f",
                        r.elapsed_ms, (int)r.success, r.iterations, r.final_error);
                    if (r.success && r.q.size()>0) {
                        std::ostringstream oss_q; oss_q<<"["; for (int i=0;i<r.q.size();++i){ if(i) oss_q<<", "; oss_q<<std::fixed<<std::setprecision(6)<<r.q[i]; } oss_q<<"]";
                        RCLCPP_INFO(this->get_logger(), "[IK RIGHT] published joints: %s", oss_q.str().c_str());
                    }
                }

                if (r.success && r.q.size() > 0) {
                    std::vector<double> joints(r.q.size());
                    for (int i = 0; i < r.q.size(); ++i) joints[i] = r.q[i];
                    set_robot_joint(joints, 1);
                } else {
                    right_ok = false;
                    RCLCPP_INFO(this->get_logger(), "No IK solver for right arm, skipping IK.");
                    // set_robot_cartesian(pose_to_cartesian(msg->right_pose), 1);
                }
            } else {
                right_ok = false;
                RCLCPP_INFO(this->get_logger(), "No IK solver for right arm, skipping IK.");
                // set_robot_cartesian(pose_to_cartesian(msg->right_pose), 1);
            }
        }
     }

    void oculus_init_joint_state_callback(const common::msg::OculusInitJointState::SharedPtr msg)
    {
        // RCLCPP_INFO(this->get_logger(), "Received one msg.");
        if (!msg)
            return;

        const bool is_init = msg->init;
        if (is_init)
        {
            RCLCPP_INFO(this->get_logger(), "JeRobotNode received one initial joint posi.");
        }
        const double init_dt = is_init ? dt_init_ : 0.0;

        std::vector<double> target_left(7, 0.0);
        std::vector<double> target_right(7, 0.0);
        std::string missing_left;
        std::string missing_right;

        const bool left_ok =
            msg->left_valid && fill_joint_target(msg->left, target_left, missing_left);
        if (msg->left_valid && !left_ok)
        {
            RCLCPP_WARN(this->get_logger(),
                        "Missing joints in Oculus left cmd: [%s]", missing_left.c_str());
        }

        const bool right_ok =
            msg->right_valid && fill_joint_target(msg->right, target_right, missing_right);
        if (msg->right_valid && !right_ok)
        {
            RCLCPP_WARN(this->get_logger(),
                        "Missing joints in Oculus right cmd: [%s]", missing_right.c_str());
        }

        if (!left_ok && !right_ok)
        {
            return;
        }

        if (joint_jump_threshold_ > 0.0)
        {
            if (left_ok)
            {
                check_and_update_joint_jump(
                    "Left",
                    target_left,
                    is_init,
                    prev_target_left_,
                    prev_left_valid_);
            }
            if (right_ok)
            {
                check_and_update_joint_jump(
                    "Right",
                    target_right,
                    is_init,
                    prev_target_right_,
                    prev_right_valid_);
            }
        }

        const double delta_time = init_dt;
        if (std::abs(delta_time - 0) < 1e-5)
        {
            global_time_ += dt_;
        }
        else
        {
            global_time_ += delta_time;
        }
        nlohmann::json data;

        if (left_ok)
        {
            data["Robot0"]["time"] = global_time_;
            data["Robot0"]["joint"] = target_left;
        }

        if (right_ok)
        {
            data["Robot1"]["time"] = global_time_;
            data["Robot1"]["joint"] = target_right;
        }

        if (left_ok)
        {
            append_oculus_init_gripper(data, 0, msg->left_gripper);
        }
        if (right_ok)
        {
            append_oculus_init_gripper(data, 1, msg->right_gripper);
        }
        // std::cout << "Publish data: " << data << std::endl;
        publisher_.send(zmq::buffer("Joint " + data.dump()));
    }

    void publish_state_once()
    {
#if JE_ENABLE_PUBLISH_STATE_FREQ_LOG
        // log frequency
        const rclcpp::Time now = this->get_clock()->now();
        if (pub_state_window_start_.nanoseconds() == 0)
        {
            pub_state_window_start_ = now;
        }
        ++pub_state_window_count_;
        const double elapsed = (now - pub_state_window_start_).seconds();
        if (elapsed >= 1.0)
        {
            const double hz = pub_state_window_count_ / elapsed;
            RCLCPP_INFO(this->get_logger(),
                        "publish_state_once rate: %.2f Hz (%zu calls / %.2fs)",
                        hz, pub_state_window_count_, elapsed);
            pub_state_window_start_ = now;
            pub_state_window_count_ = 0;
        }
#endif

        nlohmann::json state_json = get_robot_state_blocking();
        if (state_json.is_null())
        {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 5000,
                "No valid robot state received (timeout or invalid).");
            return;
        }

        {
            std::lock_guard<std::mutex> lk(state_mutex_);
            last_state_json_ = state_json;
        }

        // ===== NEW: log raw state (with stamp) =====
        const rclcpp::Time stamp = this->get_clock()->now();
        write_state_to_disk(state_json, stamp);
        // ==========================================

        try
        {
            common::msg::OculusInitJointState msg;
            msg.header.stamp = stamp;  // 用同一个 stamp，便于和日志对齐
            msg.init = false;

            auto fill_joint_fields = [&](const nlohmann::json &robot,
                                         sensor_msgs::msg::JointState &out,
                                         const char *label) -> bool
            {
                if (!robot.contains("Joint"))
                {
                    return false;
                }
                auto joint_vec = robot["Joint"].get<std::vector<double>>();
                if (joint_vec.size() < 7)
                {
                    RCLCPP_WARN(this->get_logger(),
                                "%s joint vector size %zu < 7", label, joint_vec.size());
                    return false;
                }

                out.name = expected_names_;
                out.position = joint_vec;

                if (robot.contains("JointVelocity"))
                {
                    auto vel_vec = robot["JointVelocity"].get<std::vector<double>>();
                    if (vel_vec.size() >= 7)
                    {
                        out.velocity = vel_vec;
                    }
                }

                if (robot.contains("JointSensorTorque"))
                {
                    auto eff_vec = robot["JointSensorTorque"].get<std::vector<double>>();
                    if (eff_vec.size() >= 7)
                    {
                        out.effort = eff_vec;
                    }
                }

                return true;
            };

            auto fill_gripper_field = [&](const nlohmann::json &robot,
                                          const char *label,
                                          float &out) -> bool
            {
                if (!robot.contains("EndEffector") || !robot["EndEffector"].is_object())
                {
                    return false;
                }
                const auto &ee = robot["EndEffector"];
                if (ee.contains("Valid"))
                {
                    const bool valid = ee["Valid"].get<bool>();
                    if (!valid)
                    {
                        return false;
                    }
                }
                if (!ee.contains("CurrentPosition"))
                {
                    return false;
                }
                try
                {
                    out = static_cast<float>(ee["CurrentPosition"].get<double>());
                    return true;
                }
                catch (const std::exception &e)
                {
                    RCLCPP_WARN(this->get_logger(),
                                "%s EndEffector CurrentPosition parse failed: %s",
                                label, e.what());
                    return false;
                }
            };

            if (state_json.contains("Robot0") && state_json["Robot0"].is_object())
            {
                if (fill_joint_fields(state_json["Robot0"], msg.left, "Robot0"))
                {
                    msg.left_valid = true;
                }
                fill_gripper_field(state_json["Robot0"], "Robot0", msg.left_gripper);
            }

            if (state_json.contains("Robot1") && state_json["Robot1"].is_object())
            {
                if (fill_joint_fields(state_json["Robot1"], msg.right, "Robot1"))
                {
                    msg.right_valid = true;
                }
                fill_gripper_field(state_json["Robot1"], "Robot1", msg.right_gripper);
            }

            if (!msg.left_valid && !msg.right_valid)
            {
                RCLCPP_WARN(this->get_logger(),
                            "No valid joint vectors found in robot state.");
                return;
            }

            pub_joint_state_->publish(msg);
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(this->get_logger(),
                         "Error parsing robot state json: %s", e.what());
        }
    }

    void state_publish_loop()
    {
        while (rclcpp::ok() && state_thread_running_.load())
        {
            publish_state_once();
            if (!state_thread_running_.load())
                break;
        }
    }

private:
    void check_and_update_joint_jump(const char *label,
                                     const std::vector<double> &current,
                                     bool is_init,
                                     std::vector<double> &prev,
                                     bool &prev_valid)
    {
        if (prev_valid && !is_init)
        {
            const size_t n = std::min(current.size(), prev.size());
            size_t max_idx = 0;
            double max_delta = 0.0;
            for (size_t i = 0; i < n; ++i)
            {
                const double d = std::abs(current[i] - prev[i]);
                if (d > max_delta)
                {
                    max_delta = d;
                    max_idx = i;
                }
            }
            if (max_delta > joint_jump_threshold_)
            {
                RCLCPP_ERROR(this->get_logger(),
                             "%s target jump detected at joint%zu: prev=%.6f curr=%.6f delta=%.6f threshold=%.6f. Shutting down.",
                             label,
                             max_idx + 1,
                             prev[max_idx],
                             current[max_idx],
                             max_delta,
                             joint_jump_threshold_);
                rclcpp::shutdown();
                std::exit(1);
            }
        }
        prev = current;
        prev_valid = true;
    }

    void check_and_update_pose_jump(const char *label,
                                    const std::vector<double> &current,
                                    std::vector<double> &prev,
                                    bool &prev_valid)
    {
        if (prev_valid)
        {
            size_t max_pos_idx = 0;
            double max_pos_delta = 0.0;
            for (size_t i = 0; i < 3; ++i)
            {
                const double d = std::abs(current[i] - prev[i]);
                if (d > max_pos_delta)
                {
                    max_pos_delta = d;
                    max_pos_idx = i;
                }
            }

            size_t max_rpy_idx = 3;
            double max_rpy_delta = 0.0;
            for (size_t i = 3; i < 6; ++i)
            {
                const double d = std::abs(current[i] - prev[i]);
                if (d > max_rpy_delta)
                {
                    max_rpy_delta = d;
                    max_rpy_idx = i;
                }
            }

            if (pose_jump_threshold_pos_ > 0.0 && max_pos_delta > pose_jump_threshold_pos_)
            {
                RCLCPP_ERROR(this->get_logger(),
                             "%s pose position jump at idx%zu: prev=%.6f curr=%.6f delta=%.6f threshold=%.6f. Shutting down.",
                             label,
                             max_pos_idx,
                             prev[max_pos_idx],
                             current[max_pos_idx],
                             max_pos_delta,
                             pose_jump_threshold_pos_);
                rclcpp::shutdown();
                std::exit(1);
            }

            if (pose_jump_threshold_rpy_ > 0.0 && max_rpy_delta > pose_jump_threshold_rpy_)
            {
                RCLCPP_ERROR(this->get_logger(),
                             "%s pose rpy jump at idx%zu: prev=%.6f curr=%.6f delta=%.6f threshold=%.6f. Shutting down.",
                             label,
                             max_rpy_idx,
                             prev[max_rpy_idx],
                             current[max_rpy_idx],
                             max_rpy_delta,
                             pose_jump_threshold_rpy_);
                rclcpp::shutdown();
                std::exit(1);
            }
        }
        prev = current;
        prev_valid = true;
    }

    // global time
    double global_time_ = 0.0;
    double dt_ = 0.014;   // 72hz
    double dt_init_ = 5;  // 从任意位置运动到初始位置期望的时间

    // ROS
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_joint_cmd_;
    rclcpp::Subscription<je_software::msg::EndEffectorCommandLR>::SharedPtr sub_gripper_cmd_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_end_pose_;
    rclcpp::Subscription<common::msg::OculusControllers>::SharedPtr sub_oculus_controllers_;
    rclcpp::Subscription<common::msg::OculusInitJointState>::SharedPtr sub_oculus_init_joint_;
    rclcpp::Publisher<common::msg::OculusInitJointState>::SharedPtr pub_joint_state_;

    std::vector<std::string> expected_names_;
    std::vector<double> current_cmd_joint_;
    bool joint_cmd_received_;
    geometry_msgs::msg::PoseStamped::SharedPtr latest_ee_pose_;

#if JE_ENABLE_PUBLISH_STATE_FREQ_LOG
    rclcpp::Time pub_state_window_start_{0, 0, RCL_SYSTEM_TIME};
    size_t pub_state_window_count_{0};
#endif

    // ZMQ
    zmq::context_t context_;
    zmq::socket_t publisher_;
    zmq::socket_t subscriber_;
    std::mutex state_mutex_;
    nlohmann::json last_state_json_;

    // End-effector command cache from ROS2
    std::string gripper_sub_topic_{"/end_effector_cmd_lr"};
    GripperCommand gripper_cmd_left_{};
    GripperCommand gripper_cmd_right_{};
    bool oculus_init_use_msg_gripper_{false};

    // 参数
    double publish_period_;
    double joint_jump_threshold_{0.0};
    double pose_jump_threshold_pos_{0.0};
    double pose_jump_threshold_rpy_{0.0};

    std::vector<double> prev_target_left_;
    std::vector<double> prev_target_right_;
    bool prev_left_valid_{false};
    bool prev_right_valid_{false};

    std::vector<double> prev_cart_left_;
    std::vector<double> prev_cart_right_;
    bool prev_left_pose_valid_{false};
    bool prev_right_pose_valid_{false};

    // 线程
    std::atomic_bool state_thread_running_;
    std::thread state_thread_;

    // ===== NEW: logging members =====
    bool log_enabled_{false};
    std::string log_dir_{"/tmp/je_robot_logs"};
    std::string log_prefix_{"robot_state"};
    std::string log_path_;
    int flush_every_n_{10};
    std::ofstream state_log_;
    size_t log_count_{0};
    bool ik_log_{false};
    // ==============================
    std::unique_ptr<ros2_ik_cpp::IkSolver> ik_solver_left_;
    std::unique_ptr<ros2_ik_cpp::IkSolver> ik_solver_right_;

    // ===== NEW: IK timeout parameters =====
    int ik_left_timeout_ms_{100};
    int ik_right_timeout_ms_{100};
    // ====================================
};

int main(int argc, char *argv[])
{
    auto context = std::make_shared<rclcpp::Context>();
    context->init(argc, argv);
    rclcpp::NodeOptions options;
    options.context(context);
    auto node = std::make_shared<JeRobotNode>(options);
    rclcpp::spin(node);
    context->shutdown("main shutdown");
    return 0;
}
