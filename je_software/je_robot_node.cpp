#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <unordered_map>
#include <array>
#include <iostream>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <iomanip>
#include <sstream>

// ===== NEW: logging =====
#include <fstream>
#include <filesystem>
// ========================

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "je_software/msg/end_effector_command.hpp"
#include "common/msg/oculus_controllers.hpp"
#include "common/msg/oculus_init_joint_state.hpp"

#include <zmq.hpp>
#include "nlohmann/json.hpp"
#include <Eigen/Dense>

using namespace std::chrono_literals;

using json = nlohmann::json; // 默认 std::map

class JeRobotNode : public rclcpp::Node
{
public:
    JeRobotNode()
        : Node("je_robot_node"),
          context_(1),
          publisher_(context_, zmq::socket_type::pub),
          subscriber_(context_, zmq::socket_type::sub),
          joint_cmd_received_(false),
          state_thread_running_(false)
    {
        // ---------- 声明 & 获取参数 ----------
        this->declare_parameter<std::string>("joint_sub_topic", "/joint_cmd");
        this->declare_parameter<std::string>("end_pose_topic", "/end_pose");
        this->declare_parameter<std::string>("joint_pub_topic", "/joint_states");
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

        // Gripper ROS2 command topic
        this->declare_parameter<std::string>("gripper_sub_topic", "/end_effector_cmd");

        // ===== NEW: state logging parameters =====
        this->declare_parameter<bool>("state_log_enable", true);
        this->declare_parameter<std::string>("state_log_dir", "./je_robot_logs");
        this->declare_parameter<std::string>("state_log_prefix", "robot_state");
        this->declare_parameter<int>("state_log_flush_every_n", 10);
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
        double fps = this->get_parameter("fps").as_double();

        std::string robot_ip = this->get_parameter("robot_ip").as_string();
        int pub_port = this->get_parameter("pub_port").as_int();
        int sub_port = this->get_parameter("sub_port").as_int();
        dt_ = this->get_parameter("dt").as_double();
        dt_init_ = this->get_parameter("dt_init").as_double();

        gripper_sub_topic_ = gripper_sub_topic;

        // ===== NEW: read logging params =====
        log_enabled_ = this->get_parameter("state_log_enable").as_bool();
        log_dir_ = this->get_parameter("state_log_dir").as_string();
        log_prefix_ = this->get_parameter("state_log_prefix").as_string();
        flush_every_n_ = this->get_parameter("state_log_flush_every_n").as_int();
        if (flush_every_n_ <= 0) flush_every_n_ = 1;
        // ==================================

        if (fps <= 0.0)
        {
            fps = 50.0;
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
        last_state_json_ = get_robot_state_blocking();
        if (last_state_json_.is_null())
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
        auto qos = rclcpp::QoS(rclcpp::KeepLast(1));
        qos.reliable();

        // 订阅关节目标
        sub_joint_cmd_ = this->create_subscription<sensor_msgs::msg::JointState>(
            joint_sub_topic,
            qos,
            std::bind(&JeRobotNode::joint_cmd_callback, this, std::placeholders::_1));
        RCLCPP_INFO(this->get_logger(), "[SUB] joint cmd: %s", joint_sub_topic.c_str());

        // 订阅夹爪指令（模式 + 指令值）
        sub_gripper_cmd_ = this->create_subscription<je_software::msg::EndEffectorCommand>(
            gripper_sub_topic_,
            qos,
            std::bind(&JeRobotNode::gripper_cmd_callback, this, std::placeholders::_1));
        RCLCPP_INFO(this->get_logger(), "[SUB] gripper cmd: %s", gripper_sub_topic_.c_str());

        // 订阅末端位姿（暂只缓存，不控制）
        sub_end_pose_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            end_pose_topic,
            qos,
            std::bind(&JeRobotNode::end_pose_callback, this, std::placeholders::_1));
        RCLCPP_INFO(this->get_logger(), "[SUB] end pose: %s", end_pose_topic.c_str());

        // 订阅 Oculus 控制器位姿（左右手）
        sub_oculus_controllers_ = this->create_subscription<common::msg::OculusControllers>(
            oculus_controllers_topic,
            qos,
            std::bind(&JeRobotNode::oculus_controllers_callback, this, std::placeholders::_1));
        RCLCPP_INFO(this->get_logger(), "[SUB] oculus controllers: %s", oculus_controllers_topic.c_str());

        // 订阅 Oculus 初始化关节指令（左右手）
        sub_oculus_init_joint_ = this->create_subscription<common::msg::OculusInitJointState>(
            oculus_init_joint_state_topic,
            qos,
            std::bind(&JeRobotNode::oculus_init_joint_state_callback, this, std::placeholders::_1));
        RCLCPP_INFO(this->get_logger(), "[SUB] oculus init joint: %s", oculus_init_joint_state_topic.c_str());

        // 发布关节状态
        pub_joint_state_ = this->create_publisher<sensor_msgs::msg::JointState>(
            joint_pub_topic,
            qos);
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
        if (gripper_cmd_received_)
        {
            nlohmann::json ee;
            ee["mode"] = gripper_cmd_mode_;
            if (gripper_cmd_mode_ == je_software::msg::EndEffectorCommand::MODE_POSITION)
            {
                ee["position"] = gripper_cmd_position_;
            }
            else if (gripper_cmd_mode_ == je_software::msg::EndEffectorCommand::MODE_PRESET)
            {
                ee["preset"] = gripper_cmd_preset_;
            }
            data[key]["end_effector"] = ee;
        }

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
        if (gripper_cmd_received_)
        {
            nlohmann::json ee;
            ee["mode"] = gripper_cmd_mode_;
            if (gripper_cmd_mode_ == je_software::msg::EndEffectorCommand::MODE_POSITION)
            {
                ee["position"] = gripper_cmd_position_;
            }
            else if (gripper_cmd_mode_ == je_software::msg::EndEffectorCommand::MODE_PRESET)
            {
                ee["preset"] = gripper_cmd_preset_;
            }
            data[key]["end_effector"] = ee;
        }
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

    void gripper_cmd_callback(const je_software::msg::EndEffectorCommand::SharedPtr msg)
    {
        if (!msg)
        {
            return;
        }

        if (msg->mode != je_software::msg::EndEffectorCommand::MODE_POSITION &&
            msg->mode != je_software::msg::EndEffectorCommand::MODE_PRESET)
        {
            RCLCPP_WARN(this->get_logger(), "Invalid end effector mode: %d", msg->mode);
            return;
        }

        gripper_cmd_mode_ = msg->mode;
        gripper_cmd_position_ = msg->position;
        gripper_cmd_preset_ = msg->preset;
        gripper_cmd_received_ = true;

        if (msg->mode == je_software::msg::EndEffectorCommand::MODE_POSITION)
        {
            RCLCPP_INFO_THROTTLE(
                this->get_logger(), *this->get_clock(), 1000,
                "Received end effector cmd: mode=POSITION position=%.3f",
                msg->position);
        }
        else
        {
            RCLCPP_INFO_THROTTLE(
                this->get_logger(), *this->get_clock(), 1000,
                "Received end effector cmd: mode=PRESET preset=%d",
                msg->preset);
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
        if (!msg)
            return;

        const bool left_ok = msg->left_valid;
        const bool right_ok = msg->right_valid;
        if (!left_ok && !right_ok)
        {
            return;
        }

        global_time_ += dt_;
        nlohmann::json data;

        if (left_ok)
        {
            data["Robot0"]["time"] = global_time_;
            data["Robot0"]["cartesian"] = pose_to_cartesian(msg->left_pose);
        }

        if (right_ok)
        {
            data["Robot1"]["time"] = global_time_;
            data["Robot1"]["cartesian"] = pose_to_cartesian(msg->right_pose);
        }

        if (gripper_cmd_received_)
        {
            nlohmann::json ee;
            ee["mode"] = gripper_cmd_mode_;
            if (gripper_cmd_mode_ == je_software::msg::EndEffectorCommand::MODE_POSITION)
            {
                ee["position"] = gripper_cmd_position_;
            }
            else if (gripper_cmd_mode_ == je_software::msg::EndEffectorCommand::MODE_PRESET)
            {
                ee["preset"] = gripper_cmd_preset_;
            }
            if (left_ok)
            {
                data["Robot0"]["end_effector"] = ee;
            }
            if (right_ok)
            {
                data["Robot1"]["end_effector"] = ee;
            }
        }

        publisher_.send(zmq::buffer("Cartesian " + data.dump()));
    }

    void oculus_init_joint_state_callback(const common::msg::OculusInitJointState::SharedPtr msg)
    {
        if (!msg)
            return;

        const bool is_init = msg->init;
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

        if (gripper_cmd_received_)
        {
            nlohmann::json ee;
            ee["mode"] = gripper_cmd_mode_;
            if (gripper_cmd_mode_ == je_software::msg::EndEffectorCommand::MODE_POSITION)
            {
                ee["position"] = gripper_cmd_position_;
            }
            else if (gripper_cmd_mode_ == je_software::msg::EndEffectorCommand::MODE_PRESET)
            {
                ee["preset"] = gripper_cmd_preset_;
            }
            if (left_ok)
            {
                data["Robot0"]["end_effector"] = ee;
            }
            if (right_ok)
            {
                data["Robot1"]["end_effector"] = ee;
            }
        }

        publisher_.send(zmq::buffer("Joint " + data.dump()));
    }

    void publish_state_once()
    {
        nlohmann::json state_json = get_robot_state_blocking();
        if (state_json.is_null())
        {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 5000,
                "No valid robot state received (timeout or invalid).");
            return;
        }

        // ===== NEW: log raw state (with stamp) =====
        const rclcpp::Time stamp = this->get_clock()->now();
        write_state_to_disk(state_json, stamp);
        // ==========================================

        try
        {
            auto joint_vec = state_json["Robot0"]["Joint"].get<std::vector<double>>();
            if (joint_vec.size() < 7)
            {
                RCLCPP_WARN(this->get_logger(),
                            "Joint vector size %zu < 7", joint_vec.size());
                return;
            }

            sensor_msgs::msg::JointState msg;
            msg.header.stamp = stamp;  // 用同一个 stamp，便于和日志对齐
            msg.name = expected_names_;
            msg.position = joint_vec;

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
    // global time
    double global_time_ = 0.0;
    double dt_ = 0.014;   // 72hz
    double dt_init_ = 5;  // 从任意位置运动到初始位置期望的时间

    // ROS
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_joint_cmd_;
    rclcpp::Subscription<je_software::msg::EndEffectorCommand>::SharedPtr sub_gripper_cmd_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_end_pose_;
    rclcpp::Subscription<common::msg::OculusControllers>::SharedPtr sub_oculus_controllers_;
    rclcpp::Subscription<common::msg::OculusInitJointState>::SharedPtr sub_oculus_init_joint_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_joint_state_;

    std::vector<std::string> expected_names_;
    std::vector<double> current_cmd_joint_;
    bool joint_cmd_received_;
    geometry_msgs::msg::PoseStamped::SharedPtr latest_ee_pose_;

    // ZMQ
    zmq::context_t context_;
    zmq::socket_t publisher_;
    zmq::socket_t subscriber_;
    nlohmann::json last_state_json_;

    // End-effector command cache from ROS2
    std::string gripper_sub_topic_{"/end_effector_cmd"};
    int gripper_cmd_mode_{je_software::msg::EndEffectorCommand::MODE_POSITION};
    double gripper_cmd_position_{0.0};
    int gripper_cmd_preset_{0};
    bool gripper_cmd_received_{false};

    // 参数
    double publish_period_;

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
    // ==============================
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<JeRobotNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
