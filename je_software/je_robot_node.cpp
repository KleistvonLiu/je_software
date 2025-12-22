#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <iomanip>
#include <sstream>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"

#include <zmq.hpp>
#include "nlohmann/json.hpp"

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
        this->declare_parameter<double>("fps", 50.0);

        // ZMQ 相关参数
        this->declare_parameter<std::string>("robot_ip", "192.168.0.99");
        this->declare_parameter<int>("pub_port", 8001);
        this->declare_parameter<int>("sub_port", 8000);

        // 下发关节指令时的插补时间（秒）
        this->declare_parameter<double>("dt", 0.014);
        this->declare_parameter<double>("dt_init", 5.0);

        std::string joint_sub_topic =
            this->get_parameter("joint_sub_topic").as_string();
        std::string end_pose_topic =
            this->get_parameter("end_pose_topic").as_string();
        std::string joint_pub_topic =
            this->get_parameter("joint_pub_topic").as_string();
        double fps = this->get_parameter("fps").as_double();

        std::string robot_ip = this->get_parameter("robot_ip").as_string();
        int pub_port = this->get_parameter("pub_port").as_int();
        int sub_port = this->get_parameter("sub_port").as_int();
        dt_ = this->get_parameter("dt").as_double();
        dt_init_ = this->get_parameter("dt_init").as_double();

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
        auto qos = rclcpp::QoS(rclcpp::KeepLast(10));
        qos.reliable();

        // 订阅关节目标
        sub_joint_cmd_ = this->create_subscription<sensor_msgs::msg::JointState>(
            joint_sub_topic,
            qos,
            std::bind(&JeRobotNode::joint_cmd_callback, this, std::placeholders::_1));
        RCLCPP_INFO(this->get_logger(), "[SUB] joint cmd: %s", joint_sub_topic.c_str());

        // 订阅末端位姿（暂只缓存，不控制）
        sub_end_pose_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            end_pose_topic,
            qos,
            std::bind(&JeRobotNode::end_pose_callback, this, std::placeholders::_1));
        RCLCPP_INFO(this->get_logger(), "[SUB] end pose: %s", end_pose_topic.c_str());

        // 发布关节状态
        pub_joint_state_ = this->create_publisher<sensor_msgs::msg::JointState>(
            joint_pub_topic,
            qos);
        RCLCPP_INFO(this->get_logger(), "[PUB] joint state: %s", joint_pub_topic.c_str());

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

    void set_robot_joint(const std::vector<double> &joint, double delta_time = 0.0)
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
        data["Robot0"]["time"] = global_time_;
        data["Robot0"]["joint"] = joint;

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
            "Sending joint cmd: %s", oss.str().c_str());

        std::string payload = "Joint " + data.dump();
        publisher_.send(zmq::buffer(payload), zmq::send_flags::none);
    }

    void set_robot_cartesian(const std::vector<double> &cartesian, double delta_time = 0.0)
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
        data["Robot0"]["time"] = global_time_;
        data["Robot0"]["cartesian"] = cartesian;
        publisher_.send(zmq::buffer("Cartesian " + data.dump()));
    }

    // quaternion -> RPY (rad)
    static inline void quat_to_rpy(double qx, double qy, double qz, double qw,
                                   double &roll, double &pitch, double &yaw)
    {
        // 可选：归一化，防止数值漂移
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

    // ==================== ROS 回调 ====================

    void joint_cmd_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        // std::cout << "joint cmd callback" << std::endl;
        if (msg->name.empty() || msg->position.empty())
        {
            RCLCPP_WARN(this->get_logger(), "Received empty JointState cmd.");
            return;
        }

        // name -> index 映射
        std::unordered_map<std::string, std::size_t> name_to_idx;
        name_to_idx.reserve(msg->name.size());
        for (std::size_t i = 0; i < msg->name.size(); ++i)
        {
            name_to_idx[msg->name[i]] = i;
        }

        std::vector<double> target(7, 0.0);
        std::vector<std::string> missing;
        for (std::size_t i = 0; i < expected_names_.size(); ++i)
        {
            const auto &joint_name = expected_names_[i];
            auto it = name_to_idx.find(joint_name);
            if (it != name_to_idx.end() && it->second < msg->position.size())
            {
                target[i] = msg->position[it->second];
            }
            else
            {
                missing.push_back(joint_name);
            }
        }

        if (!missing.empty())
        {
            std::string s;
            for (std::size_t i = 0; i < missing.size(); ++i)
            {
                if (i > 0)
                    s += ",";
                s += missing[i];
            }
            RCLCPP_WARN(this->get_logger(),
                        "Missing joints in JointState cmd: [%s]", s.c_str());
            return;
        }

        current_cmd_joint_ = target;
        joint_cmd_received_ = true;
        if (std::abs(msg->effort[0] -  666) < 1e-5) {
            // RCLCPP_INFO(this->get_logger(), "JeRobotNode received one initial joint posi.");
            std::cout << "JeRobotNode received one initial joint posi." << std::endl;
            set_robot_joint(current_cmd_joint_, dt_init_);
        } else {
            set_robot_joint(current_cmd_joint_);
        }
    }

    void end_pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        // std::cout << "endpose cmd callback" << std::endl;
        if (!msg)
            return;
        latest_ee_pose_ = msg;

        // 位置
        const double x = msg->pose.position.x;
        const double y = msg->pose.position.y;
        const double z = msg->pose.position.z;

        // 四元数
        const double qx = msg->pose.orientation.x;
        const double qy = msg->pose.orientation.y;
        const double qz = msg->pose.orientation.z;
        const double qw = msg->pose.orientation.w;

        // 转 RPY（弧度）
        double roll = 0.0, pitch = 0.0, yaw = 0.0;
        quat_to_rpy(qx, qy, qz, qw, roll, pitch, yaw);

        // set_robot_cartesian 需要的格式：6D [x, y, z, roll, pitch, yaw]
        std::vector<double> cartesian = {x, y, z, roll, pitch, yaw};

        // 下发：time 用你的插补时间参数
        set_robot_cartesian(cartesian);

        // 低频日志
        static int count = 0;
        if ((++count % 50) == 0)
        {
            RCLCPP_INFO(this->get_logger(),
                        "EE pose @ %s: pos(%.3f, %.3f, %.3f) rpy(rad)(%.3f, %.3f, %.3f)",
                        msg->header.frame_id.c_str(), x, y, z, roll, pitch, yaw);
        }
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

        try
        {
            // 建议：这里也可加 contains() 判定进一步增强鲁棒性
            auto joint_vec = state_json["Robot0"]["Joint"].get<std::vector<double>>();
            if (joint_vec.size() < 7)
            {
                RCLCPP_WARN(this->get_logger(),
                            "Joint vector size %zu < 7", joint_vec.size());
                return;
            }

            sensor_msgs::msg::JointState msg;
            msg.header.stamp = this->get_clock()->now();
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
    double dt_ = 0.014; // 72hz
    double dt_init_ = 5; // 从任意位置运动到初始位置期望的时间
    // ROS
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_joint_cmd_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_end_pose_;
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

    // 参数
    double publish_period_;

    // 线程
    std::atomic_bool state_thread_running_;
    std::thread state_thread_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<JeRobotNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
