/**
0: 停止 6 电机
1..6: 停止单电机 (Motor1..Motor6)
10: 二指 0%
11: 二指 71%
20: 三指 0%
21: 三指 88%
*/
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>

#include <poll.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>
#include <cctype>

#include "je_software/msg/end_effector_command.hpp"
#include "je_software/msg/end_effector_command_lr.hpp"
#include "common/msg/oculus_controllers.hpp"

using namespace std::chrono_literals;

class EndEffectorCli : public rclcpp::Node
{
public:
    EndEffectorCli()
        : Node("end_effector_cli")
    {
        this->declare_parameter<std::string>("end_effector_topic", "/end_effector_cmd_lr");
        this->declare_parameter<std::string>("hand", "both");
        this->declare_parameter<std::string>("oculus_topic", "/oculus_controllers");
        this->declare_parameter<std::string>("pose_topic", "/end_pose");
        this->declare_parameter<std::string>("frame_id", "base_link");
        this->declare_parameter<bool>("send_init_pose_on_start", false);
        this->declare_parameter<bool>("attach_init_pose_to_cmd", false);
        this->declare_parameter<std::string>(
            "init_pose",
            "[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]");

        end_effector_topic_ = this->get_parameter("end_effector_topic").as_string();
        default_hand_ = parse_hand_param(this->get_parameter("hand").as_string());
        const std::string oculus_topic = this->get_parameter("oculus_topic").as_string();
        const std::string pose_topic = this->get_parameter("pose_topic").as_string();
        oculus_topic_ = oculus_topic;
        if (oculus_topic == "/oculus_controllers" && pose_topic != "/end_pose")
        {
            oculus_topic_ = pose_topic;
        }
        frame_id_ = this->get_parameter("frame_id").as_string();
        attach_init_pose_ = this->get_parameter("attach_init_pose_to_cmd").as_bool();
        init_pose_ = parse_init_pose_param(this->get_parameter("init_pose"));

        auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
        pub_cmd_ = this->create_publisher<je_software::msg::EndEffectorCommandLR>(end_effector_topic_, qos);
        pub_oculus_ = this->create_publisher<common::msg::OculusControllers>(oculus_topic_, qos);

        RCLCPP_INFO(this->get_logger(), "[PUB] end effector cmd: %s", end_effector_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "[PUB] oculus controllers (init pose): %s (frame_id=%s)",
                    oculus_topic_.c_str(), frame_id_.c_str());
        RCLCPP_INFO(this->get_logger(), "[PUB] oculus controllers mode: %s", hand_to_string(default_hand_));

        if (this->get_parameter("send_init_pose_on_start").as_bool())
        {
            publish_init_pose(default_hand_);
        }

        running_.store(true);
        input_thread_ = std::thread(&EndEffectorCli::input_loop, this);

        print_help();
    }

    ~EndEffectorCli() override
    {
        running_.store(false);
        if (input_thread_.joinable())
        {
            input_thread_.join();
        }
    }

private:
    enum class Hand
    {
        Left,
        Right,
        Both
    };

    static std::string to_lower(std::string s)
    {
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return s;
    }

    static const char *hand_to_string(Hand hand)
    {
        switch (hand)
        {
        case Hand::Left:
            return "left";
        case Hand::Right:
            return "right";
        case Hand::Both:
            return "both";
        default:
            return "unknown";
        }
    }

    static bool is_hand_token(const std::string &token)
    {
        const std::string t = to_lower(token);
        return (t == "l" || t == "left" || t == "r" || t == "right" || t == "b" || t == "both");
    }

    static Hand parse_hand_token(const std::string &token)
    {
        const std::string t = to_lower(token);
        if (t == "r" || t == "right")
        {
            return Hand::Right;
        }
        if (t == "b" || t == "both")
        {
            return Hand::Both;
        }
        return Hand::Left;
    }

    static Hand parse_hand_param(const std::string &hand)
    {
        return parse_hand_token(hand);
    }

    static std::array<double, 6> parse_init_pose(const std::vector<double> &vals)
    {
        std::array<double, 6> pose{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (size_t i = 0; i < pose.size() && i < vals.size(); ++i)
        {
            pose[i] = vals[i];
        }
        return pose;
    }

    static std::array<double, 6> parse_init_pose_param(const rclcpp::Parameter &param)
    {
        if (param.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE_ARRAY)
        {
            return parse_init_pose(param.as_double_array());
        }

        std::string s;
        if (param.get_type() == rclcpp::ParameterType::PARAMETER_STRING)
        {
            s = param.as_string();
        }
        else
        {
            return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        }

        for (char &c : s)
        {
            if (c == ',' || c == '[' || c == ']')
            {
                c = ' ';
            }
        }
        std::istringstream iss(s);
        std::vector<double> vals;
        double v;
        while (iss >> v)
        {
            vals.push_back(v);
        }
        return parse_init_pose(vals);
    }

    static void rpy_to_quat(double roll, double pitch, double yaw,
                            double &qx, double &qy, double &qz, double &qw)
    {
        const double cy = std::cos(yaw * 0.5);
        const double sy = std::sin(yaw * 0.5);
        const double cp = std::cos(pitch * 0.5);
        const double sp = std::sin(pitch * 0.5);
        const double cr = std::cos(roll * 0.5);
        const double sr = std::sin(roll * 0.5);

        qw = cr * cp * cy + sr * sp * sy;
        qx = sr * cp * cy - cr * sp * sy;
        qy = cr * sp * cy + sr * cp * sy;
        qz = cr * cp * sy - sr * sp * cy;
    }

    static bool poll_stdin(int timeout_ms)
    {
        struct pollfd pfd;
        pfd.fd = STDIN_FILENO;
        pfd.events = POLLIN;
        const int ret = ::poll(&pfd, 1, timeout_ms);
        return (ret > 0) && (pfd.revents & POLLIN);
    }

    void print_help()
    {
        std::cout << "\n=== End Effector CLI ===\n"
                  << "Commands:\n"
                  << "  [l|r|b] p <value>    send MODE_POSITION with position\n"
                  << "  [l|r|b] m <preset>   send MODE_PRESET with preset int\n"
                  << "  pose <x y z r p y>   set init pose (rpy in radians)\n"
                  << "  [l|r|b] send_pose   publish init pose once\n"
                  << "  pose_on / pose_off   toggle attach init pose to each cmd\n"
                  << "  help\n"
                  << "  quit / exit\n"
                  << "Examples:\n"
                  << "  p 0.5\n"
                  << "  r p 0.5\n"
                  << "  b m 2\n"
                  << "  pose 0 0 0 0 0 0\n"
                  << "  pose_on\n"
                  << "========================\n\n";
    }

    static geometry_msgs::msg::Pose make_pose_from_rpy(const std::array<double, 6> &pose_rpy)
    {
        double qx = 0.0, qy = 0.0, qz = 0.0, qw = 1.0;
        rpy_to_quat(pose_rpy[3], pose_rpy[4], pose_rpy[5], qx, qy, qz, qw);

        geometry_msgs::msg::Pose pose;
        pose.position.x = pose_rpy[0];
        pose.position.y = pose_rpy[1];
        pose.position.z = pose_rpy[2];
        pose.orientation.x = qx;
        pose.orientation.y = qy;
        pose.orientation.z = qz;
        pose.orientation.w = qw;
        return pose;
    }

    void publish_init_pose(Hand hand)
    {
        common::msg::OculusControllers msg;
        msg.header.stamp = this->now();
        msg.header.frame_id = frame_id_;
        msg.left_valid = (hand == Hand::Left || hand == Hand::Both);
        msg.right_valid = (hand == Hand::Right || hand == Hand::Both);
        msg.left_pose = make_pose_from_rpy(init_pose_);
        msg.right_pose = make_pose_from_rpy(init_pose_);
        pub_oculus_->publish(msg);

        RCLCPP_INFO(this->get_logger(),
                    "Published init pose: pos(%.3f, %.3f, %.3f) rpy(%.3f, %.3f, %.3f)",
                    init_pose_[0], init_pose_[1], init_pose_[2],
                    init_pose_[3], init_pose_[4], init_pose_[5]);
    }

    void publish_end_effector_cmd(Hand hand, int8_t mode, double position, int32_t preset)
    {
        if (attach_init_pose_)
        {
            publish_init_pose(hand);
        }

        je_software::msg::EndEffectorCommandLR msg;
        msg.left_valid = (hand == Hand::Left || hand == Hand::Both);
        msg.right_valid = (hand == Hand::Right || hand == Hand::Both);
        msg.left.mode = mode;
        msg.left.position = position;
        msg.left.preset = preset;
        msg.right.mode = mode;
        msg.right.position = position;
        msg.right.preset = preset;
        pub_cmd_->publish(msg);

        if (mode == je_software::msg::EndEffectorCommand::MODE_POSITION)
        {
            RCLCPP_INFO(this->get_logger(),
                        "Published end effector cmd: mode=POSITION position=%.3f", position);
        }
        else
        {
            RCLCPP_INFO(this->get_logger(),
                        "Published end effector cmd: mode=PRESET preset=%d", preset);
        }
    }

    void input_loop()
    {
        while (rclcpp::ok() && running_.load())
        {
            if (!poll_stdin(200))
            {
                continue;
            }

            std::string line;
            if (!std::getline(std::cin, line))
            {
                break;
            }
            if (line.empty())
            {
                continue;
            }

            for (char &c : line)
            {
                if (c == ',')
                {
                    c = ' ';
                }
            }
            std::istringstream iss(line);
            std::string cmd;
            iss >> cmd;
            Hand hand = default_hand_;
            if (is_hand_token(cmd))
            {
                hand = parse_hand_token(cmd);
                if (!(iss >> cmd))
                {
                    RCLCPP_WARN(this->get_logger(), "Missing command after hand prefix.");
                    continue;
                }
            }

            if (cmd == "help" || cmd == "h" || cmd == "?")
            {
                print_help();
                continue;
            }
            if (cmd == "quit" || cmd == "exit" || cmd == "q")
            {
                RCLCPP_INFO(this->get_logger(), "Exit requested by user.");
                rclcpp::shutdown();
                break;
            }
            if (cmd == "pose_on")
            {
                attach_init_pose_ = true;
                RCLCPP_INFO(this->get_logger(), "attach_init_pose_to_cmd = true");
                continue;
            }
            if (cmd == "pose_off")
            {
                attach_init_pose_ = false;
                RCLCPP_INFO(this->get_logger(), "attach_init_pose_to_cmd = false");
                continue;
            }
            if (cmd == "send_pose")
            {
                publish_init_pose(hand);
                continue;
            }
            if (cmd == "pose")
            {
                double x, y, z, r, p, yv;
                if (!(iss >> x >> y >> z >> r >> p >> yv))
                {
                    RCLCPP_WARN(this->get_logger(), "Usage: pose <x y z r p y>");
                    continue;
                }
                init_pose_ = {x, y, z, r, p, yv};
                RCLCPP_INFO(this->get_logger(), "Init pose updated.");
                continue;
            }
            if (cmd == "p" || cmd == "pos" || cmd == "position")
            {
                double value;
                if (!(iss >> value))
                {
                    RCLCPP_WARN(this->get_logger(), "Usage: p <value>");
                    continue;
                }
                publish_end_effector_cmd(hand,
                                         je_software::msg::EndEffectorCommand::MODE_POSITION,
                                         value, 0);
                continue;
            }
            if (cmd == "m" || cmd == "mode" || cmd == "preset")
            {
                int preset;
                if (!(iss >> preset))
                {
                    RCLCPP_WARN(this->get_logger(), "Usage: m <preset>");
                    continue;
                }
                publish_end_effector_cmd(hand,
                                         je_software::msg::EndEffectorCommand::MODE_PRESET,
                                         0.0, preset);
                continue;
            }

            RCLCPP_WARN(this->get_logger(), "Unknown command: %s", cmd.c_str());
        }
    }

private:
    std::string end_effector_topic_;
    Hand default_hand_{Hand::Left};
    std::string oculus_topic_;
    std::string frame_id_;
    std::array<double, 6> init_pose_{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    bool attach_init_pose_{false};

    rclcpp::Publisher<je_software::msg::EndEffectorCommandLR>::SharedPtr pub_cmd_;
    rclcpp::Publisher<common::msg::OculusControllers>::SharedPtr pub_oculus_;

    std::atomic_bool running_{false};
    std::thread input_thread_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<EndEffectorCli>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
