/**
0: 停止 6 电机
1..6: 停止单电机 (Motor1..Motor6)
10: 二指 0%
11: 二指 71%
20: 三指 0%
21: 三指 88%
*/
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

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

#include "je_software/msg/end_effector_command.hpp"

using namespace std::chrono_literals;

class EndEffectorCli : public rclcpp::Node
{
public:
    EndEffectorCli()
        : Node("end_effector_cli")
    {
        this->declare_parameter<std::string>("end_effector_topic", "/end_effector_cmd");
        this->declare_parameter<std::string>("pose_topic", "/end_pose");
        this->declare_parameter<std::string>("frame_id", "base_link");
        this->declare_parameter<bool>("send_init_pose_on_start", false);
        this->declare_parameter<bool>("attach_init_pose_to_cmd", false);
        this->declare_parameter<std::string>(
            "init_pose",
            "[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]");

        end_effector_topic_ = this->get_parameter("end_effector_topic").as_string();
        pose_topic_ = this->get_parameter("pose_topic").as_string();
        frame_id_ = this->get_parameter("frame_id").as_string();
        attach_init_pose_ = this->get_parameter("attach_init_pose_to_cmd").as_bool();
        init_pose_ = parse_init_pose_param(this->get_parameter("init_pose"));

        auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
        pub_cmd_ = this->create_publisher<je_software::msg::EndEffectorCommand>(end_effector_topic_, qos);
        pub_pose_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(pose_topic_, qos);

        RCLCPP_INFO(this->get_logger(), "[PUB] end effector cmd: %s", end_effector_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "[PUB] init pose (optional): %s (frame_id=%s)",
                    pose_topic_.c_str(), frame_id_.c_str());

        if (this->get_parameter("send_init_pose_on_start").as_bool())
        {
            publish_init_pose();
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
                  << "  p <value>            send MODE_POSITION with position\n"
                  << "  m <preset>           send MODE_PRESET with preset int\n"
                  << "  pose <x y z r p y>   set init pose (rpy in radians)\n"
                  << "  send_pose            publish init pose once\n"
                  << "  pose_on / pose_off   toggle attach init pose to each cmd\n"
                  << "  help\n"
                  << "  quit / exit\n"
                  << "Examples:\n"
                  << "  p 0.5\n"
                  << "  m 2\n"
                  << "  pose 0 0 0 0 0 0\n"
                  << "  pose_on\n"
                  << "========================\n\n";
    }

    void publish_init_pose()
    {
        geometry_msgs::msg::PoseStamped msg;
        msg.header.stamp = this->now();
        msg.header.frame_id = frame_id_;
        msg.pose.position.x = init_pose_[0];
        msg.pose.position.y = init_pose_[1];
        msg.pose.position.z = init_pose_[2];
        double qx = 0.0, qy = 0.0, qz = 0.0, qw = 1.0;
        rpy_to_quat(init_pose_[3], init_pose_[4], init_pose_[5], qx, qy, qz, qw);
        msg.pose.orientation.x = qx;
        msg.pose.orientation.y = qy;
        msg.pose.orientation.z = qz;
        msg.pose.orientation.w = qw;
        pub_pose_->publish(msg);

        RCLCPP_INFO(this->get_logger(),
                    "Published init pose: pos(%.3f, %.3f, %.3f) rpy(%.3f, %.3f, %.3f)",
                    init_pose_[0], init_pose_[1], init_pose_[2],
                    init_pose_[3], init_pose_[4], init_pose_[5]);
    }

    void publish_end_effector_cmd(int8_t mode, double position, int32_t preset)
    {
        if (attach_init_pose_)
        {
            publish_init_pose();
        }

        je_software::msg::EndEffectorCommand msg;
        msg.mode = mode;
        msg.position = position;
        msg.preset = preset;
        pub_cmd_->publish(msg);

        if (mode == je_software::msg::EndEffectorCommand::MODE_POSITION)
        {
            RCLCPP_INFO(this->get_logger(), "Published end effector cmd: mode=POSITION position=%.3f", position);
        }
        else
        {
            RCLCPP_INFO(this->get_logger(), "Published end effector cmd: mode=PRESET preset=%d", preset);
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
                publish_init_pose();
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
                publish_end_effector_cmd(je_software::msg::EndEffectorCommand::MODE_POSITION,
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
                publish_end_effector_cmd(je_software::msg::EndEffectorCommand::MODE_PRESET,
                                         0.0, preset);
                continue;
            }

            RCLCPP_WARN(this->get_logger(), "Unknown command: %s", cmd.c_str());
        }
    }

private:
    std::string end_effector_topic_;
    std::string pose_topic_;
    std::string frame_id_;
    std::array<double, 6> init_pose_{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    bool attach_init_pose_{false};

    rclcpp::Publisher<je_software::msg::EndEffectorCommand>::SharedPtr pub_cmd_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_pose_;

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
