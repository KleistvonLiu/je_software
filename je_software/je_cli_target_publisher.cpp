#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// Linux poll for stdin
#include <poll.h>
#include <unistd.h>

using namespace std::chrono_literals;

class CliTargetPublisher : public rclcpp::Node
{
public:
    CliTargetPublisher()
        : Node("cli_target_publisher")
    {
        // ---------------- Parameters ----------------
        this->declare_parameter<std::string>("joint_topic", "/joint_cmd");
        this->declare_parameter<std::string>("pose_topic", "/end_pose");
        this->declare_parameter<std::string>("frame_id", "base_link");
        this->declare_parameter<std::vector<std::string>>(
            "joint_names",
            {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"});

        joint_topic_ = this->get_parameter("joint_topic").as_string();
        pose_topic_ = this->get_parameter("pose_topic").as_string();
        frame_id_ = this->get_parameter("frame_id").as_string();
        joint_names_ = this->get_parameter("joint_names").as_string_array();

        if (joint_names_.empty())
        {
            joint_names_ = {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"};
        }

        auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();

        pub_joint_ = this->create_publisher<sensor_msgs::msg::JointState>(joint_topic_, qos);
        pub_pose_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(pose_topic_, qos);

        RCLCPP_INFO(this->get_logger(), "Publishing JointState  -> %s", joint_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "Publishing PoseStamped -> %s (frame_id=%s)",
                    pose_topic_.c_str(), frame_id_.c_str());

        running_.store(true);
        input_thread_ = std::thread(&CliTargetPublisher::input_loop, this);

        print_help();
    }

    ~CliTargetPublisher() override
    {
        running_.store(false);
        if (input_thread_.joinable())
        {
            input_thread_.join();
        }
    }

private:
    static constexpr double kPi = 3.14159265358979323846;

    static void rpy_to_quat(double roll, double pitch, double yaw,
                            double &qx, double &qy, double &qz, double &qw)
    {
        // ZYX: yaw (Z), pitch (Y), roll (X)
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

        // normalize
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
            qx = 0.0;
            qy = 0.0;
            qz = 0.0;
            qw = 1.0;
        }
    }

    void print_help()
    {
        std::cout << "\n=== CLI Target Publisher ===\n"
                  << "Commands:\n"
                  << "  j   <q1> <q2> <q3> <q4> <q5> <q6> <q7>\n"
                  << "       Publish sensor_msgs/JointState to joint_topic\n"
                  << "  p   <x> <y> <z> <qx> <qy> <qz> <qw>\n"
                  << "       Publish geometry_msgs/PoseStamped (quaternion) to pose_topic\n"
                  << "  rpy <x> <y> <z> <roll_deg> <pitch_deg> <yaw_deg>\n"
                  << "       Publish PoseStamped, interpreting RPY in degrees\n"
                  << "  rpy_rad <x> <y> <z> <roll> <pitch> <yaw>\n"
                  << "       Publish PoseStamped, interpreting RPY in radians\n"
                  << "  help\n"
                  << "  quit / exit\n"
                  << "Examples:\n"
                  << "  j 0 0 -1.56 -0.75 0 -0.65 0\n"
                  << "  p 0.30 0.10 0.20 0 0 0 1\n"
                  << "  rpy 0.30 0.10 0.20 0 90 0\n"
                  << "============================\n\n";
    }

    bool poll_stdin(int timeout_ms)
    {
        struct pollfd pfd;
        pfd.fd = STDIN_FILENO;
        pfd.events = POLLIN;
        const int ret = ::poll(&pfd, 1, timeout_ms);
        return (ret > 0) && (pfd.revents & POLLIN);
    }

    void publish_joint(const std::vector<double> &q)
    {
        std::cout << "here" << std::endl;
        sensor_msgs::msg::JointState msg;
        msg.header.stamp = this->now();
        msg.name = joint_names_;
        msg.position = q;
        pub_joint_->publish(msg);

        RCLCPP_INFO(this->get_logger(), "Published JointState: [%zu] joints", q.size());
    }

    void publish_pose(double x, double y, double z,
                      double qx, double qy, double qz, double qw)
    {
        geometry_msgs::msg::PoseStamped msg;
        msg.header.stamp = this->now();
        msg.header.frame_id = frame_id_;
        msg.pose.position.x = x;
        msg.pose.position.y = y;
        msg.pose.position.z = z;
        msg.pose.orientation.x = qx;
        msg.pose.orientation.y = qy;
        msg.pose.orientation.z = qz;
        msg.pose.orientation.w = qw;

        pub_pose_->publish(msg);

        RCLCPP_INFO(this->get_logger(),
                    "Published PoseStamped: pos(%.3f, %.3f, %.3f) quat(%.3f, %.3f, %.3f, %.3f)",
                    x, y, z, qx, qy, qz, qw);
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
                // stdin closed
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

            if (cmd == "j" || cmd == "joint")
            {
                std::vector<double> q;
                q.reserve(joint_names_.size());
                double v;
                while (iss >> v)
                    q.push_back(v);

                if (q.size() != joint_names_.size())
                {
                    RCLCPP_WARN(this->get_logger(),
                                "Joint cmd expects %zu values, got %zu",
                                joint_names_.size(), q.size());
                    continue;
                }
                publish_joint(q);
                continue;
            }

            if (cmd == "p" || cmd == "pose")
            {
                double x, y, z, qx, qy, qz, qw;
                if (!(iss >> x >> y >> z >> qx >> qy >> qz >> qw))
                {
                    RCLCPP_WARN(this->get_logger(),
                                "Pose cmd format: p x y z qx qy qz qw");
                    continue;
                }
                publish_pose(x, y, z, qx, qy, qz, qw);
                continue;
            }

            if (cmd == "rpy" || cmd == "rpy_deg" || cmd == "rpy_rad")
            {
                double x, y, z, roll, pitch, yaw;
                if (!(iss >> x >> y >> z >> roll >> pitch >> yaw))
                {
                    RCLCPP_WARN(this->get_logger(),
                                "RPY cmd format: rpy x y z roll pitch yaw");
                    continue;
                }

                if (cmd == "rpy_deg")
                {
                    roll = roll * kPi / 180.0;
                    pitch = pitch * kPi / 180.0;
                    yaw = yaw * kPi / 180.0;
                }

                double qx, qy, qz, qw;
                rpy_to_quat(roll, pitch, yaw, qx, qy, qz, qw);
                publish_pose(x, y, z, qx, qy, qz, qw);
                continue;
            }

            RCLCPP_WARN(this->get_logger(), "Unknown command: %s (type 'help')", cmd.c_str());
        }
    }

private:
    std::string joint_topic_;
    std::string pose_topic_;
    std::string frame_id_;
    std::vector<std::string> joint_names_;

    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_joint_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_pose_;

    std::atomic_bool running_{false};
    std::thread input_thread_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CliTargetPublisher>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
