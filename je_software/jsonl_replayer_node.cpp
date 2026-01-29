#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/pose.hpp>

#include <nlohmann/json.hpp>
#include <builtin_interfaces/msg/time.hpp>

#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <optional>
#include <cctype>

#include "common/msg/oculus_controllers.hpp"
#include "common/msg/oculus_init_joint_state.hpp"
#include "ros2_qos.hpp"

using json = nlohmann::json;

static inline void rpy_to_quat(double roll, double pitch, double yaw,
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

class JsonlReplayerNode : public rclcpp::Node
{
public:
    JsonlReplayerNode()
        : Node("jsonl_replayer_node")
    {
        // -------------------- Parameters --------------------
        this->declare_parameter<std::string>("jsonl_path", "");
        this->declare_parameter<double>("rate_hz", 50.0);
        this->declare_parameter<bool>("loop", true);

        // output_type: "oculus_joint" or "oculus_pose"
        this->declare_parameter<std::string>("output_type", "oculus_joint");
        this->declare_parameter<std::string>("oculus_controllers_topic", "/oculus_controllers");
        this->declare_parameter<std::string>("oculus_init_joint_state_topic", "/oculus_init_joint_state");

        // use file stamp
        this->declare_parameter<bool>("use_file_stamp", true);
    this->declare_parameter<double>("dt_init", 5.0);

        // Pose options
        this->declare_parameter<std::string>("frame_id", "base_link");
        this->declare_parameter<std::string>("pose_field", "Cartesian"); // e.g. "Cartesian" / "TargetCartesian"
        this->declare_parameter<bool>("pose_left_valid", true);
        this->declare_parameter<bool>("pose_right_valid", true);
        this->declare_parameter<std::string>("send_arm", "both");

        // JointState options: choose which JSON fields map to position/velocity/effort
        this->declare_parameter<std::vector<std::string>>(
            "joint_names",
            std::vector<std::string>{"joint1","joint2","joint3","joint4","joint5","joint6","joint7"}
        );
        this->declare_parameter<std::string>("joint_position_field", "Joint");      // e.g. "Joint" / "TargetJoint"
        this->declare_parameter<std::string>("joint_velocity_field", "");           // e.g. "JointVelocity" or ""
        this->declare_parameter<std::string>("joint_effort_field", "");             // e.g. "JointTorque" / "JointSensorTorque" or ""
        this->declare_parameter<bool>("joint_init_flag", false);
        this->declare_parameter<std::string>(
            "init_left_joint_position",
            "[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]");
        this->declare_parameter<std::string>(
            "init_right_joint_position",
            "[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]");
        this->declare_parameter<bool>("init_left_valid", true);
        this->declare_parameter<bool>("init_right_valid", true);

        // -------------------- Read parameters --------------------
        jsonl_path_ = this->get_parameter("jsonl_path").as_string();
        rate_hz_ = this->get_parameter("rate_hz").as_double();
        loop_ = this->get_parameter("loop").as_bool();
        output_type_ = this->get_parameter("output_type").as_string();
        oculus_controllers_topic_ = this->get_parameter("oculus_controllers_topic").as_string();
        oculus_init_joint_state_topic_ = this->get_parameter("oculus_init_joint_state_topic").as_string();
        use_file_stamp_ = this->get_parameter("use_file_stamp").as_bool();
        init_delay_sec_ = this->get_parameter("dt_init").as_double();
        to_lower_inplace(output_type_);

        frame_id_ = this->get_parameter("frame_id").as_string();
        pose_field_ = this->get_parameter("pose_field").as_string();
        pose_left_valid_ = this->get_parameter("pose_left_valid").as_bool();
        pose_right_valid_ = this->get_parameter("pose_right_valid").as_bool();
        send_arm_ = this->get_parameter("send_arm").as_string();
        to_lower_inplace(send_arm_);

        joint_names_ = this->get_parameter("joint_names").as_string_array();
        joint_position_field_ = this->get_parameter("joint_position_field").as_string();
        joint_velocity_field_ = this->get_parameter("joint_velocity_field").as_string();
        joint_effort_field_ = this->get_parameter("joint_effort_field").as_string();
        joint_init_flag_ = this->get_parameter("joint_init_flag").as_bool();
        init_left_valid_ = this->get_parameter("init_left_valid").as_bool();
        init_right_valid_ = this->get_parameter("init_right_valid").as_bool();
        if (send_arm_ == "left")
        {
            pose_right_valid_ = false;
            init_right_valid_ = false;
        }
        else if (send_arm_ == "right")
        {
            pose_left_valid_ = false;
            init_left_valid_ = false;
        }

        if (jsonl_path_.empty())
        {
            throw std::runtime_error("Parameter 'jsonl_path' is empty. Please set it in launch.");
        }
        if (rate_hz_ <= 0.0)
        {
            RCLCPP_WARN(this->get_logger(), "rate_hz <= 0, fallback to 50.0");
            rate_hz_ = 50.0;
        }
        if (init_delay_sec_ < 0.0)
        {
            RCLCPP_WARN(this->get_logger(), "dt_init < 0, fallback to 5.0");
            init_delay_sec_ = 5.0;
        }

        // -------------------- Open file --------------------
        open_file_or_throw();
        prepare_init_from_first_record();

        // -------------------- Create publisher --------------------
        auto qos = common_utils::reliable_qos_shallow();
        pub_oculus_joint_ = this->create_publisher<common::msg::OculusInitJointState>(
            oculus_init_joint_state_topic_, qos);
        if (output_type_ == "oculus_joint")
        {
            RCLCPP_INFO(this->get_logger(), "Publishing OculusInitJointState on: %s",
                        oculus_init_joint_state_topic_.c_str());
        }
        else if (output_type_ == "oculus_pose")
        {
            pub_oculus_pose_ = this->create_publisher<common::msg::OculusControllers>(
                oculus_controllers_topic_, qos);
            RCLCPP_INFO(this->get_logger(), "Publishing OculusControllers on: %s (frame_id=%s)",
                        oculus_controllers_topic_.c_str(), frame_id_.c_str());
        }
        else
        {
            throw std::runtime_error("output_type must be 'oculus_joint' or 'oculus_pose'.");
        }

        // -------------------- Timer --------------------
        const auto period = std::chrono::duration<double>(1.0 / rate_hz_);
        timer_ = this->create_wall_timer(
            std::chrono::duration_cast<std::chrono::nanoseconds>(period),
            std::bind(&JsonlReplayerNode::on_timer, this)
        );

        RCLCPP_INFO(this->get_logger(),
                    "Started replay: file=%s rate_hz=%.3f loop=%s use_file_stamp=%s",
                    jsonl_path_.c_str(), rate_hz_,
                    loop_ ? "true":"false",
                    use_file_stamp_ ? "true":"false");
    }

    ~JsonlReplayerNode() override
    {
        try { if (ifs_.is_open()) ifs_.close(); } catch (...) {}
    }

private:
    // -------------------- Helpers --------------------
    static inline void to_lower_inplace(std::string &s)
    {
        for (auto &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    void open_file_or_throw()
    {
        if (ifs_.is_open()) ifs_.close();
        ifs_.open(jsonl_path_);
        if (!ifs_.is_open())
        {
            throw std::runtime_error("Failed to open jsonl file: " + jsonl_path_);
        }
    }

    std::optional<json> read_next_record()
    {
        std::string line;
        while (std::getline(ifs_, line))
        {
            // skip empty lines
            if (line.empty()) continue;
            try
            {
                return json::parse(line);
            }
            catch (const std::exception &e)
            {
                RCLCPP_WARN(this->get_logger(), "JSON parse failed, skip line: %s", e.what());
                continue;
            }
        }
        return std::nullopt; // EOF
    }

    rclcpp::Time make_stamp(const json &obj)
    {
        if (use_file_stamp_ && obj.contains("__ros_stamp_ns"))
        {
            try
            {
                const int64_t ns = obj.at("__ros_stamp_ns").get<int64_t>();
                // 默认按 SYSTEM_TIME 解释（你的示例 stamp 看起来就是 epoch ns）
                return rclcpp::Time(ns, RCL_SYSTEM_TIME);
            }
            catch (...)
            {
                // fallback
            }
        }
        if (use_file_stamp_ && obj.contains("timestamp"))
        {
            try
            {
                const double sec = obj.at("timestamp").get<double>();
                const int64_t ns = static_cast<int64_t>(sec * 1e9);
                return rclcpp::Time(ns, RCL_SYSTEM_TIME);
            }
            catch (...)
            {
                // fallback
            }
        }
        if (use_file_stamp_ && obj.contains("joints") && obj["joints"].is_array() && !obj["joints"].empty())
        {
            try
            {
                const auto &entry = obj["joints"].front();
                if (entry.contains("stamp_ns"))
                {
                    if (entry["stamp_ns"].is_number_float())
                    {
                        const double sec = entry["stamp_ns"].get<double>();
                        const int64_t ns = static_cast<int64_t>(sec * 1e9);
                        return rclcpp::Time(ns, RCL_SYSTEM_TIME);
                    }
                    const int64_t v = entry["stamp_ns"].get<int64_t>();
                    if (v > 1000000000000LL)
                    {
                        return rclcpp::Time(v, RCL_SYSTEM_TIME);
                    }
                    return rclcpp::Time(static_cast<int64_t>(static_cast<double>(v) * 1e9), RCL_SYSTEM_TIME);
                }
            }
            catch (...)
            {
                // fallback
            }
        }
        return this->get_clock()->now();
    }

    static builtin_interfaces::msg::Time to_builtin_time(const rclcpp::Time &stamp)
    {
        builtin_interfaces::msg::Time out;
        const int64_t ns = stamp.nanoseconds();
        out.sec = static_cast<int32_t>(ns / 1000000000LL);
        out.nanosec = static_cast<uint32_t>(ns % 1000000000LL);
        return out;
    }

    static std::optional<std::vector<double>> get_vec7(const json &robot0, const std::string &field)
    {
        if (field.empty()) return std::nullopt;
        if (!robot0.contains(field)) return std::nullopt;
        if (!robot0.at(field).is_array()) return std::nullopt;

        const auto &arr = robot0.at(field);
        if (arr.size() < 7) return std::nullopt;

        std::vector<double> v(7);
        for (size_t i = 0; i < 7; ++i) v[i] = arr.at(i).get<double>();
        return v;
    }

    static bool is_left_topic(const std::string &topic)
    {
        return topic.find("left") != std::string::npos;
    }

    static bool is_right_topic(const std::string &topic)
    {
        return topic.find("right") != std::string::npos;
    }

    static bool fill_joint_from_meta_entry(const json &entry,
                                           sensor_msgs::msg::JointState &out,
                                           bool use_effort_filtered)
    {
        if (!entry.contains("position") || !entry["position"].is_array())
        {
            return false;
        }

        out.name.clear();
        out.position.clear();
        out.velocity.clear();
        out.effort.clear();

        if (entry.contains("name") && entry["name"].is_array())
        {
            out.name = entry["name"].get<std::vector<std::string>>();
        }

        out.position = entry["position"].get<std::vector<double>>();
        if (entry.contains("velocity") && entry["velocity"].is_array())
        {
            out.velocity = entry["velocity"].get<std::vector<double>>();
        }
        if (use_effort_filtered && entry.contains("effort_filtered") && entry["effort_filtered"].is_array())
        {
            out.effort = entry["effort_filtered"].get<std::vector<double>>();
        }
        else if (entry.contains("effort") && entry["effort"].is_array())
        {
            out.effort = entry["effort"].get<std::vector<double>>();
        }

        return !out.position.empty();
    }

    static bool parse_gripper_value(const json &entry, float &out)
    {
        if (!entry.contains("gripper"))
        {
            return false;
        }
        const auto &grip = entry["gripper"];
        if (grip.is_number())
        {
            out = static_cast<float>(grip.get<double>());
            return true;
        }
        if (grip.is_array() && !grip.empty() && grip[0].is_number())
        {
            out = static_cast<float>(grip[0].get<double>());
            return true;
        }
        return false;
    }

    bool fill_from_meta_joints(const json &obj,
                               common::msg::OculusInitJointState &msg)
    {
        if (!obj.contains("joints") || !obj["joints"].is_array())
        {
            return false;
        }

        for (const auto &entry : obj["joints"])
        {
            if (!entry.is_object())
            {
                continue;
            }
            std::string topic;
            if (entry.contains("topic") && entry["topic"].is_string())
            {
                topic = entry["topic"].get<std::string>();
            }

            if (is_left_topic(topic))
            {
                if (fill_joint_from_meta_entry(entry, msg.left, false))
                {
                    msg.left_valid = true;
                }
                parse_gripper_value(entry, msg.left_gripper);
            }
            else if (is_right_topic(topic))
            {
                if (fill_joint_from_meta_entry(entry, msg.right, false))
                {
                    msg.right_valid = true;
                }
                parse_gripper_value(entry, msg.right_gripper);
            }
        }

        return msg.left_valid || msg.right_valid;
    }

    static std::optional<std::array<double,6>> get_cart6(const json &robot0, const std::string &field)
    {
        if (field.empty()) return std::nullopt;
        if (!robot0.contains(field)) return std::nullopt;
        if (!robot0.at(field).is_array()) return std::nullopt;

        const auto &arr = robot0.at(field);
        if (arr.size() < 6) return std::nullopt;

        std::array<double,6> out{};
        for (size_t i = 0; i < 6; ++i) out[i] = arr.at(i).get<double>();
        return out;
    }

    static geometry_msgs::msg::Pose make_pose_from_cart(const std::array<double, 6> &cart)
    {
        const double x = cart[0];
        const double y = cart[1];
        const double z = cart[2];
        const double roll  = cart[3];
        const double pitch = cart[4];
        const double yaw   = cart[5];

        double qx, qy, qz, qw;
        rpy_to_quat(roll, pitch, yaw, qx, qy, qz, qw);

        geometry_msgs::msg::Pose pose;
        pose.position.x = x;
        pose.position.y = y;
        pose.position.z = z;
        pose.orientation.x = qx;
        pose.orientation.y = qy;
        pose.orientation.z = qz;
        pose.orientation.w = qw;
        return pose;
    }

    bool fill_joint_state(const json &robot, sensor_msgs::msg::JointState &msg)
    {
        auto pos = get_vec7(robot, joint_position_field_);
        auto vel = get_vec7(robot, joint_velocity_field_);
        auto eff = get_vec7(robot, joint_effort_field_);

        if (!pos.has_value())
        {
            return false;
        }

        msg.name = joint_names_;
        msg.position = *pos;
        if (vel.has_value()) msg.velocity = *vel;
        if (eff.has_value()) msg.effort = *eff;
        return true;
    }

    bool build_init_joint_from_record(const json &obj,
                                      common::msg::OculusInitJointState &msg)
    {
        msg.left = sensor_msgs::msg::JointState();
        msg.right = sensor_msgs::msg::JointState();
        msg.left_valid = false;
        msg.right_valid = false;

        bool filled = fill_from_meta_joints(obj, msg);

        if (!filled)
        {
            if (init_left_valid_ && obj.contains("Robot0") && obj.at("Robot0").is_object())
            {
                const auto &robot0 = obj.at("Robot0");
                if (fill_joint_state(robot0, msg.left))
                {
                    msg.left_valid = true;
                }
            }

            if (init_right_valid_ && obj.contains("Robot1") && obj.at("Robot1").is_object())
            {
                const auto &robot1 = obj.at("Robot1");
                if (fill_joint_state(robot1, msg.right))
                {
                    msg.right_valid = true;
                }
            }
        }

        if (!init_left_valid_)
        {
            msg.left_valid = false;
            msg.left = sensor_msgs::msg::JointState();
        }
        if (!init_right_valid_)
        {
            msg.right_valid = false;
            msg.right = sensor_msgs::msg::JointState();
        }

        if (msg.left_valid && msg.left.name.empty())
        {
            msg.left.name = joint_names_;
        }
        if (msg.right_valid && msg.right.name.empty())
        {
            msg.right.name = joint_names_;
        }

        return msg.left_valid || msg.right_valid;
    }

    bool build_init_pose_from_record(const json &obj,
                                     common::msg::OculusControllers &msg)
    {
        msg.left_valid = false;
        msg.right_valid = false;

        if (pose_left_valid_ && obj.contains("Robot0") && obj.at("Robot0").is_object())
        {
            const auto &robot0 = obj.at("Robot0");
            auto cart = get_cart6(robot0, pose_field_);
            if (cart.has_value())
            {
                msg.left_pose = make_pose_from_cart(*cart);
                msg.left_valid = true;
            }
        }

        if (pose_right_valid_ && obj.contains("Robot1") && obj.at("Robot1").is_object())
        {
            const auto &robot1 = obj.at("Robot1");
            auto cart = get_cart6(robot1, pose_field_);
            if (cart.has_value())
            {
                msg.right_pose = make_pose_from_cart(*cart);
                msg.right_valid = true;
            }
        }

        return msg.left_valid || msg.right_valid;
    }

    void prepare_init_from_first_record()
    {
        if (output_type_ != "oculus_joint" && output_type_ != "oculus_pose")
        {
            throw std::runtime_error("output_type must be 'oculus_joint' or 'oculus_pose'.");
        }

        auto rec = read_next_record();
        if (!rec.has_value())
        {
            throw std::runtime_error("jsonl file is empty; cannot initialize from first record.");
        }
        first_record_ = *rec;

        common::msg::OculusInitJointState joint_msg;
        const bool joint_ok = build_init_joint_from_record(*rec, joint_msg);

        common::msg::OculusControllers pose_msg;
        const bool pose_ok = build_init_pose_from_record(*rec, pose_msg);

        if (output_type_ == "oculus_joint")
        {
            if (!joint_ok)
            {
                throw std::runtime_error("First record has no valid joint data for initialization.");
            }
            init_joint_msg_ = joint_msg;
            init_use_pose_ = false;
            return;
        }

        if (pose_ok)
        {
            init_pose_msg_ = pose_msg;
            init_use_pose_ = true;
            return;
        }

        if (joint_ok)
        {
            RCLCPP_WARN(this->get_logger(),
                        "First record missing pose_field='%s'; init from joint instead.",
                        pose_field_.c_str());
            init_joint_msg_ = joint_msg;
            init_use_pose_ = false;
            return;
        }

        throw std::runtime_error("First record has no valid pose or joint data for initialization.");
    }

    void publish_init_joint()
    {
        common::msg::OculusInitJointState msg = init_joint_msg_;
        msg.header.stamp = to_builtin_time(this->get_clock()->now());
        msg.header.frame_id = frame_id_;
        msg.init = true;

        pub_oculus_joint_->publish(msg);
    }

    void publish_init_pose()
    {
        common::msg::OculusControllers msg = init_pose_msg_;
        msg.header.stamp = to_builtin_time(this->get_clock()->now());
        msg.header.frame_id = frame_id_;

        pub_oculus_pose_->publish(msg);
    }

    void publish_oculus_joint(const json &obj)
    {
        common::msg::OculusInitJointState msg;
        msg.header.stamp = to_builtin_time(make_stamp(obj));
        msg.header.frame_id = frame_id_;
        msg.init = joint_init_flag_;

        if (fill_from_meta_joints(obj, msg))
        {
            pub_oculus_joint_->publish(msg);
            return;
        }

        if (init_left_valid_ && obj.contains("Robot0") && obj.at("Robot0").is_object())
        {
            const auto &robot0 = obj.at("Robot0");
            if (fill_joint_state(robot0, msg.left))
            {
                msg.left_valid = true;
            }
        }

        if (init_right_valid_ && obj.contains("Robot1") && obj.at("Robot1").is_object())
        {
            const auto &robot1 = obj.at("Robot1");
            if (fill_joint_state(robot1, msg.right))
            {
                msg.right_valid = true;
            }
        }

        if (!msg.left_valid && !msg.right_valid)
        {
            RCLCPP_WARN(this->get_logger(),
                        "No valid joint fields found (pos=%s) for Robot0/Robot1; skip.",
                        joint_position_field_.c_str());
            return;
        }

        pub_oculus_joint_->publish(msg);
    }

    void publish_oculus_pose(const json &obj)
    {
        common::msg::OculusControllers msg;
        msg.header.stamp = to_builtin_time(make_stamp(obj));
        msg.header.frame_id = frame_id_;

        if (pose_left_valid_ && obj.contains("Robot0") && obj.at("Robot0").is_object())
        {
            const auto &robot0 = obj.at("Robot0");
            auto cart = get_cart6(robot0, pose_field_);
            if (cart.has_value())
            {
                msg.left_pose = make_pose_from_cart(*cart);
                msg.left_valid = true;
            }
        }

        if (pose_right_valid_ && obj.contains("Robot1") && obj.at("Robot1").is_object())
        {
            const auto &robot1 = obj.at("Robot1");
            auto cart = get_cart6(robot1, pose_field_);
            if (cart.has_value())
            {
                msg.right_pose = make_pose_from_cart(*cart);
                msg.right_valid = true;
            }
        }

        if (!msg.left_valid && !msg.right_valid)
        {
            RCLCPP_WARN(this->get_logger(),
                        "No valid pose_field='%s' found for Robot0/Robot1; skip.",
                        pose_field_.c_str());
            return;
        }

        pub_oculus_pose_->publish(msg);
    }

    void on_timer()
    {
        if (init_sent_count_ < init_repeat_count_)
        {
            // RCLCPP_INFO(this->get_logger(), "Publish initial position.");
            if (init_use_pose_)
            {
                publish_init_pose();
            }
            else
            {
                publish_init_joint();
            }
            ++init_sent_count_;
            if (init_sent_count_ >= init_repeat_count_)
            {
                init_sent_time_ = this->get_clock()->now();
            }
            return;
        }
        if ((this->get_clock()->now() - init_sent_time_).seconds() < init_delay_sec_)
        {
            return;
        }

        std::optional<json> rec;
        if (first_record_.has_value())
        {
            rec = *first_record_;
            first_record_.reset();
        }
        else
        {
            rec = read_next_record();
        }
        if (!rec.has_value())
        {
            if (loop_)
            {
                open_file_or_throw();
                rec = read_next_record();
                if (!rec.has_value())
                {
                    RCLCPP_ERROR(this->get_logger(), "File empty after reopen; stop timer.");
                    timer_->cancel();
                    return;
                }
            }
            else
            {
                RCLCPP_INFO(this->get_logger(), "Reached EOF and loop=false; stop timer.");
                timer_->cancel();
                return;
            }
        }

        if (output_type_ == "oculus_joint")
            publish_oculus_joint(*rec);
        else
            publish_oculus_pose(*rec);
    }

private:
    // params
    std::string jsonl_path_;
    double rate_hz_{50.0};
    bool loop_{true};
    std::string output_type_{"oculus_joint"};
    std::string oculus_controllers_topic_{"/oculus_controllers"};
    std::string oculus_init_joint_state_topic_{"/oculus_init_joint_state"};
    bool use_file_stamp_{true};

    // pose params
    std::string frame_id_{"base_link"};
    std::string pose_field_{"Cartesian"};
    bool pose_left_valid_{true};
    bool pose_right_valid_{true};
    std::string send_arm_{"both"};

    // joint params
    std::vector<std::string> joint_names_;
    std::string joint_position_field_{"Joint"};
    std::string joint_velocity_field_{""};
    std::string joint_effort_field_{""};
    bool joint_init_flag_{false};
    bool init_left_valid_{true};
    bool init_right_valid_{true};
    bool init_use_pose_{false};
    common::msg::OculusInitJointState init_joint_msg_{};
    common::msg::OculusControllers init_pose_msg_{};
    std::optional<json> first_record_{};
    int init_sent_count_{0};
    int init_repeat_count_{10};
    rclcpp::Time init_sent_time_{0, 0, RCL_SYSTEM_TIME};
    double init_delay_sec_{5.0};

    // io
    std::ifstream ifs_;

    // pubs/timer
    rclcpp::Publisher<common::msg::OculusInitJointState>::SharedPtr pub_oculus_joint_;
    rclcpp::Publisher<common::msg::OculusControllers>::SharedPtr pub_oculus_pose_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<JsonlReplayerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
