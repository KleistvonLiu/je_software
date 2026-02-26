#include "HYYRobotInterface.h"
#include "device_interface.h"
#include "end_effector_device.h"
#include "dh_modbus_gripper.h"
#include <zmq.hpp>
#include <string>
#include <iostream>
#include <fstream>   // ① 增加这一行
#include <thread>
#include "nlohmann/json.hpp"
#include <vector>
#include <atomic>
#include <cstring>
#include <memory>
#include <unistd.h>
#include <iomanip>
#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief 插件入库函数,用于实现插件初始化(被控制系初始函数调用),该函数要求非阻塞，如阻塞需要开线程运行
 */
extern void PluginMain();
#ifdef __cplusplus
}
#endif

#define CYCLIE 10

/// log相关
#ifndef JOINT_DEBUG_LOG
#define JOINT_DEBUG_LOG 0
#endif

#if JOINT_DEBUG_LOG
#define JLOG(msg) do { std::cout << msg << std::endl; } while(0)
#define JERR(msg) do { std::cerr << msg << std::endl; } while(0)
#else
#define JLOG(msg) do {} while(0)
#define JERR(msg) do {} while(0)
#endif

static zmq::context_t context(1);
static zmq::socket_t publisher(context, zmq::socket_type::pub);
static zmq::socket_t subscriber(context, zmq::socket_type::sub);
static std::thread* pub_th=nullptr;
static std::thread* sub_th=nullptr;
static std::atomic<bool> is_stop(true);
static std::thread* stop_th=nullptr;
static std::atomic<double> g_gripper_position{-1.0};  // last known gripper position (0~1, -1 invalid)

// End effector serial config (adjust port if needed)
static const char* kEndEffectorPort = "/dev/ttyS1";
static constexpr int kEndEffectorBaud = 115200;
static std::unique_ptr<EndEffectorDevice> end_effector_device;

// DH Modbus gripper config (adjust port/id if needed)
static const char* kGripperPort = "/dev/ttyUSB0";
static constexpr int kGripperBaud = 115200;
static constexpr int kGripperId = 1;
static std::unique_ptr<DH_Modbus_Gripper> dh_gripper;

// Binding: Robot1 -> EndEffectorDevice, Robot0 -> Gripper
static constexpr int kRobotIndexGripper = 0;
static constexpr int kRobotIndexEndEffector = 1;

static EndEffectorDevice* get_end_effector()
{
    if (!end_effector_device)
    {
        std::cerr << "end effector serial not ready, skip command\n";
        return nullptr;
    }
    return end_effector_device.get();
}

static DH_Modbus_Gripper* get_gripper()
{
    if (!dh_gripper)
    {
        std::cerr << "gripper not ready, skip command\n";
        return nullptr;
    }
    return dh_gripper.get();
}

static void handle_bound_end_effector(int robot_index,
                                      const nlohmann::ordered_json& ee,
                                      const char* context,
                                      bool debug_log)
{
    // std::cout << "Here: " << robot_index << std::endl;
    auto log_err = [&](const std::string& msg) {
        if (debug_log)
            JERR(msg);
        else
            std::cerr << msg << std::endl;
    };

    if (robot_index == kRobotIndexEndEffector)
    {
        if (!ee.contains("mode") || !ee["mode"].is_number_integer())
        {
            log_err(std::string(context) + " EndEffector missing/invalid mode");
            return;
        }

        int mode = ee["mode"].get<int>();
        if (mode == 0 && ee.contains("position") && ee["position"].is_number())
        {
            if (EndEffectorDevice* ee_dev = get_end_effector())
                ee_dev->HandlePosition(ee["position"].get<double>());
        }
        else if (mode == 1 && ee.contains("preset") && ee["preset"].is_number_integer())
        {
            if (EndEffectorDevice* ee_dev = get_end_effector())
                ee_dev->HandlePreset(ee["preset"].get<int>());
        }
        else
        {
            log_err(std::string(context) + " EndEffector missing/invalid fields");
        }
        return;
    }

    if (robot_index == kRobotIndexGripper)
    {
        // std::cout << "Here: " << ee << std::endl;
        DH_Modbus_Gripper* gripper = get_gripper();
        if (!gripper)
        {
            log_err(std::string(context) + " gripper not ready");
            return;
        }

        bool handled = false;
        if (ee.contains("init") && ee["init"].is_boolean() && ee["init"].get<bool>())
        {
            if (!gripper->Initialization())
                log_err(std::string(context) + " gripper init failed");
            handled = true;
        }
        if (ee.contains("position") && ee["position"].is_number())
        {
            int pos = static_cast<int>(ee["position"].get<double>() * 1000);
            // std::cout << "control gripper position: " << pos << std::endl;
            if (!gripper->SetTargetPosition(pos))
                log_err(std::string(context) + " gripper set position failed");
            handled = true;
        }
        if (ee.contains("force") && ee["force"].is_number())
        {
            int force = static_cast<int>(ee["force"].get<double>());
            if (!gripper->SetTargetForce(force))
                log_err(std::string(context) + " gripper set force failed");
            handled = true;
        }
        if (ee.contains("speed") && ee["speed"].is_number())
        {
            int speed = static_cast<int>(ee["speed"].get<double>());
            if (!gripper->SetTargetSpeed(speed))
                log_err(std::string(context) + " gripper set speed failed");
            handled = true;
        }

        if (!handled)
        {
            log_err(std::string(context) + " gripper missing/invalid fields");
            return;
        }

        int curpos_raw = 0;
        if (gripper->GetCurrentPosition(curpos_raw))
        {
            g_gripper_position.store(static_cast<double>(curpos_raw) / 1000.0,
                                     std::memory_order_relaxed);
        }
        else
        {
            log_err(std::string(context) + " gripper read position failed");
        }
        return;
    }

    log_err(std::string(context) + " no bound end effector for robot_index=" + std::to_string(robot_index));
}

static void save_data()
{
    HYYRobotBase::RTimer timer;
    HYYRobotBase::initUserTimer(&timer,1,1);
    double data[14];
    while (!is_stop.load())
    {
        HYYRobotBase::userTimer(&timer);
        memset(data,0,sizeof(data));
        HYYRobotBase::GetCurrentJoint(data, 0);
        HYYRobotBase::GetCurrentLastTargetJoint(&(data[7]), 0);
        HYYRobotBase::RSaveDataFast1("jeserver",1, 100, 14, data );
    }
}

static void publisher_loop()
{
    HYYRobotBase::RTimer timer;
    HYYRobotBase::initUserTimer(&timer, 0, CYCLIE); // 10ms
    nlohmann::ordered_json data;

    // 参照示例：拿到 device_name，后续按 i 拿 robot_name（子设备名）
    const char* device_name = HYYRobotBase::get_deviceName(0, nullptr);

    printf("start publisher_loop\n");
    while (true)
    {
        HYYRobotBase::userTimer(&timer);

        int rn = HYYRobotBase::robot_getNUM();
        for (int i = 0; i < rn; i++)
        {
            const std::string rk = std::string("Robot") + std::to_string(i);

            data[rk]["MoveState"]  = HYYRobotBase::get_robot_move_state(i);
            data[rk]["PowerState"] = HYYRobotBase::GetRobotPowerState(i);

            int dof = HYYRobotBase::robot_getDOF(i);

            // -------- Joint position / target --------
            std::vector<double> joint(dof);
            std::vector<double> target_joint(dof);

            // 修正：使用 i（对应每台机器人）
            HYYRobotBase::GetCurrentJoint(joint.data(), i);
            data[rk]["Joint"] = joint;

            HYYRobotBase::GetCurrentLastTargetJoint(target_joint.data(), i);
            data[rk]["TargetJoint"] = target_joint;

            // -------- Joint velocity / torque (NEW) --------
            // 参照示例：获取该 robot 的子设备名
            const char* robot_name = HYYRobotBase::get_name_robot_device(device_name, i);

            std::vector<double> joint_vel(dof, 0.0);
            std::vector<double> joint_torque(dof, 0.0);
            std::vector<double> joint_sensor_torque(dof, 0.0);

            int vel_ret = 0;
            int tq_ret  = 0;
            int stq_ret = 0;
            if (robot_name != nullptr)
            {
                vel_ret = HYYRobotBase::GetGroupVelocity(robot_name, joint_vel.data());
                tq_ret  = HYYRobotBase::GetGroupTorque(robot_name, joint_torque.data());
                stq_ret = HYYRobotBase::GetGroupSensorTorque(robot_name, joint_sensor_torque.data());
            }
            else
            {
                vel_ret = -1;
                tq_ret  = -1;
                stq_ret = -1;

            }

            // 失败时仍发布零向量，并打印日志（避免 JSON 缺字段导致下游解析不一致）
            if (vel_ret < 0)
            {
                std::cerr << "[publisher_loop] GetGroupVelocity failed, robot_index="
                          << i << ", robot_name=" << (robot_name ? robot_name : "null")
                          << ", ret=" << vel_ret << std::endl;
            }
            if (tq_ret < 0)
            {
                std::cerr << "[publisher_loop] GetGroupTorque failed, robot_index="
                          << i << ", robot_name=" << (robot_name ? robot_name : "null")
                          << ", ret=" << tq_ret << std::endl;
            }
            if (stq_ret < 0)
            {
                std::cerr << "[publisher_loop] GetGroupSensorTorque failed, robot_index="
                          << i << ", robot_name=" << (robot_name ? robot_name : "null")
                          << ", ret=" << stq_ret << std::endl;
            }

            data[rk]["JointVelocity"] = joint_vel;
            data[rk]["JointTorque"]   = joint_torque;
            data[rk]["JointSensorTorque"] = joint_sensor_torque;

            // -------- Cartesian / target --------
            std::vector<double> Cartesian(6);

            // 修正：使用 i（对应每台机器人）
            HYYRobotBase::GetCurrentCartesian(NULL, NULL, (HYYRobotBase::robpose*)Cartesian.data(), i);
            data[rk]["Cartesian"] = Cartesian;

            HYYRobotBase::GetCurrentLastTargetCartesian(NULL, NULL, (HYYRobotBase::robpose*)Cartesian.data(), i);
            data[rk]["TargetCartesian"] = Cartesian;

            // -------- Gripper state (cached from control thread) --------
            if (i == kRobotIndexGripper)
            {
                double gripper_pos = g_gripper_position.load(std::memory_order_relaxed);
                data[rk]["EndEffector"]["CurrentPosition"] = gripper_pos;
                // data[rk]["EndEffector"]["Valid"] = (gripper_pos >= 0.0);
            }
        }

        publisher.send(zmq::buffer("State " + data.dump()));
    }
}

static void subscriber_loop()
{
    // 2 增加落盘
    // static std::ofstream cart_log("cartesian_log.csv", std::ios::out | std::ios::app);
    printf("start subscriber_loop\n");
    while(true)
    {
        zmq::message_t msg;
        subscriber.recv(msg);
        std::string cmd(static_cast<char*>(msg.data()), msg.size());
        auto pos = cmd.find(' ');
        std::string topic = cmd.substr(0, pos);
        nlohmann::ordered_json cmd_json = nlohmann::json::parse(cmd.substr(pos + 1));
#if JOINT_DEBUG_LOG
        std::cout<<cmd_json.dump(4)<<std::endl;
        std::cout<<topic<<std::endl;
#endif
        int rn=HYYRobotBase::robot_getNUM();
        if ("Switch"==topic)
        {
            if (cmd_json.contains("Switch"))
            {
                HYYRobotBase::ClearRobotError();
                if (cmd_json["Switch"].get<bool>())
                {

                    for (int i=0;i<rn;i++)
                    {
                        is_stop.store(true);
                        HYYRobotBase::RobotStopRecover(i);
                        HYYRobotBase::ServoEnd(i);
                        HYYRobotBase::RobotPoweroff(i);
                        usleep(100000);
                        HYYRobotBase::RobotPower(i);
                        HYYRobotBase::ServoStart(1,0.0001, i);
                        is_stop.store(false);
                        stop_th=new std::thread(save_data);
                        stop_th->detach();
                    }
                }
                else
                {
                    for (int i=0;i<rn;i++)
                    {
                        is_stop.store(true);
                        HYYRobotBase::ServoEnd(i);
                        HYYRobotBase::RobotPoweroff(i);
                    } 
                }
            } 
        }else if("Joint"==topic)
        {
            JLOG("[Joint] received joint: rn=" << rn);

            auto vec_to_string = [](const std::vector<double>& v) -> std::string {
                std::ostringstream oss;
                oss << "[";
                for (size_t k = 0; k < v.size(); ++k) {
                    oss << std::fixed << std::setprecision(6) << v[k];
                    if (k + 1 < v.size()) oss << ", ";
                }
                oss << "]";
                return oss.str();
            };

            for (int i = 0; i < rn; i++)
            {
                const std::string rk = std::string("Robot") + std::to_string(i);

                if (!cmd_json.contains(rk)) {
                    JERR("[Joint] " << rk << " not found in cmd_json (skip). keys=" << cmd_json.dump());
                    continue;
                }

                // 打印该 robot 的原始片段（必要时你也可以注释掉，避免太多输出）
                JLOG("[Joint] " << rk << " payload=" << cmd_json[rk].dump());

                if (!cmd_json[rk].contains("time") || !cmd_json[rk]["time"].is_number())
                {
                    JERR("[Joint] missing/invalid time for " << rk << ", payload=" << cmd_json[rk].dump());
                    continue;
                }
                if (!cmd_json[rk].contains("joint") || !cmd_json[rk]["joint"].is_array())
                {
                    JERR("[Joint] missing/invalid joint for " << rk << ", payload=" << cmd_json[rk].dump());
                    continue;
                }

                double time = cmd_json[rk]["time"].get<double>();
                std::vector<double> joint = cmd_json[rk]["joint"].get<std::vector<double>>();

                const int dof = HYYRobotBase::robot_getDOF(i);

                JLOG("[Joint] " << rk
                    << " time=" << std::fixed << std::setprecision(6) << time
                    << " dof=" << dof
                    << " joint.size=" << joint.size()
                    << " joint=" << vec_to_string(joint));

                if (static_cast<int>(joint.size()) < dof) {
                    JERR("[Joint] " << rk << " joint.size < dof, will still call init_robjoint (risk).");
                } else if (static_cast<int>(joint.size()) > dof) {
                    JERR("[Joint] " << rk << " joint.size > dof, extra elements will be ignored by init_robjoint? (please confirm).");
                }

                HYYRobotBase::robjoint jt;
                HYYRobotBase::init_robjoint(&jt, joint.data(), dof);

                JLOG("[Joint] " << rk << " call ServoJoint(time=" << std::fixed << std::setprecision(6) << time
                    << ", idx=" << i << ")");
                HYYRobotBase::ServoJoint(&jt, time, i);
                JLOG("[Joint] " << rk << " ServoJoint done.");

                if (cmd_json[rk].contains("EndEffector"))
                {
                    const auto& ee = cmd_json[rk]["EndEffector"];
                    JLOG("[Joint] " << rk << " EndEffector payload=" << ee.dump());
                    handle_bound_end_effector(i, ee, "[Joint]", true);
                }
                else
                {
                    JLOG("[Joint] " << rk << " EndEffector not present.");
                }
            }
        }
        else if ("Cartesian"==topic)
        {
            // std::cout << "received joint: " << std::endl;
            for (int i=0;i<rn;i++)
            {
                const std::string rk = std::string("Robot")+std::to_string(i);
                if (cmd_json.contains(rk))
                {
                    if (!cmd_json[rk].contains("time") || !cmd_json[rk]["time"].is_number())
                    {
                        std::cerr << "[Cartesian] missing/invalid time for " << rk << "\n";
                        continue;
                    }
                    if (!cmd_json[rk].contains("cartesian") || !cmd_json[rk]["cartesian"].is_array())
                    {
                        std::cerr << "[Cartesian] missing/invalid cartesian for " << rk << "\n";
                        continue;
                    }
                    double time=cmd_json[rk]["time"].get<double>();
                    std::vector<double> cartesian=cmd_json[rk]["cartesian"].get<std::vector<double>>();

                    // // ③ 增加：落盘（CSV：robot_id,time,c0,c1,...）
                    // cart_log << i << "," << time;
                    // for (double v : cartesian) cart_log << "," << v;
                    // cart_log << "\n";
                    // cart_log.flush();  // 最简单：每条都刷盘

                    HYYRobotBase::robpose pt;
                    HYYRobotBase::init_robpose(&pt,cartesian.data(),cartesian.data()+3);
                    HYYRobotBase::ServoCartesian(&pt,time,NULL,NULL,i);

                    if (cmd_json[rk].contains("EndEffector"))
                    {
                        const auto& ee = cmd_json[rk]["EndEffector"];
                        handle_bound_end_effector(i, ee, "[Cartesian]", false);
                    }
                }
            }
        }
    }
}

void PluginMain()
{
    SerialOptions ee_serial_opts(0, 5, SerialOptions::kProfileJEServerLegacy);
    end_effector_device = std::unique_ptr<EndEffectorDevice>(
        new EndEffectorDevice(kEndEffectorPort, kEndEffectorBaud, ee_serial_opts));
    if (!end_effector_device->Open())
    {
        std::cerr << "end effector serial init failed, end effector disabled\n";
        end_effector_device.reset();
    }

    dh_gripper = std::unique_ptr<DH_Modbus_Gripper>(
        new DH_Modbus_Gripper(kGripperId, kGripperPort, kGripperBaud));
    if (dh_gripper->open() < 0)
    {
        std::cerr << "DH gripper open failed, gripper disabled\n";
        dh_gripper.reset();
    }
    else
    {
        int initstate = 0;
        bool ok = dh_gripper->GetInitState(initstate);
        std::cout << "GetInitState ok=" << ok << " state=" << initstate << std::endl;

        if (initstate != DH_Modbus_Gripper::S_INIT_FINISHED)
        {
            dh_gripper->Initialization();
            std::cout << "Trying to init gripper" << std::endl;

            const int max_try = 40; // 200 * 50ms = 10s
            for (int i = 0; i < max_try; ++i)
            {
                ok = dh_gripper->GetInitState(initstate);
                std::cout << "GetInitState ok=" << ok << " state=" << initstate << std::endl;
                if (ok && initstate == DH_Modbus_Gripper::S_INIT_FINISHED)
                    break;
                usleep(50000);
            }
            std::cout << "Gripper init succeeded, " << initstate << std::endl;
        }
    }
    publisher.set(zmq::sockopt::sndhwm, 0);  // 0 表示无限小队列，但行为是：不能缓存
    publisher.set(zmq::sockopt::immediate, 1);  // SUB 未连接时直接丢弃
    publisher.bind("tcp://*:8000");
    pub_th=new std::thread(publisher_loop);
    pub_th->detach();
    subscriber.connect("tcp://192.168.0.35:8001");
    // subscriber.set(zmq::sockopt::subscribe, "");
    subscriber.set(zmq::sockopt::rcvhwm, 1);
    subscriber.set(zmq::sockopt::conflate, 1);   // 只保留最后一条
    subscriber.set(zmq::sockopt::subscribe, "Switch ");
    subscriber.set(zmq::sockopt::subscribe, "Cartesian ");  // 只收你需要的
    subscriber.set(zmq::sockopt::subscribe, "Joint ");  // 只收你需要的
    sub_th=new std::thread(subscriber_loop);
    sub_th->detach();
}
