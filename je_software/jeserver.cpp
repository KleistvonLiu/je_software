#include "HYYRobotInterface.h"
#include "device_interface.h"
#include "end_effector_manager.h"
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
// #include <chrono>
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
static EndEffectorManager g_end_effector_manager;

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

            EndEffectorSlotState slot_state = g_end_effector_manager.GetSlotState(i);
            data[rk]["EndEffector"]["CurrentPosition"] = slot_state.current_position;
        }

        std::vector<EndEffectorSlotState> slot_states = g_end_effector_manager.GetAllStates();
        data["EndEffectors"] = nlohmann::ordered_json::array();
        for (size_t idx = 0; idx < slot_states.size(); ++idx)
        {
            nlohmann::ordered_json slot_json;
            slot_json["slot_index"] = slot_states[idx].slot_index;
            slot_json["type"] = slot_states[idx].type;
            slot_json["ready"] = slot_states[idx].ready;
            slot_json["current_position"] = slot_states[idx].current_position;
            data["EndEffectors"].push_back(slot_json);
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
            // std::cout << "Received one msg with time: " << cmd_json["Robot0"]["time"].get<double>() << "\n";

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
#if !JOINT_DEBUG_LOG
            (void)vec_to_string;
#endif

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
                    g_end_effector_manager.DispatchByRobotIndex(i, ee, "[Joint]", true);
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
                        g_end_effector_manager.DispatchByRobotIndex(i, ee, "[Cartesian]", false);
                    }
                }
            }
        }
        else if ("MoveA"==topic)
        {
            // 固定格式：
            // MoveA {"Robot0":{"joint":[j1,j2,j3,j4,j5,j6,j7],"speed":0.2}}
            std::vector<double> joint = cmd_json["Robot0"]["joint"].get<std::vector<double>>();
            double speed_value = cmd_json["Robot0"]["speed"].get<double>();

            int dof = HYYRobotBase::robot_getDOF(0);
            HYYRobotBase::robjoint jt;
            HYYRobotBase::init_robjoint(&jt, joint.data(), dof);

            std::vector<double> joint_speed(dof, speed_value);
            HYYRobotBase::speed sp;
            HYYRobotBase::init_speed(&sp, joint_speed.data(), dof, 1, speed_value, speed_value, 1);

            HYYRobotBase::MoveA(&jt, &sp, NULL, NULL, NULL);

            if (cmd_json["Robot0"].contains("EndEffector"))
            {
                const auto& ee = cmd_json["Robot0"]["EndEffector"];
                g_end_effector_manager.DispatchByRobotIndex(0, ee, "[MoveA]", false);
            }
        }
        else if ("MoveL"==topic)
        {
            // 固定格式：
            // MoveL {"Robot0":{"cartesian":[x,y,z,rx,ry,rz],"rot_speed":0.5,"tra_speed":0.1}}
            std::vector<double> cartesian = cmd_json["Robot0"]["cartesian"].get<std::vector<double>>();
            double rot_speed = cmd_json["Robot0"]["rot_speed"].get<double>();
            double tra_speed = cmd_json["Robot0"]["tra_speed"].get<double>();

            HYYRobotBase::robpose pt;
            HYYRobotBase::init_robpose(&pt, cartesian.data(), cartesian.data() + 3);

            int dof = HYYRobotBase::robot_getDOF(0);
            std::vector<double> joint_speed(dof, rot_speed);
            HYYRobotBase::speed sp;
            HYYRobotBase::init_speed(&sp, joint_speed.data(), dof, 1, tra_speed, rot_speed, 1);

            HYYRobotBase::MoveL(&pt, &sp, NULL, NULL, NULL);

            if (cmd_json["Robot0"].contains("EndEffector"))
            {
                const auto& ee = cmd_json["Robot0"]["EndEffector"];
                g_end_effector_manager.DispatchByRobotIndex(0, ee, "[MoveL]", false);
            }
        }
    }
}

void PluginMain()
{
    const std::string kEndEffectorConfigPath = "/home/robot/Work/code/hyy_controller/config/jeserver_end_effectors.json";
    const int rn = HYYRobotBase::robot_getNUM();
    std::string ee_err;
    if (!g_end_effector_manager.LoadAndInit(kEndEffectorConfigPath, rn, &ee_err))
    {
        std::cerr << "end effector manager init failed: " << ee_err << std::endl;
        return;
    }

    publisher.set(zmq::sockopt::sndhwm, 0);  // 0 表示无限小队列，但行为是：不能缓存
    publisher.set(zmq::sockopt::immediate, 1);  // SUB 未连接时直接丢弃
    publisher.bind("tcp://*:8000");
    pub_th=new std::thread(publisher_loop);
    pub_th->detach();
    subscriber.connect("tcp://192.168.0.35:8001");
    subscriber.set(zmq::sockopt::rcvhwm, 1);
    subscriber.set(zmq::sockopt::conflate, 1);   // 只保留最后一条
    // subscriber.set(zmq::sockopt::subscribe, "");
    subscriber.set(zmq::sockopt::subscribe, "Switch ");
    subscriber.set(zmq::sockopt::subscribe, "Cartesian ");  // 只收你需要的
    subscriber.set(zmq::sockopt::subscribe, "Joint ");  // 只收你需要的
    subscriber.set(zmq::sockopt::subscribe, "MoveA ");  // 单臂MoveA
    subscriber.set(zmq::sockopt::subscribe, "MoveL ");  // 单臂MoveL
    sub_th=new std::thread(subscriber_loop);
    sub_th->detach();
}
