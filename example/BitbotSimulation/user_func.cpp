/**
 * @file user_func.cpp
 * @author Zishun Zhou
 * @brief
 * @date 2025-03-10
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "user_func.h"

#include <chrono>
#define _USE_MATH_DEFINES
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>  // std::cout
#include <memory>
#include <nlohmann/json.hpp>
#include <thread>

#include "types.hpp"

template <typename Container, typename Func>
void Apply(Container& container, Func func) {
  for (size_t i = 0; i < container.size(); ++i) {
    func(&container[i], i);
  }
}
void ConfigFunc(const KernelBus& bus, UserData& d) {
  // 读取json配置文件,并初始化各个worker
  nlohmann::json cfg_root;
  nlohmann::json cfg_workers;
  {
    // NOTE: 注意将配置文件路径修改为自己的路径
    std::string path =
        PROJECT_ROOT_DIR + std::string("/config_Unitree_AMP.json");
    std::ifstream cfg_file(path);
    cfg_root = nlohmann::json::parse(cfg_file, nullptr, true, true);
    cfg_workers = cfg_root["Workers"];
  }
  d.ImuPtr = bus.GetDevice<DeviceImu>(IMU_ID_MAP).value();
  d.ImuAlterPtr = bus.GetDevice<DeviceImu>(ALTER_IMU_ID_MAP).value();
  Apply(d.JointsPtr, [&bus](DeviceJoint** joint, size_t i) {
    *joint = bus.GetDevice<DeviceJoint>(JOINT_ID_MAP[i]).value();
  });

  // 创建调度器，创建数据总线
  d.TaskScheduler = SchedulerType::Create(cfg_root["Scheduler"]);

  // 获取14个关节的设备指针
  // std::array<DeviceJoint*, 14> joints = {
  //     bus.GetDevice<DeviceJoint>(0).value(),
  //     bus.GetDevice<DeviceJoint>(1).value(),
  //     bus.GetDevice<DeviceJoint>(2).value(),
  //     bus.GetDevice<DeviceJoint>(3).value(),
  //     bus.GetDevice<DeviceJoint>(4).value(),
  //     bus.GetDevice<DeviceJoint>(5).value(),
  //     bus.GetDevice<DeviceJoint>(6).value(),
  //     bus.GetDevice<DeviceJoint>(7).value(),
  //     bus.GetDevice<DeviceJoint>(8).value(),
  //     bus.GetDevice<DeviceJoint>(9).value(),
  //     bus.GetDevice<DeviceJoint>(10).value(),
  //     bus.GetDevice<DeviceJoint>(11).value(),
  //     bus.GetDevice<DeviceJoint>(12).value(),
  //     bus.GetDevice<DeviceJoint>(13).value(),
  // };

  // 创建各个worker
  d.ImuWorker = d.TaskScheduler->template CreateWorker<ImuWorkerType>(
      d.ImuPtr, cfg_workers["ImuProcess"]);
  d.MotorWorker = d.TaskScheduler->template CreateWorker<MotorWorkerType>(
      cfg_workers["MotorControl"], d.JointsPtr);
  d.MotorPDWorker = d.TaskScheduler->template CreateWorker<MotorPDWorkerType>(
      cfg_workers["MotorPDLoop"]);
  d.Logger = d.TaskScheduler->template CreateWorker<LoggerWorkerType>(
      cfg_workers["AsyncLogger"]);
  d.CommanderWorker = d.TaskScheduler->template CreateWorker<CmdWorkerType>(
      cfg_workers["Commander"]);
  d.ActionManagementWorker =
      d.TaskScheduler->template CreateWorker<ActionManagementWorkerType>(
          cfg_workers["ActionManager"]);
  d.AlterImuWorker = d.TaskScheduler->template CreateWorker<AlterImuWorkerType>(
      [&d](SchedulerType::Ptr scheduler) {
        RealNumber roll = d.ImuAlterPtr->GetRoll();
        RealNumber pitch = d.ImuAlterPtr->GetPitch();
        RealNumber yaw = d.ImuAlterPtr->GetYaw();
        Vec3 euler_angles = Vec3({roll, pitch, yaw});
        scheduler->template SetData<"AlterAngleValue">(euler_angles);
      });
  // 创建主任务MainTask列表，并添加worker
  d.TaskScheduler->CreateTaskList("MainTask", 1, true);
  // // BHR机器人
  // d.TaskScheduler->AddWorkers("MainTask",
  //                             {d.ImuWorker, d.MotorPDWorker, d.MotorWorker});
  // 宇树机器人
  d.TaskScheduler->AddWorkers("MainTask",
                              {d.ImuWorker, d.AlterImuWorker, d.MotorWorker});

  // 创建推理任务列表，并添加worker，设置推理任务频率
  d.NetInferWorker =
      d.TaskScheduler->template CreateWorker<AMPInferenceWorkerType>(
          cfg_workers["NN"], cfg_workers["MotorControl"]);
  // 推理任务
  d.TaskScheduler->CreateTaskList(
      "InferTask", cfg_root["Scheduler"]["InferTask"]["PolicyFrequency"]);

  d.TaskScheduler->AddWorkers("InferTask",
                              {d.CommanderWorker, d.NetInferWorker,
                               d.ActionManagementWorker, d.Logger});

  // 创建复位任务列表，并添加worker，设置复位任务频率为主任务频率的1/10
  d.MotorResetWorker =
      d.TaskScheduler->template CreateWorker<MotorResetWorkerType>(
          cfg_workers["MotorControl"], cfg_workers["ResetPosition"]);
  d.TaskScheduler->CreateTaskList("ResetTask", 10);
  d.TaskScheduler->AddWorker("ResetTask", d.MotorResetWorker);

  // 启用
  d.TaskScheduler->Start();
}

void FinishFunc(UserData& d) {}

std::optional<bitbot::StateId> EventInitPose(bitbot::EventValue value,
                                             UserData& d) {
  if (value == static_cast<bitbot::EventValue>(bitbot::KeyboardEvent::Up)) {
    d.MotorResetWorker->StartReset();  // 开始复位
    d.TaskScheduler->EnableTaskList(
        "ResetTask");  // 在复位任务列表中启用复位任务
    return static_cast<bitbot::StateId>(States::PF2InitPose);
  }
  return std::optional<bitbot::StateId>();
}

std::optional<bitbot::StateId> EventPolicyRun(bitbot::EventValue value,
                                              UserData& d) {
  if (value == static_cast<bitbot::EventValue>(bitbot::KeyboardEvent::Up)) {
    std::cout << "policy run\n";
    d.MotorResetWorker->StopReset();  // 停止复位
    d.TaskScheduler->DisableTaskList(
        "ResetTask");  // 在复位任务列表中禁用复位任务
    d.ActionManagementWorker->template SwitchTo<Net1OutPair>();
    d.TaskScheduler->EnableTaskList(
        "InferTask");  // 在推理任务列表中启用推理任务
    return static_cast<bitbot::StateId>(States::PF2PolicyRun);
  }
  return std::optional<bitbot::StateId>();
}

std::optional<bitbot::StateId> EventSystemTest(bitbot::EventValue value,
                                               UserData& user_data) {
  if (value == static_cast<bitbot::EventValue>(bitbot::KeyboardEvent::Up)) {
    return static_cast<bitbot::StateId>(States::PF2SystemTest);
  }
  return std::optional<bitbot::StateId>();
}

std::optional<bitbot::StateId> EventVeloXIncrease(bitbot::EventValue keyState,
                                                  UserData& d) {
  if (keyState == static_cast<bitbot::EventValue>(
                      bitbot::KeyboardEvent::Up)) {  // 设置x轴速度
    d.CommanderWorker->IncreaseCmd(0);
  }
  return std::optional<bitbot::StateId>();
}

std::optional<bitbot::StateId> EventVeloXDecrease(bitbot::EventValue keyState,
                                                  UserData& d) {
  if (keyState == static_cast<bitbot::EventValue>(
                      bitbot::KeyboardEvent::Up)) {  // 设置x轴速度
    d.CommanderWorker->DecreaseCmd(0);
  }
  return std::optional<bitbot::StateId>();
}

std::optional<bitbot::StateId> EventVeloYIncrease(bitbot::EventValue keyState,
                                                  UserData& d) {
  if (keyState == static_cast<bitbot::EventValue>(bitbot::KeyboardEvent::Up)) {
    d.CommanderWorker->IncreaseCmd(1);
  }
  return std::optional<bitbot::StateId>();
}

std::optional<bitbot::StateId> EventVeloYDecrease(bitbot::EventValue keyState,
                                                  UserData& d) {
  if (keyState == static_cast<bitbot::EventValue>(bitbot::KeyboardEvent::Up)) {
    d.CommanderWorker->DecreaseCmd(1);
  }
  return std::optional<bitbot::StateId>();
}

std::optional<bitbot::StateId> EventVeloYawIncrease(bitbot::EventValue keyState,
                                                    UserData& d) {
  if (keyState == static_cast<bitbot::EventValue>(bitbot::KeyboardEvent::Up)) {
    d.CommanderWorker->IncreaseCmd(2);
  }
  return std::optional<bitbot::StateId>();
}

std::optional<bitbot::StateId> EventVeloYawDecrease(bitbot::EventValue keyState,
                                                    UserData& d) {
  if (keyState == static_cast<bitbot::EventValue>(bitbot::KeyboardEvent::Up)) {
    d.CommanderWorker->DecreaseCmd(2);
  }
  return std::optional<bitbot::StateId>();
}

std::optional<bitbot::StateId> EventJoystickXChange(bitbot::EventValue keyState,
                                                    UserData& d) {
  double vel = static_cast<double>(keyState / 32768.0);
  d.CommanderWorker->SetCmd(0, static_cast<RealNumber>(vel));
  return std::optional<bitbot::StateId>();
}

std::optional<bitbot::StateId> EventJoystickYChange(bitbot::EventValue keyState,
                                                    UserData& d) {
  double vel = static_cast<double>(keyState / 32768.0);
  d.CommanderWorker->SetCmd(1, static_cast<RealNumber>(vel));
  return std::optional<bitbot::StateId>();
}

std::optional<bitbot::StateId> EventJoystickYawChange(
    bitbot::EventValue keyState, UserData& d) {
  double vel = static_cast<double>(keyState / 32768.0);
  d.CommanderWorker->SetCmd(2, static_cast<RealNumber>(vel));
  return std::optional<bitbot::StateId>();
}

void StateWaiting(const bitbot::KernelInterface& kernel,
                  bitbot::ExtraData& extra_data, UserData& d) {
  // 空闲等待状态，重置目标位置防止突变
  d.MotorWorker->SetCurrentPositionAsTargetPosition();
  // 在具体状态中只需要调用调度器的SpinOnce函数即可，调度器会根据设置自动调度任务
  d.TaskScheduler->SpinOnce();
}

void StateSystemTest(const bitbot::KernelInterface& kernel,
                     bitbot::ExtraData& extra_data, UserData& user_data) {}

void StatePolicyRun(const bitbot::KernelInterface& kernel,
                    bitbot::ExtraData& extra_data, UserData& d) {
  d.TaskScheduler->SpinOnce();
};

void StateJointInitPose(const bitbot::KernelInterface& kernel,
                        bitbot::ExtraData& extra_data, UserData& d) {
  d.TaskScheduler->SpinOnce();
}
