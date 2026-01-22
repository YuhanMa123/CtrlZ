/**
 * @file types.hpp
 * @author Zishun Zhou
 * @brief 类型定义
 * @date 2025-03-10
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include <array>

#include "Schedulers/AbstractScheduler.hpp"
#include "Utils/MathTypes.hpp"
#include "Utils/StaticStringUtils.hpp"
#include "Workers/AbstractWorker.hpp"
#include "Workers/ActionManagementWorker.hpp"
#include "Workers/AsyncLoggerWorker.hpp"
#include "Workers/ImuProcessWorker.hpp"
#include "Workers/MotorControlWorker.hpp"
#include "Workers/MotorResetPositionWorker.hpp"
// #include "Workers/NN/EncoderOutputWorker.hpp"
#include "Workers/NN/AMPInferenceWorker.hpp"
#include "Workers/NN/EraxLikeInferenceWorker.hpp"
#include "Workers/NN/HumanoidGymInferenceWorker.hpp"
#include "Workers/NN/UniFPInferenceWorker.hpp"
#include "Workers/NetCmdWorker.hpp"
#include "bitbot_mujoco/device/mujoco_imu.h"
#include "bitbot_mujoco/device/mujoco_joint.h"

using DeviceImu = bitbot::MujocoImu;
using DeviceJoint = bitbot::MujocoJoint;

/************ basic definintion***********/
using RealNumber = float;
constexpr size_t JOINT_NUMBER = 29;
using Vec3 = z::math::Vector<RealNumber, 3>;
using Vec6 = z::math::Vector<RealNumber, 6>;
using MotorVec = z::math::Vector<RealNumber, JOINT_NUMBER>;

constexpr std::array<size_t, JOINT_NUMBER> JOINT_ID_MAP = {
    0,  6,  12, 1,  7,  13, 2,  8,  14, 3,  9,  15, 22, 4, 10,
    16, 23, 5,  11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28};
constexpr size_t IMU_ID_MAP = 29;
constexpr size_t ALTER_IMU_ID_MAP = 30;
/********** IMU Data Pair******************/
constexpr z::CTSPair<"AccelerationRaw", Vec3> ImuAccRawPair;
constexpr z::CTSPair<"AngleVelocityRaw", Vec3> ImuGyroRawPair;
constexpr z::CTSPair<"AngleRaw", Vec3> ImuMagRawPair;

constexpr z::CTSPair<"AccelerationValue", Vec3> ImuAccFilteredPair;
constexpr z::CTSPair<"AngleValue", Vec3> ImuMagFilteredPair;
constexpr z::CTSPair<"AngleVelocityValue", Vec3> ImuGyroFilteredPair;
constexpr z::CTSPair<"AlterAngleValue", Vec3> ImuAlterAngleFilteredPair;

/********** Linear Velocity Pair ***********/
constexpr z::CTSPair<"LinearVelocityValue", Vec3> LinearVelocityValuePair;

/********** Motor control Pair ************/
constexpr z::CTSPair<"TargetMotorPosition", MotorVec> TargetMotorPosPair;
constexpr z::CTSPair<"TargetMotorVelocity", MotorVec> TargetMotorVelPair;
constexpr z::CTSPair<"TargetMotorTorque", MotorVec> TargetMotorTorquePair;
constexpr z::CTSPair<"CurrentMotorPosition", MotorVec> CurrentMotorPosPair;
constexpr z::CTSPair<"CurrentMotorVelocity", MotorVec> CurrentMotorVelPair;
constexpr z::CTSPair<"CurrentMotorTorque", MotorVec> CurrentMotorTorquePair;
constexpr z::CTSPair<"LimitTargetMotorTorque", MotorVec>
    LimitTargetMotorTorquePair;
constexpr z::CTSPair<"CurrentMotorPositionRaw", MotorVec>
    CurrentMotorPosRawPair;
constexpr z::CTSPair<"CurrentMotorVelocityRaw", MotorVec>
    CurrentMotorVelRawPair;

/********* NN pair ********************/
constexpr z::CTString Net1Name = "Net1";
constexpr z::CTSPair<z::concat(Net1Name, "NetLastAction"), MotorVec>
    NetLastActionPair;
constexpr z::CTSPair<z::concat(Net1Name, "NetUserCommand3"), Vec3>
    NetCommand3Pair;
constexpr z::CTSPair<z::concat(Net1Name, "NetUserCommand6"), Vec6>
    NetCommand6Pair;
constexpr z::CTSPair<z::concat(Net1Name, "NetProjectedGravity"), Vec3>
    NetProjectedGravityPair;
constexpr z::CTSPair<z::concat(Net1Name, "NetScaledAction"), MotorVec>
    NetScaledActionPair;
constexpr z::CTSPair<z::concat(Net1Name, "NetClockVector"),
                     z::math::Vector<RealNumber, 2>>
    NetClockVectorPair;
constexpr z::CTSPair<z::concat(Net1Name, "InferenceTime"), RealNumber>
    InferenceTimePair;
constexpr z::CTSPair<z::concat(Net1Name, "Action"), MotorVec> Net1OutPair;

// define scheduler
using SchedulerType = z::AbstractScheduler<
    ImuAccRawPair, ImuGyroRawPair, ImuMagRawPair, LinearVelocityValuePair,
    ImuAccFilteredPair, ImuGyroFilteredPair, ImuMagFilteredPair,
    TargetMotorPosPair, TargetMotorVelPair, CurrentMotorPosPair,
    CurrentMotorVelPair, CurrentMotorTorquePair, TargetMotorTorquePair,
    LimitTargetMotorTorquePair, CurrentMotorVelRawPair, CurrentMotorPosRawPair,
    NetLastActionPair, NetCommand3Pair, NetCommand6Pair,
    ImuAlterAngleFilteredPair, NetProjectedGravityPair, NetScaledActionPair,
    NetClockVectorPair, InferenceTimePair, Net1OutPair>;

// define workers
using MotorResetWorkerType =
    z::MotorResetPositionWorker<SchedulerType, RealNumber, JOINT_NUMBER>;
using ImuWorkerType =
    z::ImuProcessWorker<SchedulerType, DeviceImu*, RealNumber>;
using AlterImuWorkerType = z::SimpleCallbackWorker<SchedulerType>;
using MotorWorkerType = z::MotorControlWorker<SchedulerType, DeviceJoint*,
                                              RealNumber, JOINT_NUMBER>;
using MotorPDWorkerType =
    z::MotorPDControlWorker<SchedulerType, RealNumber, JOINT_NUMBER>;
using LoggerWorkerType = z::AsyncLoggerWorker<
    SchedulerType, RealNumber, ImuAccRawPair, ImuGyroRawPair, ImuMagRawPair,
    LinearVelocityValuePair, ImuAccFilteredPair, ImuGyroFilteredPair,
    ImuMagFilteredPair, TargetMotorPosPair, TargetMotorVelPair,
    CurrentMotorPosPair, CurrentMotorVelPair, CurrentMotorTorquePair,
    NetCommand3Pair, TargetMotorTorquePair, LimitTargetMotorTorquePair,
    NetLastActionPair, NetCommand6Pair, ImuAlterAngleFilteredPair,
    NetProjectedGravityPair, NetScaledActionPair, NetClockVectorPair,
    InferenceTimePair, Net1OutPair>;

using CmdWorkerType =
    z::NetCmdWorker<SchedulerType, RealNumber, NetCommand3Pair>;
// using CmdWorkerType =
//     z::NetCmdWorker<SchedulerType, RealNumber, NetCommand6Pair>;
using FlexPatchWorkerType = z::SimpleCallbackWorker<SchedulerType>;

using ActionManagementWorkerType =
    z::ActionManagementWorker<SchedulerType, RealNumber, Net1OutPair>;

/******define actor net************/
constexpr size_t OBSERVATION_STUCK_LENGTH = 10;
constexpr size_t OBSERVATION_EXTRA_LENGTH = 10;

// using HumanoidGymInferWorkerType =
// z::HumanoidGymInferenceWorker<SchedulerType,
// RealNumber,OBSERVATION_STUCK_LENGTH, JOINT_NUMBER>;

// using EraxLikeInferWorkerType =
//     z::EraxLikeInferenceWorker<SchedulerType, Net1Name, RealNumber,
//                                OBSERVATION_STUCK_LENGTH,
//                                OBSERVATION_EXTRA_LENGTH, JOINT_NUMBER>;

// using UniFPInferWorkerType =
//     z::UniFPInferenceWorker<SchedulerType, Net1Name, RealNumber,
//                             OBSERVATION_STUCK_LENGTH, JOINT_NUMBER>;
using AMPInferenceWorkerType =
    z::AMPInferenceWorker<SchedulerType, Net1Name, RealNumber,
                          OBSERVATION_STUCK_LENGTH, JOINT_NUMBER>;
