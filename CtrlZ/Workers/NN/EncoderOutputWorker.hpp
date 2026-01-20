/**
 * @file EncoderOutputWorker.hpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2025-12-06
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <chrono>
#include <cmath>

#include "CommonLocoInferenceWorker.hpp"
#include "NetInferenceWorker.h"
#include "Utils/StaticStringUtils.hpp"
#include "Utils/ZenBuffer.hpp"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace z {
/**
 * @brief UniFPInferenceWorker
 * 类型是一个人形机器人推理工人类型，该类实现了UniFP网络兼容的推理功能。
 * @details UniFPInferenceWorker
 * 类型是一个人形机器人推理工人类型，该类实现了UniFP网络兼容的推理功能。
 * UniFP参见[https://github.com/unified-force/UniFP](https://github.com/unified-force/UniFP)
 *
 * @details config.json配置文件示例：
 * @code {.json}
 * {
 *  "Workers": {
 *     "NN": {
 *        "NetWork":{
 *           "Cycle_time": 0.63 //步频周期
 *       }
 *     }
 *   }
 * }
 * @endcode
 *
 * @tparam SchedulerType 调度器类型
 * @tparam NetName 网络名称，用户可以通过这个参数来指定网络的名称,
 * 这在有多个网络时可以区分数据总线上的不同网络数据
 * @tparam InferencePrecision
 * 推理精度，用户可以通过这个参数来指定推理的精度，比如可以指定为float或者double
 * @tparam INPUT_STUCK_LENGTH UniFP网络的Actor输入堆叠长度
 * @tparam JOINT_NUMBER 关节数量
 */
template <typename SchedulerType, CTString NetName, typename InferencePrecision,
          size_t INPUT_STUCK_LENGTH, size_t JOINT_NUMBER>
class EncoderOutputWorker
    : public CommonLocoInferenceWorker<SchedulerType, NetName,
                                       InferencePrecision, JOINT_NUMBER> {
 public:
  using MotorValVec = math::Vector<InferencePrecision, JOINT_NUMBER>;
  using ValVec3 = math::Vector<InferencePrecision, 3>;
  using ValVec6 = math::Vector<InferencePrecision, 6>;

 public:
  /**
   * @brief 构造一个EncoderOutputWorker类型，构造构造函数
   *
   * @param scheduler 调度器的指针
   * @param Net_cfg 网络配置
   * @param Motor_cfg 电机配置
   */
  EncoderOutputWorker(SchedulerType::Ptr scheduler,
                      const nlohmann::json& Net_cfg,
                      const nlohmann::json& Motor_cfg)
      : CommonLocoInferenceWorker<SchedulerType, NetName, InferencePrecision,
                                  JOINT_NUMBER>(scheduler, Net_cfg, Motor_cfg),
        GravityVector({0.0, 0.0, -1.0}),
        HistoryInputBuffer(INPUT_STUCK_LENGTH) {
    // 读取通用配置
    nlohmann::json InferenceCfg = Net_cfg["Inference"];
    nlohmann::json NetworkCfg = Net_cfg["Network"];
    this->cycle_time = NetworkCfg["Cycle_time"].get<InferencePrecision>();
    this->dt = scheduler->getSpinOnceTime();

    // 读取Encoder专用的节点名称配置
    if (Net_cfg.contains("Encoder")) {
      nlohmann::json EncoderCfg = Net_cfg["Encoder"];
      // 设置Encoder的输入输出节点名称
      this->InputNodeNames =
          EncoderCfg["InputNodeNames"].get<std::vector<std::string>>();
      this->OutputNodeNames =
          EncoderCfg["OutputNodeNames"].get<std::vector<std::string>>();
    }
    // 打印初始化信息到控制台，显示关键参数
    this->PrintSplitLine();
    std::cout << "EncoderOutputWorker" << std::endl;
    std::cout << "JOINT_NUMBER=" << JOINT_NUMBER << std::endl;
    std::cout << "Cycle_time=" << this->cycle_time << std::endl;
    std::cout << "dt=" << this->dt << std::endl;
    this->PrintSplitLine();

    // concatenate all scales
    auto clock_scales = math::Vector<InferencePrecision, 2>::ones();
    this->InputScaleVec = math::cat(
        this->Scales_project_gravity, this->Scales_ang_vel,
        this->Scales_dof_pos, this->Scales_dof_vel, this->Scales_last_action,
        clock_scales, this->Scales_command6);
    this->OutputScaleVec = this->ActionScale;

    // warp input tensor
    this->InputOrtTensors__.push_back(this->WarpOrtTensor(InputTensor));
    this->OutputOrtTensors__.push_back(this->WarpOrtTensor(OutputTensor));
  }
  /**
   * @brief 析构函数
   *
   */
  virtual ~EncoderOutputWorker() {}

  /**
   * @brief
   * 推理前的准备工作，主要将数据从总线中读取出来，并将数据缩放到合适的范围
   * 构造堆叠的输入数据，并准备好输入张量
   */
  void PreProcess() override {
    this->start_time = std::chrono::steady_clock::now();

    MotorValVec CurrentMotorVel;
    this->Scheduler->template GetData<"CurrentMotorVelocity">(CurrentMotorVel);

    MotorValVec CurrentMotorPos;
    this->Scheduler->template GetData<"CurrentMotorPosition">(CurrentMotorPos);
    CurrentMotorPos -= this->JointDefaultPos;

    MotorValVec LastAction;
    this->Scheduler->template GetData<concat(NetName, "NetLastAction")>(
        LastAction);

    ValVec3 UserCmd3;
    this->Scheduler->template GetData<concat(NetName, "NetUserCommand3")>(
        UserCmd3);

    ValVec6 UserCmd6;
    this->Scheduler->template GetData<concat(NetName, "NetUserCommand6")>(
        UserCmd6);

    ValVec3 LinVel;
    this->Scheduler->template GetData<"LinearVelocityValue">(LinVel);

    ValVec3 AngVel;
    this->Scheduler->template GetData<"AngleVelocityValue">(AngVel);

    ValVec3 Ang;
    this->Scheduler->template GetData<"AngleValue">(Ang);

    size_t t = this->Scheduler->getTimeStamp();
    InferencePrecision phase =
        this->dt * static_cast<InferencePrecision>(t) / this->cycle_time;
    InferencePrecision clock_sin = std::sin(phase * 2 * M_PI);
    InferencePrecision clock_cos = std::cos(phase * 2 * M_PI);
    z::math::Vector<InferencePrecision, 2> ClockVector = {clock_sin, clock_cos};
    this->Scheduler->template SetData<concat(NetName, "NetClockVector")>(
        ClockVector);

    auto SingleInputVecScaled =
        math::cat(ClockVector, UserCmd3, CurrentMotorPos, CurrentMotorVel,
                  LastAction, AngVel, Ang) *
        this->InputScaleVec;

    this->HistoryInputBuffer.push(SingleInputVecScaled);

    math::Vector<InferencePrecision, INPUT_TENSOR_LENGTH> InputVec;
    for (size_t i = 0; i < INPUT_STUCK_LENGTH; i++) {
      std::copy(this->HistoryInputBuffer[i].begin(),
                this->HistoryInputBuffer[i].end(),
                InputVec.begin() + i * INPUT_TENSOR_LENGTH_UNIT);
    }
    this->InputTensor.Array() = decltype(InputVec)::clamp(
        InputVec, -this->ClipObservation, this->ClipObservation);
  }

  /**
   * @brief
   * 推理后的处理工作，主要是将推理的结构从数据总线中读取出来，并将数据缩放到合适的范围
   */
  void PostProcess() override {
    auto LastAction = this->OutputTensor.toVector();
    auto ClipedLastAction =
        MotorValVec::clamp(LastAction, -this->ClipAction, this->ClipAction);
    this->Scheduler->template SetData<concat(NetName, "NetLastAction")>(
        ClipedLastAction);

    auto ScaledAction =
        ClipedLastAction * this->OutputScaleVec + this->JointDefaultPos;
    this->Scheduler->template SetData<concat(NetName, "NetScaledAction")>(
        ScaledAction);

    auto clipedAction = MotorValVec::clamp(ScaledAction, this->JointClipLower,
                                           this->JointClipUpper);
    this->Scheduler->template SetData<concat(NetName, "Action")>(clipedAction);

    auto latent_vector = this->OutputTensor.toVector();
    this->Scheduler->template SetData<concat(NetName, "LatentVariables")>(
        latent_vector);

    this->end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        this->end_time - this->start_time);
    InferencePrecision inference_time =
        static_cast<InferencePrecision>(duration.count());
    this->Scheduler->template SetData<concat(NetName, "InferenceTime")>(
        inference_time);
  }

 private:
  // cat
  static constexpr size_t INPUT_TENSOR_LENGTH_UNIT =
      2 + 3 + JOINT_NUMBER + JOINT_NUMBER + JOINT_NUMBER + 3 + 3;
  static constexpr size_t INPUT_TENSOR_LENGTH =
      INPUT_TENSOR_LENGTH_UNIT * INPUT_STUCK_LENGTH;

  // joint number
  static constexpr size_t OUTPUT_TENSOR_LENGTH = JOINT_NUMBER;

  // input tensor
  z::math::Tensor<InferencePrecision, 1, INPUT_TENSOR_LENGTH> InputTensor;
  z::math::Vector<InferencePrecision, INPUT_TENSOR_LENGTH_UNIT> InputScaleVec;
  z::RingBuffer<z::math::Vector<InferencePrecision, INPUT_TENSOR_LENGTH_UNIT>>
      HistoryInputBuffer;
  // output tensor
  z::math::Tensor<InferencePrecision, 1, OUTPUT_TENSOR_LENGTH> OutputTensor;
  z::math::Vector<InferencePrecision, OUTPUT_TENSOR_LENGTH> OutputScaleVec;

  /// @brief 重力向量{0,0,-1}
  const ValVec3 GravityVector;

  // cycle time and dt
  InferencePrecision cycle_time;
  InferencePrecision dt;

  // compute time
  std::chrono::steady_clock::time_point start_time;
  std::chrono::steady_clock::time_point end_time;
};
};  // namespace z