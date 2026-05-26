/**
 * @file AMPVAEInferenceWorker.hpp
 * @brief AMP-VAE dual-input inference worker for BitbotSimulation sim2sim.
 *
 * @date 2026-05-08
 */
#pragma once

#include <algorithm>
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

template <typename SchedulerType, CTString NetName, typename InferencePrecision,
          size_t HISTORY_LENGTH, size_t JOINT_NUMBER>
class AMPVAEInferenceWorker
    : public CommonLocoInferenceWorker<SchedulerType, NetName,
                                       InferencePrecision, JOINT_NUMBER> {
 public:
  static constexpr size_t ACTOR_OBS_LENGTH =
      3 + 3 + 3 + JOINT_NUMBER + JOINT_NUMBER + JOINT_NUMBER;
  static constexpr size_t HISTORY_FRAME_LENGTH = ACTOR_OBS_LENGTH;
  static constexpr size_t HISTORY_OBS_LENGTH =
      HISTORY_FRAME_LENGTH * HISTORY_LENGTH;
  static constexpr size_t OUTPUT_TENSOR_LENGTH = JOINT_NUMBER;
  static constexpr std::array<size_t, 6> HISTORY_TERM_SIZES = {
      3, 3, 3, JOINT_NUMBER, JOINT_NUMBER, JOINT_NUMBER};

  using MotorValVec = math::Vector<InferencePrecision, JOINT_NUMBER>;
  using ValVec3 = math::Vector<InferencePrecision, 3>;
  using ActorObsVec = math::Vector<InferencePrecision, ACTOR_OBS_LENGTH>;
  using HistoryFrameVec =
      math::Vector<InferencePrecision, HISTORY_FRAME_LENGTH>;
  using HistoryObsVec = math::Vector<InferencePrecision, HISTORY_OBS_LENGTH>;

 public:
  AMPVAEInferenceWorker(SchedulerType::Ptr scheduler,
                        const nlohmann::json& Net_cfg,
                        const nlohmann::json& Motor_cfg)
      : CommonLocoInferenceWorker<SchedulerType, NetName, InferencePrecision,
                                  JOINT_NUMBER>(scheduler, Net_cfg, Motor_cfg),
        GravityVector({0.0, 0.0, -1.0}),
        HistoryInputBuffer(HISTORY_LENGTH) {
    this->dt = scheduler->getSpinOnceTime();
    this->HistoryInputBuffer.flush();

    // Match the exported AMP-VAE deployment contract exactly:
    // actor_obs = [ang_vel, projected_gravity, cmd3, dof_pos_rel,
    //              dof_vel_rel, last_action]
    // obs_history uses the same per-frame feature order, then is flattened
    // according to Preprocess.HistoryLayout.
    this->CommandScaleVec = ValVec3::ones();
    this->ActorObsScaleVec = static_cast<ActorObsVec>(
        math::cat(this->Scales_ang_vel, this->Scales_project_gravity,
                  this->CommandScaleVec, this->Scales_dof_pos,
                  this->Scales_dof_vel, this->Scales_last_action));
    this->HistoryFrameScaleVec = static_cast<HistoryFrameVec>(
        math::cat(this->Scales_ang_vel, this->Scales_project_gravity,
                  this->CommandScaleVec, this->Scales_dof_pos,
                  this->Scales_dof_vel, this->Scales_last_action));
    this->OutputScaleVec = this->ActionScale;

    this->InputOrtTensors__.push_back(this->WarpOrtTensor(ActorObsTensor));
    this->InputOrtTensors__.push_back(this->WarpOrtTensor(HistoryObsTensor));
    this->OutputOrtTensors__.push_back(this->WarpOrtTensor(OutputTensor));

    this->PrintSplitLine();
    std::cout << "AMPVAEInferenceWorker" << std::endl;
    std::cout << "JOINT_NUMBER=" << JOINT_NUMBER << std::endl;
    std::cout << "ACTOR_OBS_LENGTH=" << ACTOR_OBS_LENGTH << std::endl;
    std::cout << "HISTORY_FRAME_LENGTH=" << HISTORY_FRAME_LENGTH << std::endl;
    std::cout << "HISTORY_LENGTH=" << HISTORY_LENGTH << std::endl;
    std::cout << "HISTORY_OBS_LENGTH=" << HISTORY_OBS_LENGTH << std::endl;
    std::cout << "CommandScaleVec=" << this->CommandScaleVec << std::endl;
    std::cout << "dt=" << this->dt << std::endl;
    this->PrintSplitLine();
  }

  virtual ~AMPVAEInferenceWorker() {}

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

    ValVec3 AngVel;
    this->Scheduler->template GetData<"AngleVelocityValue">(AngVel);

    ValVec3 Ang;
    this->Scheduler->template GetData<"AngleValue">(Ang);

    const ValVec3 ProjectedGravity =
        z::ComputeProjectedGravity(Ang, this->GravityVector);
    this->Scheduler->template SetData<concat(NetName, "NetProjectedGravity")>(
        ProjectedGravity);

    const ActorObsVec ActorObsScaled =
        static_cast<ActorObsVec>(math::cat(AngVel, ProjectedGravity, UserCmd3,
                                           CurrentMotorPos, CurrentMotorVel,
                                           LastAction)) *
        this->ActorObsScaleVec;

    const HistoryFrameVec HistoryFrameScaled =
        static_cast<HistoryFrameVec>(math::cat(AngVel, ProjectedGravity,
                                               UserCmd3, CurrentMotorPos,
                                               CurrentMotorVel, LastAction)) *
        this->HistoryFrameScaleVec;

    this->HistoryInputBuffer.push(HistoryFrameScaled);

    const HistoryObsVec FlattenedHistory =
        this->template FlattenHistory<HISTORY_FRAME_LENGTH, HISTORY_LENGTH,
                                      HISTORY_TERM_SIZES.size()>(
            this->HistoryInputBuffer, HISTORY_TERM_SIZES);

    this->ActorObsTensor.Array() = ActorObsVec::clamp(
        ActorObsScaled, -this->ClipObservation, this->ClipObservation);
    this->HistoryObsTensor.Array() = HistoryObsVec::clamp(
        FlattenedHistory, -this->ClipObservation, this->ClipObservation);
  }

  void PostProcess() override {
    const auto LastAction = this->OutputTensor.toVector();
    const auto ClippedLastAction =
        MotorValVec::clamp(LastAction, -this->ClipAction, this->ClipAction);
    this->Scheduler->template SetData<concat(NetName, "NetLastAction")>(
        ClippedLastAction);

    const auto ScaledAction =
        ClippedLastAction * this->OutputScaleVec + this->JointDefaultPos;
    this->Scheduler->template SetData<concat(NetName, "NetScaledAction")>(
        ScaledAction);

    const auto ClippedAction = MotorValVec::clamp(
        ScaledAction, this->JointClipLower, this->JointClipUpper);
    this->Scheduler->template SetData<concat(NetName, "Action")>(ClippedAction);

    this->end_time = std::chrono::steady_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        this->end_time - this->start_time);
    const InferencePrecision inference_time =
        static_cast<InferencePrecision>(duration.count());
    this->Scheduler->template SetData<concat(NetName, "InferenceTime")>(
        inference_time);
  }

  z::math::Tensor<InferencePrecision, 1, ACTOR_OBS_LENGTH> ActorObsTensor;
  z::math::Tensor<InferencePrecision, 1, HISTORY_OBS_LENGTH> HistoryObsTensor;
  z::math::Tensor<InferencePrecision, 1, OUTPUT_TENSOR_LENGTH> OutputTensor;

  ActorObsVec ActorObsScaleVec;
  HistoryFrameVec HistoryFrameScaleVec;
  z::math::Vector<InferencePrecision, OUTPUT_TENSOR_LENGTH> OutputScaleVec;
  ValVec3 CommandScaleVec;
  z::RingBuffer<HistoryFrameVec> HistoryInputBuffer;

  const ValVec3 GravityVector;

  InferencePrecision dt;

  std::chrono::steady_clock::time_point start_time;
  std::chrono::steady_clock::time_point end_time;
};

};  // namespace z
