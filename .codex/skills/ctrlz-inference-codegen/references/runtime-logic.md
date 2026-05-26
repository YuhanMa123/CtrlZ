# Runtime Logic

This project's inference path is centered on scheduler workers and compile-time data-bus pairs.

## Top-level call chain

1. `example/BitbotSimulation/main.cpp`
   - creates the Mujoco kernel
   - registers `ConfigFunc`, `FinishFunc`, events, and states
   - enters `kernel.Run()`
2. `example/BitbotSimulation/user_func.cpp`
   - `ConfigFunc` loads the selected JSON config
   - creates `SchedulerType`
   - creates device-facing workers, command worker, action manager, and the selected inference worker
   - assigns workers to task lists such as `MainTask`, `InferTask`, and `ResetTask`
   - starts the scheduler
3. state handlers call `TaskScheduler->SpinOnce()`
   - this advances each enabled task list at the configured rate
4. in `InferTask`
   - command worker writes command data
   - inference worker runs `PreProcess -> InferenceOnce -> PostProcess`
   - action manager remaps the selected net output to `TargetMotorPosition`
   - logger flushes selected telemetry

## Worker layering

### `AbstractNetInferenceWorker`

File: `CtrlZ/Workers/NN/AbstractInferenceWorker.hpp`

Responsibilities:

- reads `Inference` and `Network` config blocks
- creates the ONNX Runtime session
- stores input and output node name arrays
- binds `Ort::Value` tensors in `TaskCreate()`
- optionally warms up the model
- runs the default task loop

Important constraint:

- Derived workers must fill `InputOrtTensors__` and `OutputOrtTensors__` before scheduler launch, otherwise `TaskCreate()` throws.

### `CommonLocoInferenceWorker`

File: `CtrlZ/Workers/NN/CommonLocoInferenceWorker.hpp`

Responsibilities:

- reads observation scales, clips, default joint positions, action scales, and joint limits
- exposes common vectors such as `Scales_ang_vel`, `Scales_dof_pos`, `ActionScale`, `JointDefaultPos`

Use this as the base class for most locomotion policies.

### Concrete inference workers

Files: `CtrlZ/Workers/NN/*InferenceWorker.hpp`

Responsibilities:

- declare concrete input and output tensor shapes
- build scale vectors
- prepare input tensors in `PreProcess()`
- decode and publish outputs in `PostProcess()`
- optionally maintain history buffers or recurrent state

## Scheduler data flow

Common reads in `PreProcess()`:

- `"CurrentMotorVelocity"`
- `"CurrentMotorPosition"`
- `concat(NetName, "NetLastAction")`
- `concat(NetName, "NetUserCommand3")` or `concat(NetName, "NetUserCommand6")`
- `"AngleVelocityValue"`
- `"AngleValue"`

Common writes in `PostProcess()`:

- `concat(NetName, "NetLastAction")`
- `concat(NetName, "NetScaledAction")`
- `concat(NetName, "Action")`
- `concat(NetName, "InferenceTime")`

Optional writes:

- `concat(NetName, "NetProjectedGravity")`
- `concat(NetName, "NetClockVector")`
- recurrent hidden states or auxiliary heads

## Why `types.hpp` is critical

`example/BitbotSimulation/types.hpp` is the compile-time contract for the entire example:

- robot dimensions such as `JOINT_NUMBER`
- all scheduler `CTSPair` registrations
- scheduler alias `SchedulerType`
- logger alias contents
- command worker alias dimension
- action manager output pair
- selected inference worker alias

If a worker reads or writes a pair that is missing here, the integration is incomplete.
