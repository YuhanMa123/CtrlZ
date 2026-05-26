# BitbotSimulation Change Map

This is the concrete checklist for adding or switching an inference task under `example/BitbotSimulation`.

## Files that usually change

### `example/BitbotSimulation/types.hpp`

Usually the most important integration file.

Check or update:

- `JOINT_NUMBER`
- `MotorVec` and other aliases that depend on joint count
- `Net1Name` and net-related `CTSPair` definitions
- any new command pair such as 3D or 6D commands
- any extra published outputs required by the worker
- `SchedulerType` registration list
- `LoggerWorkerType` template arguments if new data should be logged
- `CmdWorkerType` alias to the right command pair
- `ActionManagementWorkerType` alias and selected action pair
- the selected inference worker alias, for example `AMPVAEInferenceWorkerType`
- `OBSERVATION_STUCK_LENGTH` or related compile-time dimensions

### `example/BitbotSimulation/user_func.h`

Usually change only the concrete inference worker pointer type in `UserData`.

### `example/BitbotSimulation/user_func.cpp`

Check or update:

- which config file path is loaded in `ConfigFunc`
- which inference worker type is instantiated
- whether the command worker type matches the chosen network
- whether `InferTask` worker ordering still makes sense
- whether `ActionManagementWorker->SwitchTo<...>()` targets the correct action pair

Normal order in `InferTask`:

- commander
- inference worker
- action manager
- logger

### `example/BitbotSimulation/config_*.json`

This is where most task-specific changes live when an existing worker can be reused.

Check or update:

- `Scheduler.dt`
- `Scheduler.InferTask.PolicyFrequency`
- `Workers.Commander` dimension and limits
- `Workers.MotorControl.DefaultPosition`
- `Workers.NN.Inference`
- `Workers.NN.Network.ModelPath`
- `Workers.NN.Network.InputNodeNames`
- `Workers.NN.Network.OutputNodeNames`
- worker-specific fields under `Workers.NN.Network`
- `Workers.NN.Preprocess.ObservationScales`
- `Workers.NN.Preprocess.ClipObservations`
- `Workers.NN.Postprocess.action_scale`
- `Workers.NN.Postprocess.clip_actions`
- `Workers.NN.Postprocess.joint_clip_upper`
- `Workers.NN.Postprocess.joint_clip_lower`

### `example/BitbotSimulation/main.cpp`

Usually unchanged for a pure inference-task swap.

Touch this only if the new task needs different states, events, or operator controls.

## Scheduler guardrail

- `Scheduler.dt` and `Scheduler.InferTask.PolicyFrequency` in `example/BitbotSimulation/config_*.json` are MuJoCo-side scheduling parameters.
- Do not map Isaac Sim `sim dt`, `decimation`, or training policy rate into these fields automatically.
- Keep the existing MuJoCo scheduler values unless the user explicitly asks to retune simulation scheduling or there is local code evidence that the example must change.
- If training docs and local scheduler values disagree, preserve the local scheduler first and note the mismatch in the final response.

## Decision shortcut

- If only checkpoint path, node names, observation scales, gait parameters, command limits, or default pose changed, edit config and integration files only.
- If observation tensors or output semantics changed, also update or add a worker under `CtrlZ/Workers/NN`.

## Prompt-to-code checklist

When generating code from a future prompt, translate the prompt into these edits:

1. Pick or create the worker type.
2. Ensure `types.hpp` declares every scheduler pair used by that worker.
3. Point `user_func.cpp` at the intended config and worker alias.
4. Encode the tensor contract in the config JSON.
5. Confirm that command dimension, joint count, and action vector length all match.

## Notes specific to the current repo snapshot

- The active example is currently wired to `AMPVAEInferenceWorkerType`.
- `ConfigFunc` currently loads `config_AMP_VAE.json`.
- `CmdWorkerType` currently publishes a 3D command pair.
- `ActionManagementWorkerType` currently forwards `Net1OutPair` to `TargetMotorPosition`.

That means many future AMP-VAE tasks will only need changes in config and example wiring, not a brand-new worker.
