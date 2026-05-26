# Worker Selection

Use this file to decide whether the task only needs integration changes or a new worker in `CtrlZ/Workers/NN`.

## Reuse an existing worker when

- the ONNX input count matches an existing worker
- the observation ordering is the same, even if scales or checkpoint paths differ
- only the robot joint count changed and the worker is already templated for `JOINT_NUMBER`
- only history length changed and the worker already templates that dimension
- only action scale, command limits, gait parameters, or checkpoint file changed

Typical reuse cases:

- new AMP-VAE checkpoint with the same `actor_obs + obs_history` interface
- new AMP checkpoint with the same single actor input and output action vector
- new UniFP checkpoint with the same 6D command input logic

## Create or modify a worker when

- the prompt introduces a new input tensor count or output tensor count
- the observation composition or ordering differs from all existing workers
- the model has extra heads such as latent variables, reference trajectories, or hidden states
- the action decoding logic changes materially
- the network needs recurrent state persistence across steps
- the prompt requires a different command dimension than the current worker expects

## Practical mapping in this repo

### `AMPVAEInferenceWorker.hpp`

Use when:

- two inputs: `actor_obs` and flattened history
- one action output
- no explicit gait clock is needed

Current pattern:

- actor observation and history frame are both built from `ang_vel + projected_gravity + cmd3 + dof_pos + dof_vel + last_action`
- history is maintained with `RingBuffer`

### `AMPInferenceWorker.hpp`

Use when:

- single input actor tensor
- gait-phase logic is part of preprocessing
- history stacking is present

### `UniFPInferenceWorker.hpp`

Use when:

- command vector is 6D
- clock features are part of the policy input
- walking mask and phase logic matter

### `UnitreeRlGymInferenceWorker.hpp`

Use when:

- ONNX exports hidden and cell state tensors
- recurrent state must be fed back every step

## Worker authoring checklist

When a new worker is necessary:

1. derive from `CommonLocoInferenceWorker` unless the task clearly needs a different base
2. declare tensor shapes as `static constexpr`
3. allocate concrete `math::Tensor` members with stable lifetime
4. push every input and output tensor into `InputOrtTensors__` and `OutputOrtTensors__`
5. make `PreProcess()` read scheduler data in the exact training order
6. clamp observations before inference
7. make `PostProcess()` clamp raw actions, rescale with `ActionScale`, add `JointDefaultPos` when needed, then clamp to joint limits
8. publish the final action with `concat(NetName, "Action")`
9. publish `InferenceTime`

## Common failure modes

- input node name count does not match the number of bound tensors
- history flatten order differs from training
- command dimension in `CmdWorkerType` does not match the worker's expectation
- `JOINT_NUMBER` does not match `action_scale`, joint clips, or default pose length
- extra scheduler pairs were added to worker code but not to `types.hpp`
