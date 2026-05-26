---
name: ctrlz-inference-codegen
description: Use this skill when asked to understand, extend, or generate CtrlZ C++ ONNX inference code, especially for `CtrlZ/Workers/NN` and `example/BitbotSimulation`. It guides Codex to analyze the existing inference pipeline, decide whether an existing inference worker can be reused, and then implement the required C++ integration from a user prompt, including worker code, scheduler data pairs, BitbotSimulation wiring, and JSON config updates.
---

# CtrlZ Inference Codegen

Use this skill when the task is about the CtrlZ C++ inference framework, adding a new inference task, or generating sim2sim inference code from a prompt.

## What this skill covers

- The runtime path from `example/BitbotSimulation/main.cpp` into scheduler task lists and ONNX inference workers.
- How `AbstractNetInferenceWorker` and `CommonLocoInferenceWorker` constrain new worker implementations.
- How to decide whether only `example/BitbotSimulation` needs changes, or whether `CtrlZ/Workers/NN` also needs a new worker.
- How to generate code directly from a prompt instead of only writing a plan.

## Load these references

- For the runtime call chain and worker layering, read `references/runtime-logic.md`.
- For deciding whether to reuse or create a worker, read `references/worker-selection.md`.
- For the concrete file checklist under `example/BitbotSimulation`, read `references/bitbotsimulation-change-map.md`.

## Default workflow

1. Read the user prompt and extract:
   - robot joint count
   - model path
   - input node names and count
   - output node names and count
   - observation composition and order
   - history stacking or recurrent state requirements
   - command vector dimension
   - action semantics: delta action, scaled action, absolute target, latent outputs, extra heads
2. Inspect the current repo state before editing:
   - `example/BitbotSimulation/types.hpp`
   - `example/BitbotSimulation/user_func.h`
   - `example/BitbotSimulation/user_func.cpp`
   - the target config JSON in `example/BitbotSimulation`
   - matching files in `CtrlZ/Workers/NN`
3. Decide the implementation path:
   - Reuse an existing worker if the prompt matches an existing observation and output pattern closely enough.
   - Create or update a worker in `CtrlZ/Workers/NN` only when the network structure or tensor contract is materially different.
4. Implement the code directly.
5. Verify wiring consistency:
   - tensor count matches `InputNodeNames` and `OutputNodeNames`
   - scheduler data pairs exist for every `GetData` and `SetData`
   - `JOINT_NUMBER`, action scale size, default position size, and joint clip sizes agree
   - commander dimension matches the worker's expected command vector
6. Summarize what changed and call out any unresolved ambiguity.

## Hard rules

- Prefer reusing existing workers when the new task is only a new checkpoint or a parameter retune.
- Treat `CtrlZ/Workers/NN` as the main extension point only for genuinely new tensor layouts, new recurrent state handling, or different pre/post-process semantics.
- Keep new worker logic aligned with training-side observation order. Never silently reorder features.
- Treat `example/BitbotSimulation` scheduler timing as a protected integration contract. Do not casually change `Scheduler.dt` or `Scheduler.InferTask.PolicyFrequency` based only on training-side Isaac Sim control rates.
- Distinguish clearly between MuJoCo-side simulation scheduling and training-side control/decimation settings. Only change MuJoCo scheduler timing when the user explicitly asks for it or the local example wiring proves it is required.
- When generating code, update the real project files instead of returning pseudo-code unless the user explicitly asked for design only.
- Do not invent scheduler data names. Reuse existing names when semantics match; otherwise add new `CTSPair` definitions in `types.hpp` and thread them through `SchedulerType`, logger aliases, action manager aliases, and worker aliases.

## Compatibility heuristics

- `AMPVAEInferenceWorker` is the closest fit for two-input actor-plus-history models.
- `AMPInferenceWorker` is the closest fit for stacked-history AMP actor models with gait phase logic.
- `UniFPInferenceWorker` is the closest fit for 6D command inputs and clocked locomotion.
- `UnitreeRlGymInferenceWorker` is the closest fit for LSTM-style hidden and cell state tensors.
- `PlainInferenceWorker` or `CommonLocoInferenceWorker`-based patterns are the best starting point for simple single-input feed-forward policies.

## When a prompt is incomplete

If a prompt omits details, infer from this priority order:

1. existing worker patterns in `CtrlZ/Workers/NN`
2. the target ONNX node names and current config JSON
3. the currently selected robot setup in `example/BitbotSimulation/types.hpp`
4. neighboring configs such as `config_AMP.json`, `config_AMP_VAE.json`, `config_UniFP.json`

Only ask the user a question if a wrong assumption would likely break tensor dimensions or action semantics.

When training docs mention `sim dt`, `decimation`, or policy frequency, do not directly overwrite the MuJoCo example scheduler. First preserve the local `Scheduler` block and treat training timing as reference information unless the user explicitly requests a scheduler change.

## Expected output style for future codegen tasks

When this skill is used for actual implementation, do the work in this order:

1. choose the worker strategy
2. patch worker and integration files
3. run a lightweight consistency check if possible
4. report the concrete changes and any assumptions
