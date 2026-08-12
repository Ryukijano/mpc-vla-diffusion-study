# From Toy Sim to Real-Physics Simulators

**Status:** Phase-2 planning  
**Skills:** collaborative-research, experiment-protocol, sim-to-real  
**Owner:** Gyanateet  
**Created:** 2026-08-11  

---

## 1. Scope: What Changes in Phase 2

Phase 1 answers the *mechanism* question using 2D reaching and PushT. Phase 2 replaces those toy tasks with real-physics, vision-language, contact-rich benchmarks. The goal is not just bigger simulators — it is to introduce the variables that dominate real robot performance:

- **Real contact and friction** (multi-body impact, slip, deformation)
- **3D visual scenes** with camera calibration, occlusion, and clutter
- **Natural language task instructions**
- **Whole-body dynamics** for legged / mobile manipulation
- **Sim-to-real visual distribution shift**

Phase 2 also adds a fourth comparison family: **World Action Models (WAMs)**. WAMs are positioned between classical MPC (explicit model, optimization) and end-to-end VLA / Diffusion (implicit model, reactive generation), and they may be the right architecture for long-horizon, contact-rich tasks.

---

## 2. Simulator Stack and Replacement Tasks

### 2.1 Target Simulators

| Simulator | Physics Engine | Rendering | Key Strengths | Candidate Tasks | Family Fit |
|-----------|---------------|-----------|---------------|-----------------|------------|
| **MuJoCo + robosuite** | Generalized coordinates, accurate contact | Basic / MuJoCo renderer | Fast, mature, full Franka models, rich manipulation task suite | `Lift`, `PickPlaceCan`, `Square`, `NutAssembly` | MPC, WAM, Diffusion, VLA |
| **ManiSkill3** | SAPIEN, GPU-parallel rigid body | Photorealistic PBR, ray-traced option | Massively parallel, language-conditioned, dexterous manipulation | `PickCube`, `StackCube`, `PushCube`, `TurnFaucet`, `OpenCabinet` | VLA, Diffusion, WAM |
| **Isaac Sim** | NVIDIA PhysX, GPU | RTX photorealistic, material-rich | Best-in-class photorealism, domain randomization, sim-to-real asset pipeline | Warehouse manipulation, cobot pick-and-place, humanoid/quadruped | All, especially sim-to-real |
| **RoboDojo** (if available) | Differentiable / whole-body dynamics | Lightweight | Whole-body humanoid/quadruped optimal control | Locomotion, balance, reaching with floating base | MPC, WAM |

### 2.2 Task Replacement Map

| Toy Phase-1 Task | Phase-2 Replacement | Rationale |
|------------------|--------------------|-----------|
| 2D Reaching (clear) | robosuite `Lift`, ManiSkill3 `PickCube` | Single-stage reaching + grasp in 3D with real contact. |
| 2D Reaching (cluttered) | robosuite `Can`, `Square`, ManiSkill3 `PushCube` | Obstacle avoidance, multi-modal solutions, contact. |
| PushT | ManiSkill3 `PushCube`, `StackCube`, robosuite `Square` | Contact-rich pushing with 6-DOF objects; long-horizon variants via stacking. |

### 2.3 Language Conditioning

Use task descriptors for both VLA and WAM:

- **Robosuite:** wrap object names and target locations into text instructions, e.g., "Pick the red can and place it in the bin."
- **ManiSkill3:** use built-in language descriptions or templates from the task specification.
- **VLA:** feed text as a separate language token stream to the VLM backbone.
- **WAM:** embed language into the latent planning/action model or use a cross-modal attention layer.

---

## 3. Updated Comparison Matrix (with World Action Model)

This extends the family comparison in [`../comparison_plan.md`](../comparison_plan.md) by adding **World Action Models (WAMs)** as a fourth column and emphasizing real-robot capabilities.

| Capability | Classical MPC | VLA | Diffusion / GCP | World Action Model (WAM) | Hybrid (Diffusion Warm-Start + MPC) |
|------------|--------------|-----|-----------------|--------------------------|-------------------------------------|
| **Action representation** | Continuous optimization variables | Discrete tokens / regression | Iterative denoising trajectory | Latent action sequence + planned state sequence | Learned trajectory + MPC refinement |
| **Multi-modality** | Single solution (local opt) | Single (regression) or multi (diffusion head) | Claimed multi-modal | Multi-modal in latent plan | Single after refinement |
| **Observation conditioning** | State vector | Image + language + proprioception | Image / state / point cloud / language | Image + language + proprioception + future latent | Image / state + language |
| **Dynamics model** | Explicit (analytical / learned) | Implicit in policy | Implicit in policy | Explicit learned world model + action policy | Explicit + learned prior |
| **Constraint handling** | Hard constraints (solver) | None (soft, from data) | None (soft, from data) | Soft (learned dynamics can be used for safety checks) | Hard (MPC refine) |
| **Generalization** | Limited (needs re-modeling) | Strong (VLM pretraining) | Moderate (from demo data) | Moderate (rollout generalization) | Combines model + learned prior |
| **Language conditioning** | None | Native | Via VLM / text embedding | Via VLM or task embedding | Via VLM or text embedding |
| **Inference latency** | <1 ms to ~10 ms | 50–200 ms+ | 10–100 ms | 10–150 ms (includes latent rollout) | 10–50 ms |
| **Data efficiency** | N/A (model-based) | Low (large demos) | Moderate | Moderate (needs future/rollout labels) | Moderate + model cost |
| **Safety guarantees** | Yes (constraint enforcement) | No | No | No (unless safety layer added) | Yes (MPC constraints) |
| **OOD robustness** | Poor (model mismatch) | Moderate (VLM helps) | Key: noise + iterative compute | Moderate (predictive dynamics) | Strong if dynamics model transfers |
| **Long-horizon planning** | Strong if model is accurate | Poor (chunked, reactive) | Moderate | Strong (explicit future rollouts) | Strong (MPC horizon + warm start) |
| **Sim-to-real transfer** | Needs model re-identification | Strong (visual pretraining) | Moderate | Moderate (dynamics mismatch) | Strong if dynamics model transfers |
| **Contact-rich tasks** | Strong if contact model is accurate | Moderate | Moderate | Strong (contact prediction in latent space) | Strong |
| **Real-time feasibility** | Yes (kHz) | No (open-loop chunking) | Borderline (depends on steps) | Borderline (depends on rollout length) | Yes with fast warm start |

---

## 4. Real-Robot Vision Preprocessing

All vision-conditioned conditions in Phase 2 use a **shared real-robot vision pipeline**. This is the same idea as the fairness controls in [`../methodology.md`](../methodology.md), but extended for real-style images and sim-to-real transfer.

### 4.1 Pipeline Stages

| Stage | Operation | Purpose | Notes |
|-------|-----------|---------|-------|
| **Camera calibration** | Intrinsics, distortion, extrinsics | Correct projection; depth-to-RGB alignment | Required for real depth; optional in sim |
| **RGB/Depth sync** | Time-stamp alignment, depth registration | Fused 3D observation | Use if depth is available |
| **Crop/resize** | 224×224 or 320×240, keep aspect ratio | VLA / diffusion / WAM input | Same across methods |
| **Normalize** | ImageNet, DINO, or CLIP statistics | Match pretrained vision backbone | Match the VLA/WAM pretraining |
| **Domain randomization** | Lighting, texture, background, camera pose | Sim-to-real regularization | Training-time only |
| **Photorealistic rendering** | RTX / PBR materials, ray-traced shadows | Close sim/real visual gap | Isaac Sim / ManiSkill3 |
| **Feature extraction** | DINO, CLIP, SAM, or learned encoder | Compact visual representation | Optional; may be used by WAM/MPC |
| **Sim-to-real adaptation** | CycleGAN / AdaIN / latent adaptation | Reduce visual domain shift | Optional; evaluated in `EXP-006` |
| **Point cloud / voxel prep** | RGBD fusion, downsampling, voxelization | 3D conditioning for diffusion/WAM | Optional for tasks where geometry matters |

### 4.2 Controlled Preprocessing Rules

1. **Same input size and normalization** across VLA, diffusion, MIP, and WAM.
2. **Same domain randomization schedule** applied at training time for all learning-based conditions.
3. **No task-specific augmentation** that is not shared.
4. **Latency measurements include the full preprocessing time** if it runs at every control step.
5. **Real-robot camera intrinsics** are logged and committed in `configs/cameras/`.

---

## 5. Phase-2 Experiment List (EXP-005 to EXP-008)

| ID | Title | Objective | Benchmark(s) | Key Conditions | Primary Metric | Next Phase Linkage |
|----|-------|-----------|--------------|----------------|----------------|--------------------|
| [`EXP-005`](../../experiments/EXP-005-world-models/protocol.md) | World Action Model Baseline | Add WAM baseline and compare against VLA, diffusion, MIP, and MPC. | MuJoCo robosuite / ManiSkill3 manipulation | WAM + policy, SmallVLA, DDPM, MIP, MPC | Long-horizon & contact-rich success rate | Feeds WAM condition into `EXP-008`; provides candidate WAM for Phase 3. |
| [`EXP-006`](../../experiments/EXP-006-sim-to-real/protocol.md) | Sim-to-Real Domain Gap | Test domain randomization and photorealistic rendering for sim-to-real transfer. | ManiSkill3 / Isaac Sim / VLA-REPLICA or RoboDojo-Real (if available) | No-DR, DR, photoreal, image transfer, real VLA | Sim vs. real success gap | Informs real-vision preprocessing and Phase 3 dataset collection. |
| [`EXP-007`](../../experiments/EXP-007-real-robot-mpc/protocol.md) | Whole-Body Real-Robot MPC | Evaluate whole-body MPC for real-robot-like humanoid/quadruped dynamics. | MuJoCo humanoid / quadruped / RoboDojo | Centroidal/SRB MPC, full-order MPC, diffusion warm-start MPC, learned policy | Stability, task success, control frequency | Identifies best MPC variant for Phase 4 hardware. |
| [`EXP-008`](../../experiments/EXP-008-real-vision/protocol.md) | Real-Vision OOD Robustness | Stress test vision-conditioned controllers under real-robot vision perturbations. | robosuite / ManiSkill3 with real-style camera | VLA, DDPM, MIP, WAM × {clean, lighting, occlusions, distractors, background} | OOD success rate and manifold adherence | Decides which vision-conditioned family is most robust for Phase 3/4. |

---

## 6. Phase 2 → Phase 3 → Phase 4 Gates

| Gate | Criteria | Evidence Required |
|------|----------|-------------------|
| **2 → 3** | 1. All baselines run ≥ 10 Hz in sim. <br> 2. WAM matches SmallVLA on long-horizon task (gap ≤ 3 pp). <br> 3. Sim-to-real visual gap ≤ 15 pp on easiest task. | Benchmark logs; latency harness output; sim-to-real gap table. |
| **3 → 4** | 1. ≥ 10% real-image success on target tasks under OOD suite. <br> 2. Whole-body MPC ≥ 95% stability on stand/walk. <br> 3. Safety constraints violated in < 1% of test episodes. | OOD success per perturbation; fall-rate table; constraint violation log. |
| **4 complete** | 1. ≥ 50% real-world success on 3 contact-rich tasks. <br> 2. Zero serious safety violations. <br> 3. Reproducible deployment logs and video. | Hardware run logs; safety review; final report. |

---

## 7. Timeline and Deliverables

### Proposed Timeline

| Weeks | Activity | Output |
|-------|----------|--------|
| 9–10 | Sim stack setup, environment bindings, camera/real-vision preprocessing | Working `src/` simulators + camera config templates |
| 11–12 | `EXP-005` WAM baseline + family comparison on ManiSkill3 / robosuite | WAM checkpoints; comparison tables |
| 13–14 | `EXP-006` sim-to-real; `EXP-007` whole-body MPC | DR/photoreal configs; MPC solver logs; sim-to-real gap report |
| 15–16 | `EXP-008` real-vision OOD; phase-gate review | OOD robustness figures; phase-gate document |

### Deliverables

1. `docs/real_robotics/phase2_report.md` — Phase 2 results and go/no-go decision.
2. `results/tables/real_physics_comparison.csv` — Updated family comparison on real-physics tasks.
3. `configs/real_robotics/` — Sim configs, camera configs, DR schedules, MPC configs.
4. `src/real_robotics/` — Sim bindings, WAM baseline, whole-body MPC wrappers, real-vision preprocessing.
5. `data/manifests/` — Sim-to-real data manifests and hashes.
6. `experiments/EXP-005/outputs/` through `EXP-008/outputs/` — Results and checkpoints.

---

## 8. Related Documents

- Real-robotics phase master plan: [`README.md`](README.md)
- Comparison plan: [`../comparison_plan.md`](../comparison_plan.md)
- Methodology: [`../methodology.md`](../methodology.md)
- Research questions: [`../research_questions.md`](../research_questions.md)
