# Real-Robotics Phase Roadmap

**Status:** Phase-2 planning  
**Skills:** collaborative-research, experiment-protocol, sim-to-real  
**Owner:** Gyanateet  
**Created:** 2026-08-11  

---

## 1. Why Move Beyond Toy Experiments

The first phase of the study uses 2D reaching and PushT to isolate the **mechanism** question (H3/H4): is the advantage of diffusion/flow-based generative control policies (GCPs) due to distribution fitting or to iterative compute + noise? Those tasks are clean, low-dimensional, and fast to iterate on — but they deliberately strip away the properties that make real robot control hard.

Phase 2 and beyond must answer a harder question: **do the same family rankings hold when the robot is embodied, contact-rich, vision-conditioned, language-conditioned, and operating under real physics?**

Toy experiments cannot capture:

- **Real contact and friction:** 2D reaching has no impact/collision dynamics; PushT has simple contact. Real manipulation has multi-contact, slip, and deformable objects.
- **Long-horizon task structure:** Reaching is one-step; real tasks require sequencing grasp → transport → place/insert.
- **Real visual observations:** 224×224 synthetic crops are not the same as real RGB/Depth with lighting variation, occlusion, and clutter.
- **Sim-to-real distribution shift:** A policy trained on a single renderer often collapses on the first real image.
- **Whole-body dynamics:** Legged or mobile manipulators have unilateral contacts, floating bases, and torque limits that 2D point-mass tasks cannot exercise.

The real-robotics phase therefore transitions from **toy simulators → real-physics simulators → real-robot datasets → hardware deployment**, with explicit phase-gate criteria at each step.

---

## 2. Three Focus Areas

### 2.1 World Action Models (WAMs)

**What:** Joint dynamics + action models that explicitly predict future states or observations and produce actions conditioned on a latent plan. They sit between classical MPC (explicit model, optimization) and end-to-end VLA/Diffusion (implicit model, reactive generation).

**Why it matters:** WAMs can improve long-horizon credit assignment, contact prediction, and safety by rolling out futures before acting. They may be especially useful when the task requires multiple sub-goals, tool use, or recovery from contact.

**Research question:** Does adding an explicit, small world action model improve long-horizon and contact-rich manipulation over a pure VLA or diffusion policy?

- See: [`world_action_models_briefing.md`](world_action_models_briefing.md)

### 2.2 Real-World MPC

**What:** Whole-body, contact-aware model predictive control — including centroidal / single-rigid-body (SRB) approximations, full-order nonlinear MPC, and learned warm-start variants such as diffusion warm-start MPC. The focus is on stability, real-time rates, and constraint handling on real-robot dynamics.

**Why it matters:** Classical MPC dominated the toy 2D reaching tasks because the model was perfect. On real robots the model is approximate, the horizon is longer, and the solver must run fast enough to close the loop. Whole-body MPC is the test of whether optimization-based control can scale beyond low-dimensional arms.

**Research question:** When does full-order whole-body MPC beat centroidal/SRB approximations, and can diffusion-based warm starts close the real-time gap while preserving constraint satisfaction?

- See: [`real_world_mpc_briefing.md`](real_world_mpc_briefing.md)

### 2.3 Real Robotic Vision

**What:** Vision preprocessing, domain randomization, photorealistic rendering, sim-to-real image transfer, and out-of-distribution (OOD) perturbation tests. This focus area turns synthetic visual inputs into real-robot visual inputs.

**Why it matters:** Vision-conditioned controllers (VLA, diffusion, WAM) are only as good as the visual distribution they see at training. Real deployment introduces lighting changes, occlusions, distractors, and background changes. Systematic perturbation testing on real-style images is the only way to measure OOD robustness before hardware.

**Research question:** How robust are VLA, diffusion/GCP, and WAMs to real-world visual perturbations, and which architectural components confer robustness?

- See: [`real_robotic_vision_briefing.md`](real_robotic_vision_briefing.md)

---

## 3. Briefing Documents

These briefings are maintained alongside the phase master plan and provide the detailed background for each focus area.

| Briefing | Purpose |
|----------|---------|
| [`world_action_models_briefing.md`](world_action_models_briefing.md) | WAM landscape, candidate architectures, and how to train a small baseline from scratch. |
| [`real_world_mpc_briefing.md`](real_world_mpc_briefing.md) | Centroidal/SRB vs full-order MPC, whole-body solvers, warm-start strategies, and real-time requirements. |
| [`real_robotic_vision_briefing.md`](real_robotic_vision_briefing.md) | Camera preprocessing, domain randomization, photorealistic rendering, OOD perturbation suite. |
| [`tools_and_data_inventory.md`](tools_and_data_inventory.md) | Simulators, datasets, real-robot platforms, compute requirements, and software versions. |

---

## 4. Roadmap: From Sim to Hardware

| Phase | Title | Objective | Key Experiments / Deliverables | Gate to Next Phase |
|-------|-------|-----------|-------------------------------|--------------------|
| **2** | Real-physics simulators | Replace 2D reaching / PushT with real-physics, vision-language, contact-rich tasks; add WAM baseline and sim-to-real validation. | `EXP-005`, `EXP-006`; sim stack (MuJoCo / robosuite, ManiSkill3, Isaac Sim, RoboDojo if available); updated comparison matrix; real-vision preprocessing pipeline. | 1. All baselines run ≥ 10 Hz in sim at the target control frequency. <br> 2. WAM matches or exceeds SmallVLA on a long-horizon task (gap ≤ 3 pp). <br> 3. Sim-to-real visual gap on the easiest task is ≤ 15 pp. |
| **3** | Real-robot datasets | Whole-body MPC stress tests; real-image OOD robustness; alignment of sim and real demonstration data. | `EXP-007`, `EXP-008`; real-robot dataset manifest; controller checkpoints; real-vision perturbation suite. | 1. ≥ 10% real-image success on target manipulation tasks under the OOD suite. <br> 2. Whole-body MPC achieves ≥ 95% stability on humanoid/quadruped stand-walk tasks. |
| **4** | Hardware deployment | Deploy best-performing controllers on physical robot(s): Franka / mobile manipulator / humanoid / quadruped. | Real-robot deployment protocol; safety checklists; hardware logs; failure taxonomy. | 1. ≥ 50% real-world success on 3 contact-rich tasks. <br> 2. Zero serious safety violations. <br> 3. Reproducible run logs and video records. |

---

## 5. Real-Robotics Experiments

| ID | Title | Hypothesis | Primary Outcome | Link |
|----|-------|------------|-----------------|------|
| `EXP-005` | World Action Model Baseline | `H_WAM`: WAMs improve long-horizon and contact-rich manipulation over pure VLA. | Success rate on MuJoCo robosuite / ManiSkill3 manipulation. | [`../../experiments/EXP-005-world-models/protocol.md`](../../experiments/EXP-005-world-models/protocol.md) |
| `EXP-006` | Sim-to-Real Domain Gap | `H_S2R`: Domain randomization and photorealistic rendering reduce the sim-to-real success gap. | Sim vs. real success gap. | [`../../experiments/EXP-006-sim-to-real/protocol.md`](../../experiments/EXP-006-sim-to-real/protocol.md) |
| `EXP-007` | Whole-Body Real-Robot MPC | `H_RRMPC`: Full-order / diffusion warm-start whole-body MPC outperforms centroidal/SRB and learned policies on stability and task success. | Stability, task success, and control frequency. | [`../../experiments/EXP-007-real-robot-mpc/protocol.md`](../../experiments/EXP-007-real-robot-mpc/protocol.md) |
| `EXP-008` | Real-Vision OOD Robustness | `H_RVV`: World models and iterative policies are more robust than pure VLA under real-robot vision perturbations. | Success rate and manifold adherence per perturbation level. | [`../../experiments/EXP-008-real-vision/protocol.md`](../../experiments/EXP-008-real-vision/protocol.md) |

---

## 6. Related Documents

- Study root: [`../../README.md`](../../README.md)
- Comparison plan: [`../comparison_plan.md`](../comparison_plan.md)
- Research questions: [`../research_questions.md`](../research_questions.md)
- Methodology: [`../methodology.md`](../methodology.md)
- Phase 2 detailed plan: [`phase2_roadmap.md`](phase2_roadmap.md)
