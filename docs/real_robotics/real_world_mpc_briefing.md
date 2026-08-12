# Real-World MPC and Its Dissection for Robotics

**Study:** MPC vs VLA vs Diffusion  
**Date:** August 2026  
**Purpose:** Map the gap between toy 2D reaching and real-world robots, and identify where MPC is essential

---

## 1. What Breaks When Moving from Toy 2D Reaching to Real Robots?

### 1.1 Contact Dynamics and Friction

- **Toy models:** assume smooth differentiable dynamics (point-mass, double integrator).
- **Real robots:** make and break contact — hybrid, non-smooth dynamics.
- **Solutions:** Contact-Implicit MPC (LCPs, Signorini conditions, Coulomb friction), contact-implicit trajectory optimization.
- **Key paper:** "Fast Contact-Implicit Model Predictive Control" (Le Cleach et al., 2024).

### 1.2 Latency and Real-Time Constraints

- Toy baselines assume instantaneous computation; real robots have 50–200ms observation-execution latency.
- Control frequency: legged locomotion 500–1000Hz; manipulation 100–500Hz; contact-rich >1kHz.
- **Latency-aware execution** is required for stable deployment.
- **TinyMPC** (arXiv:2310.16985) runs at kHz on microcontrollers.

### 1.3 Sensor Noise and State Estimation

- Real robots must estimate state from noisy encoders, IMU, vision, force sensors.
- Floating-base robots cannot measure base pose directly.
- **Uncertainty-aware MPC** (SupeR-MPC, chance-constrained MPC) accounts for estimation error.

### 1.4 Actuator Limits and Saturation

- Torque is velocity-dependent; joint velocity limits depend on position; torque rate limits exist.
- Franka FCI requires explicit torque rate limits.
- Simple box constraints on `u` are insufficient.

### 1.5 Whole-Body Dynamics vs Point-Mass Models

- Point-mass ignores rotation, angular momentum, contact forces, underactuation.
- Hierarchy: point-mass → LIPM → centroidal (6 DOF) → single rigid body → kino-dynamics → full whole-body.
- Floating-base robots (quadrupeds, humanoids) are underactuated: base has 6 DOF controlled only through contact forces.

### 1.6 Torque-Level vs Position/Velocity Control

- **Torque control:** direct force, natural compliance, best for contact.
- **Position/velocity control:** relies on low-level PD/PI; lacks force control.
- Whole-body inverse-dynamics MPC directly optimizes joint torques.

---

## 2. Real-World MPC Formulations

### 2.1 Whole-Body Inverse Dynamics MPC

- **Reference:** "Whole-Body Inverse Dynamics MPC for Legged Loco-Manipulation" (arXiv:2511.19709, IEEE RA-L 2026)
- **Robot:** Unitree B2 quadruped + Unitree Z1 arm
- **Solver:** Fatrop (interior-point, block-sparse)
- **Performance:** 80Hz MPC, 500Hz torque interpolation
- **Stack:** Pinocchio + CasADi + Fatrop

### 2.2 Nonlinear MPC for Loco-Manipulation

- **Reference:** "A Nonlinear MPC Framework for Loco-Manipulation of Quadrupedal Robots with Non-Negligible Manipulator Dynamics" (IEEE RA-L 2026)
- **Robot:** Unitree Go2 + Kinova arm
- **Decomposition:** SRB for locomotion + full-order manipulator dynamics
- **Performance:** 60Hz MPC, 500Hz WBC

### 2.3 Centroidal / Single Rigid Body MPC

- Centroidal dynamics (CoM + angular momentum).
- <5ms solve time on CPU; 10,000 rollouts in 2ms on RTX 4050 (sampling-based).
- Tools: acados, JAX/MPPI, closed-form solvers.

### 2.4 Tools for Real-Time Solving

| Tool | Purpose | DGX Spark / aarch64 | Notes |
|------|---------|---------------------|-------|
| **Pinocchio** | Dynamics + derivatives | ✅ conda/source | Fast RNEA, URDF loading |
| **CasADi** | Symbolic OCP, AD, code gen | ✅ native ARM64 wheels | Interfaces to IPOPT, Fatrop |
| **Crocoddyl** | DDP, contact-aware OC | ✅ if Pinocchio builds | Good for legged/contact robots |
| **OCS2** | Switched-systems OC | likely | ROS integration, centroidal models |
| **acados** | Fast NMPC | ✅ | C code gen, SQP |
| **Fatrop** | Block-sparse IPM | unknown | Very fast, code gen |
| **Drake** | Robotics toolkit | ✅ arm64 binaries | IDTO, contact-implicit MPC |
| **MuJoCo** | Physics sim | ✅ pip | Fast, accurate, MJX for GPU |
| **Isaac Sim** | Photorealistic sim | ✅ DGX Spark | NVIDIA RTX/Blackwell |

---

## 3. MPC + Learned Components

### 3.1 Learning-Based Dynamics

- **KNODE:** Knowledge-based Neural ODEs + deep ensembles.
- **Residual dynamics:** `f_total = f_physics + f_nn`.
- **TD-MPC2:** implicit world model + local trajectory optimization in latent space.

### 3.2 Diffusion Warm-Start for MPC

- **Reference:** "Warm-Starting Collision-Free MPC with Object-Centric Diffusion" (arXiv:2601.02873, IEEE RA-L 2026)
- Diffusion trajectory prior → MPC refines with SDF + rigid-body dynamics.
- Already implemented in `mpc_baselines_repo` as `DiffusionWarmStartMPC` (but with simple NumPy diffusion).

### 3.3 VLMPC

- **Reference:** "VLMPC: Vision-Language Model Predictive Control" (arXiv:2407.09829, RSS 2024)
- VLM samples action candidates → video prediction model → hierarchical cost → select best.
- Can combine with MPC for constraint satisfaction.

### 3.4 World-Model + MPC Hybrid

- World model predicts future states; MPC optimizes in latent space.
- Examples: Dream-MPC, TD-MPC2, Ego-Vision World Model.
- Advantage: sample efficiency and high-dimensional observations.
- Challenge: world model errors compound.

---

## 4. Dissection: When is MPC Truly Needed?

| Situation | MPC | Learned Policy | Hybrid |
|-----------|-----|----------------|--------|
| Hard safety constraints | ✅ essential | ⚠️ soft | ✅ safety filter |
| High-rate (>100Hz) | ✅ essential | ❌ too slow | ✅ high-rate MPC + low-rate policy |
| Contact-rich, force control | ✅ essential | ⚠️ needs data | ✅ contact-implicit MPC + learned cost |
| Long-horizon, feasibility | ✅ good | ⚠️ myopic | ✅ hierarchical |
| Perception-heavy, language | ⚠️ needs hand-crafted | ✅ excellent | ✅ VLM + MPC safety filter |
| Generalization to novel scenes | ⚠️ needs re-modeling | ✅ excellent | ✅ learned model + MPC |
| Computation limited | ⚠️ may be heavy | ✅ amortized | ✅ tinyMPC + policy |

**Rule of thumb:**
- **MPC** wins on safety, high-rate, contact, constraints, and analytical dynamics.
- **Learned policies** win on perception, language, generalization, and amortized computation.
- **Hybrid** is the future for real-world robots.

---

## 5. Recommendations for Extending the Study

### Extend MPC baselines

1. **Floating-base / centroidal MPC:** add SRB + contact force optimization.
2. **Torque-level manipulator MPC:** add RNEA, inverse dynamics, actuator limits.
3. **Contact-implicit MPC:** LCP-based contact and friction.
4. **State estimation + uncertainty:** EKF and chance constraints.
5. **Latency compensation:** latency-aware execution and action chunking.

### Real-physics benchmarks to add

- **MetaWorld / LIBERO / RoboSuite** (MuJoCo manipulation)
- **ManiSkill3** (GPU-parallelized manipulation)
- **CALVIN** (long-horizon language-conditioned)
- **Quadruped walking** (SRB/centroidal MPC)
- **Loco-manipulation** (quadruped + arm)

### Tools to integrate

1. **Pinocchio + CasADi** for accurate dynamics and optimization.
2. **MuJoCo/MJX** for fast physics on DGX Spark.
3. **Crocoddyl / acados** for contact and whole-body MPC.
4. **Isaac Sim** for photorealistic visual training.

### Research questions

**RQ9:** How does MPC performance degrade from point-mass to whole-body dynamics?  
**RQ10:** Can learned dynamics models replace analytical models in MPC on real robots?  
**RQ11:** Is contact-implicit MPC necessary or is contact-sequence planning sufficient?  
**RQ12:** How do hybrid approaches (diffusion warm-start, VLMPC, world-model MPC) scale?

---

## 6. Key References

- Molnar et al., "Whole-Body Inverse Dynamics MPC for Legged Loco-Manipulation", arXiv:2511.19709
- "A Nonlinear MPC Framework for Loco-Manipulation", IEEE RA-L 2026, arXiv:2507.22042
- "Warm-Starting Collision-Free MPC with Object-Centric Diffusion", arXiv:2601.02873
- "VLMPC: Vision-Language Model Predictive Control", arXiv:2407.09829
- Hansen et al., "TD-MPC2: Scalable, Robust World Models for Continuous Control", arXiv:2310.16828
- "Fast Contact-Implicit Model Predictive Control", arXiv:2107.05616
- "BiConMP: A Nonlinear MPC Framework for Whole Body Motion Planning", arXiv:2201.07601
- "TinyMPC", arXiv:2310.16985
- "A Latency-Aware Framework for Visuomotor Policy Learning on Industrial Robots", arXiv:2602.14255
- Pinocchio: https://github.com/stack-of-tasks/pinocchio
- CasADi: https://github.com/casadi/casadi
- Crocoddyl: https://github.com/loco-3d/crocoddyl
