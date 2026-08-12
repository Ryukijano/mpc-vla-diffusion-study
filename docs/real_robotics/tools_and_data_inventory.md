# Tools and Data Inventory for Real-Robotics Transition

**Study:** MPC vs VLA vs Diffusion  
**Date:** August 2026  
**Purpose:** Practical inventory of open-source repos, datasets, and simulators for moving from toy sim to real-robot problems

---

## 1. Simulators with Real-Robot-Like Physics

| Simulator | GPU Parallel | aarch64/DGX Spark | Photorealism | ROS2 | Best For |
|-----------|--------------|-------------------|--------------|------|----------|
| **MuJoCo + MJX** | ✅ (JAX) | ✅ | ❌ | ❌ | Fast policy training, contact physics |
| **Isaac Sim / Isaac Lab** | ✅ | ✅ (DGX Spark) | ✅ | ⚠️ via bridge | Photorealistic training, massive scale |
| **PyBullet** | ❌ | ⚠️ CPU only | ❌ | ❌ | Beginners, simple tasks |
| **ManiSkill3** | ✅ | ⚠️ Linux only | ✅ | ❌ | GPU manipulation, sim2real |
| **Genesis** | ✅ | ✅ | ✅ | ❌ | Ultra-fast sim (emerging) |
| **Gazebo Harmonic** | ✅ | ⚠️ | ⚠️ | ✅ | ROS2 stacks |
| **Drake** | ❌ | ⚠️ | ❌ | ❌ | Control theory, optimization, MPC |
| **TDW / SURREAL** | ✅ | ⚠️ | ✅ | ❌ | Multi-modal, distributed RL |

### Top DGX Spark choices

1. **MuJoCo + MJX** — fastest install, DGX Spark compatible, accurate contact.
2. **ManiSkill3** — GPU-parallelized manipulation with visual baselines.
3. **Isaac Lab** — photorealistic, designed for DGX/Blackwell, but heavy install.
4. **Drake** — for serious MPC/trajectory optimization with arm64 binaries.

---

## 2. Robot Learning Environments

| Environment | Vision | Language | GPU | Sim+Real | Best For |
|-------------|--------|----------|-----|----------|----------|
| **CALVIN** | ✅ | ✅ | ⚠️ | ❌ | Long-horizon language tasks |
| **Meta-World** | ⚠️ | ❌ | ⚠️ | ❌ | Meta-RL, multi-task |
| **LIBERO** | ✅ | ✅ | ⚠️ | ❌ | Lifelong learning |
| **RoboSuite** | ✅ | ❌ | ⚠️ | ❌ | MuJoCo learning, procedural envs |
| **ManiSkill3** | ✅ | ⚠️ | ✅ | ✅ | GPU manipulation, sim2real |
| **RoboDojo** | ✅ | ✅ | ✅ | ✅ | Sim-and-real evaluation |
| **VLA-REPLICA** | ✅ | ✅ | ❌ | ✅ | Low-cost real-world VLA eval |

---

## 3. Pre-Trained Models / Checkpoints

| Model | Params | HuggingFace / GitHub | Notes |
|-------|--------|----------------------|-------|
| **OpenVLA 7B** | 7B | openvla/openvla-7b | MIT, 970K OXE episodes |
| **Octo-Base** | 93M | rail-berkeley/octo-base | 800K trajectories, 13 it/s on 4090 |
| **Octo-Small** | 27M | rail-berkeley/octo-small | Lightweight, 17 it/s on 4090 |
| **π0 / π0.5** | - | gs://openpi-assets/ | Flow-based VLA, open-world generalization |
| **UnifoLM-WMA-0** | - | unitreerobotics/UnifoLM-WMA-0-* | World-model-action, Unitree |
| **Diffusion Policy** | 8-90M | real-stanford/diffusion_policy | Official PushT checkpoints |
| **MoDE** | - | mbreuss/MoDE_Pretrained | Mixture of experts on OXE |
| **Cosmos 3 Policy** | - | nvidia/cosmos3 | Video-to-action, SOTA on LIBERO |

**DGX Spark feasibility:**
- Octo-Small, Octo-Base, OpenVLA 7B fit easily.
- π0/π0.5 fit on 128GB unified memory.
- Motus (2.5B), DreamZero (14B), Cosmos 3 possible but require multi-GPU or long inference.

---

## 4. Datasets

| Dataset | Size | Robot | Format | Best For |
|---------|------|-------|--------|----------|
| **BridgeData V2** | 53,896 traj | WidowX 250 | Raw / RLDS | Accessible multi-task |
| **DROID** | 76K traj, 350h | Various | RLDS | In-the-wild diversity |
| **RH20T** | 110K+ seq | 4 arms | MP4 + NPY | Multi-modal (force, audio, tactile) |
| **Open X-Embodiment** | 1M+ | 22 types | RLDS | Large-scale pretraining |
| **G1-Dex** | 468 eps | Unitree G1 | LeRobot v3 | Humanoid manipulation |

**Common action space:** 7-DOF (x, y, z, roll, pitch, yaw, gripper) as absolute, delta, or velocity.

**Conversion to study format:**
- Use LeRobot `PolicyProcessorPipeline` for normalization/unnormalization.
- Use `rlds_dataset_builder` for custom RLDS → custom formats.
- Use `Seer` or `OpenRobotLab/Seer` for DROID/OXE → `.png`/`.npz`/`.h5`.

---

## 5. Hardware Interface Tools

| Tool | Robot/Camera | ROS2 | Best For |
|------|--------------|------|----------|
| **ROS2 Humble/Jazzy** | All | ✅ Native | Middleware |
| **MoveIt 2** | Manipulators | ✅ | Motion planning, collision |
| **Franka ROS2** | Franka Panda | ✅ | Research arm, FCI |
| **Unitree SDK2** | Go2, B2, H1 | ✅ (CycloneDDS) | Quadrupeds / humanoids |
| **RealSense ROS2** | D400 series | ✅ | RGB-D sensing |
| **ZED ROS2** | ZED series | ✅ | Stereo vision, SLAM |

---

## 6. Quick Installation Reference

```bash
# Simulators
pip install mujoco
pip install maniskill3
pip install robosuite
pip install pybullet

# Models
pip install transformers accelerate
pip install git+https://github.com/octo-models/octo.git
pip install git+https://github.com/openvla/openvla.git

# Datasets
pip install tensorflow-datasets  # for OXE/DROID (RLDS)
pip install lerobot              # for LeRobot format

# MPC tools
conda install -c conda-forge pinocchio
pip install casadi
pip install crocoddyl  # if Pinocchio present
```

---

## 7. Recommended Transition Path

1. **Phase 2 (now):** Add MuJoCo/ManiSkill manipulation tasks to the comparison.
2. **Phase 3:** Integrate real-robot datasets (Bridge, DROID) and evaluate sim-to-real.
3. **Phase 4 (if hardware):** Deploy on Franka via ROS2 / VLA-REPLICA / RoboDojo-RealEval.

---

## 8. Citations

- BridgeData V2: https://bridgedata-v2.github.io/
- DROID: https://droid-dataset.github.io/
- Open X-Embodiment: arXiv:2310.08864
- MuJoCo: https://github.com/google-deepmind/mujoco
- ManiSkill3: https://github.com/haosulab/ManiSkill
- OpenVLA: https://openvla.github.io/
- Octo: https://github.com/octo-models/octo
- π0: https://github.com/Physical-Intelligence/openpi
- Pinocchio: https://github.com/stack-of-tasks/pinocchio
- CasADi: https://github.com/casadi/casadi
