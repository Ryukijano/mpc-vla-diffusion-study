# World Action Models (WAMs) for Real-World Robotics: Comprehensive Briefing

**Study Location:** `/home/aimsgroupuol/AIMSgeneral/Gyanateet/mpc_vla_diffusion_study/`  
**Date:** August 2026  
**Research Scope:** MPC vs VLA vs Diffusion comparison study

---

## 1. What are World Action Models (WAMs)?

### Definition and Core Concept

A **World Action Model (WAM)** is an embodied predictive-action model that makes a forecast of the future available to action generation. Unlike plain Vision-Language-Action (VLA) models that map observations directly to actions, or diffusion policies that generate actions without explicit world modeling, WAMs explicitly learn environment dynamics and use predicted future states to inform action decisions.

**Key distinction:**
- **VLA:** Observation + Language → Action (reactive mapping)
- **Diffusion Policy:** Observation → Action (generative sampling)
- **World Model:** Observation + Action → Future State (prediction only)
- **WAM:** Observation + Language → (Future State + Action) (joint prediction with action coupling)

### Core Architectural Paradigms

According to the WAM survey (arXiv:2606.20781), WAMs fall into three design philosophy families:

1. **Render-and-Decode:** Generates full visual futures (pixels/latents) and decodes actions from them
2. **Latent-Only:** Predicts latent representations of futures without full rendering
3. **Video-Generation-Free:** Uses predictive supervision outside video generation (e.g., inverse dynamics, affordance maps)

### Foundational Models (Pre-WAM Era)

These models established the concepts that WAMs build upon:

#### DreamerV3
- **arXiv:** 2301.04104
- **Contribution:** General RL algorithm based on world models that masters 150+ diverse tasks with fixed hyperparameters; first to collect diamonds in Minecraft from scratch
- **Architecture:** Learns a world model from experience, uses it to train actor-critic policy from imagined trajectories
- **Relevance:** Demonstrates that world models can enable efficient sample-efficient learning across domains
- **GitHub:** https://github.com/danijar/dreamerv3

#### UniPi
- **Contribution:** Unified policy for imitation with planning; demonstrates that planning with world models can improve sample efficiency
- **Architecture:** Uses world model rollouts for planning during policy learning
- **Relevance:** Early demonstration of imagine-then-execute paradigm

#### GATO (A Generalist Agent)
- **arXiv:** 2205.06175
- **Contribution:** Single 1.2B parameter transformer that performs 600+ tasks including Atari, image captioning, chat, and robot arm control
- **Architecture:** Multi-modal, multi-task, multi-embodiment generalist policy; serializes all data into flat token sequences
- **Relevance:** Early generalist agent showing cross-modal transfer, but lacks explicit world modeling
- **URL:** https://deepmind.google/blog/a-generalist-agent/

#### VIMA (VisuoMotor Attention)
- **arXiv:** ICML 2023
- **Contribution:** General robot manipulation with multimodal prompts (text + visual tokens)
- **Architecture:** Encoder-decoder transformer with pretrained T5, object-centric representation
- **Relevance:** Demonstrates strong scaling properties and data efficiency for multimodal prompts
- **GitHub:** https://github.com/vimalabs/VIMA
- **HuggingFace:** https://huggingface.co/VIMA/VIMA

---

## 2. Key Papers (2024-2026) with arXiv IDs and Contributions

### π0 (pi-zero)
- **arXiv:** 2410.24164
- **Date:** October 2024
- **Authors:** Physical Intelligence team (Black, Brown, Driess, et al.)
- **Contribution:** Vision-Language-Action flow model for general robot control using flow matching on pre-trained VLM backbone; enables zero-shot task performance, language instruction following, and skill acquisition via fine-tuning across diverse robotic platforms
- **Architecture:** Pre-trained VLM backbone + separate action expert producing continuous actions via flow matching
- **Performance:** Controls robots at up to 50 Hz for dexterous tasks
- **Code:** https://github.com/Physical-Intelligence/openpi
- **URL:** https://arxiv.org/abs/2410.24164

### π0.5 (pi-oh-five)
- **arXiv:** 2504.16054
- **Venue:** PMLR 305 (CoRL 2025)
- **Date:** 2025
- **Authors:** Physical Intelligence team (Black, Brown, et al.)
- **Contribution:** Upgraded π0 with open-world generalization using co-training on heterogeneous tasks (multiple robots, high-level semantic prediction, web data); first end-to-end learning-enabled robotic system to perform long-horizon dexterous manipulation in entirely new homes
- **Architecture:** Based on π0 with knowledge insulation training; uses hybrid multi-modal examples (image observations, language commands, object detections, semantic subtask prediction, low-level actions)
- **Data:** 400 hours mobile manipulator data + heterogeneous sources
- **Code:** https://github.com/Physical-Intelligence/openpi
- **URL:** https://proceedings.mlr.press/v305/black25a.html

### Motus
- **arXiv:** 2512.13030
- **Venue:** CVPR 2026
- **Date:** December 2025
- **Authors:** Bi, Tan, Xie, et al. (Tsinghua University)
- **Contribution:** Unified latent action world model with Mixture-of-Transformers (MoT) architecture integrating three experts (understanding, video generation, action); achieves +15% improvement over X-VLA and +45% over π0.5 in simulation, +11-48% in real-world
- **Architecture:** MoT with UniDiffuser-style scheduler for flexible switching between modeling modes (world models, VLAs, inverse dynamics, video generation, video-action joint prediction); uses optical flow to learn latent actions
- **Training:** Three-phase pipeline with six-layer data pyramid
- **Code:** https://github.com/thu-ml/Motus
- **HuggingFace:** https://huggingface.co/motus-robotics/Motus_Wan2_2_5B_pretrain
- **URL:** https://arxiv.org/abs/2512.13030

### UnifoLM-WMA-0 (Unitree)
- **Date:** 2025
- **Authors:** Unitree Robotics
- **Contribution:** Open-source world-model-action architecture spanning multiple robotic embodiments; world model operates as (a) Simulation Engine for synthetic data generation, (b) Policy Enhancement by predicting future interaction processes
- **Architecture:** Video generation model fine-tuned as world model; operates in decision-making mode and simulation mode
- **Data:** Fine-tuned on Open-X dataset and five Unitree open-source datasets
- **Code:** https://github.com/unitreerobotics/unifolm-world-model-action
- **HuggingFace:** https://huggingface.co/unitreerobotics/UnifoLM-WMA-0-Base
- **URL:** https://unigen-x.github.io/unifolm-world-model-action.github.io/

### DreamZero (NVIDIA)
- **arXiv:** 2602.15922
- **Date:** February 2026
- **Authors:** Ye, Ge, Zheng, et al. (NVIDIA)
- **Contribution:** 14B World Action Model built on pretrained video diffusion backbone; achieves 2× improvement in generalization to new tasks/environments vs state-of-the-art VLAs; enables real-time closed-loop control at 7Hz; demonstrates cross-embodiment transfer with 10-20 minutes of data
- **Architecture:** Joint Video-Action Diffusion Transformer (DiT) predicting both future visual tokens and robot actions
- **Key Innovation:** Shifts action learning from dense state-action imitation to inverse dynamics
- **Code:** https://github.com/dreamzero0/dreamzero
- **URL:** https://arxiv.org/abs/2602.15922

### Dexterous World Models (DWM)
- **arXiv:** CVPR 2026
- **Authors:** Kim et al. (SNU VCLab)
- **Contribution:** Scene-action-conditioned video diffusion framework for simulating embodied dexterous actions in static 3D scenes; enables realistic human-scene interactions (grasping, opening, moving objects) while maintaining camera and scene consistency
- **Architecture:** Conditions video generation on (1) static scene renderings with camera trajectory, (2) egocentric hand mesh renderings encoding geometry and motion
- **Data:** Hybrid interaction video dataset (synthetic egocentric + real-world fixed-camera)
- **Code:** https://github.com/snuvclab/dwm
- **URL:** https://arxiv.org/abs/2xxx.xxxxx (CVPR 2026)

### DWM (Decomposed World Model)
- **arXiv:** 2607.18715
- **Authors:** (Separate from Dexterous World Models)
- **Contribution:** Supervision-level framework that decomposes latent transitions into action-invariant world effects and action-driven components; achieves 13.1% mean absolute improvement in CEM planning success on W-variant benchmarks
- **Architecture:** Augments predictor with auxiliary world head (action-invariant) + orthogonality constraint with original prediction head
- **Relevance:** Addresses action-effect entanglement in latent world models
- **URL:** https://arxiv.org/abs/2607.18715

### Helix (Figure AI)
- **Date:** 2025-2026
- **Authors:** Figure AI
- **Contribution:** First VLA to control full humanoid upper body including individual fingers from one set of weights; first to run two robots simultaneously from one set of weights; first VLA to run entirely on embedded low-power GPUs
- **Architecture:** Dual-system VLA - System 2 (7B VLM, 7-9 Hz) for scene understanding/language, System 1 (80M policy, 200 Hz) for fast reactive control; Helix-02 adds System 0 (1 kHz neural prior trained on 1000+ hours human motion data)
- **Status:** Proprietary (not open-source)
- **URL:** https://www.figure.ai/helix

### Octo
- **arXiv:** 2405.12213
- **Date:** May 2024
- **Authors:** Octo Model Team (Ghosh, Walke, Pertsch, et al.)
- **Contribution:** Open-source transformer-based diffusion policy pretrained on 800k trajectories from Open X-Embodiment dataset; supports flexible task/observation definitions and efficient fine-tuning to new observation/action spaces
- **Architecture:** Transformer with modular attention structure; predicts 7-dimensional actions 4 steps into future using diffusion policy
- **Models:** Octo-Small (27M params), Octo-Base (93M params)
- **Code:** https://github.com/octo-models/octo
- **URL:** https://arxiv.org/abs/2405.12213

### NVIDIA Cosmos 3
- **Contribution:** Open omnimodel with native reasoning, world and action generation built on Mixture-of-Transformers; Cosmos Policy fine-tunes video models for visuomotor control achieving SOTA on LIBERO (98.5%) and RoboCasa (67.1%)
- **Architecture:** Unified framework subsuming VLMs, video generators, world simulators, and WAMs
- **Code:** https://github.com/NVIDIA/cosmos
- **URL:** https://www.nvidia.com/en-us/ai/cosmos/

### τ₀-World Model (SII Research)
- **Contribution:** Unified video-action world model integrating policy learning, video prediction, and action evaluation; trained on 27,300 hours of diverse data
- **Architecture:** Video Action Model (VAM) + Action-conditioned Video Simulator on Wan-2.2 backbone
- **HuggingFace:** https://huggingface.co/sii-research/tau-0-wm

---

## 3. How WAMs Relate to MPC vs VLA vs Diffusion Question

### Internal MPC Usage

**WAMs that use MPC internally:**

1. **WorldPlanner (arXiv:2511.03077):** Uses action-conditioned visual world model with Monte Carlo Tree Search (MCTS) planner, executed via zeroth-order Model Predictive Controller (MPC)
2. **PointWorld:** Action-conditioned 3D world model that can be integrated into MPC framework for manipulation; achieves real-time (0.1s) inference
3. **World Action Planner (arXiv:2607.27599):** Leverages VLM reasoning + action-conditioned world model for propose-simulate-refine planning loop
4. **ACID (arXiv:2607.02403):** Decision-time planning framework with action consistency via inverse dynamics for planning with world models
5. **Temporal-Distance-JEPA (arXiv:2607.25337):** Plan-aware representation learning for latent world model predictive control

**WAMs that do NOT use MPC internally:**

- **π0 / π0.5:** VLM + action head (flow matching), no explicit MPC
- **Motus:** Unified latent action world model with UniDiffuser-style scheduler, no MPC
- **DreamZero:** Joint video-action diffusion, no MPC
- **Helix:** Dual-system VLA (S2 VLM + S1 policy), no MPC
- **Octo:** Transformer-based diffusion policy, no MPC
- **UnifoLM-WMA-0:** World model + action head, no MPC

### Where Diffusion / Flow Matching / Autoregression Wins

**Diffusion/Flow Matching advantages:**
- **Multi-modal action distributions:** When tasks have multiple valid solution strategies (e.g., different grasping approaches)
- **High-precision manipulation:** Flow matching provides smooth, precise action trajectories (ActionFlow, RFMP, ManiFlow)
- **Physical generalization:** WAMs built on video diffusion (DreamZero, Motus) show 2× improvement in generalization to novel environments vs VLAs
- **Inverse dynamics learning:** Shifting from state-action imitation to inverse dynamics (DreamZero) enables better cross-embodiment transfer

**Autoregression advantages:**
- **Language conditioning:** VLM backbones excel at semantic understanding and instruction following
- **Long-horizon reasoning:** Autoregressive models can maintain context over extended sequences
- **Zero-shot transfer:** Pretrained on web-scale data enables generalization to novel objects/scenes

**Empirical findings:**
- **Simchowitz et al. (arXiv:2512.01809):** GCPs' success is from iterative compute + noise injection, not multi-modal distribution fitting; MIP (2-step regression) matches full flow GCPs
- **Counter-evidence (DAWN CVPR 2026):** Claims pixel motion diffusion IS what we need for robot control
- **Flow matching variants (ActionFlow, VITA, ManiFlow):** Show 1.5-2.3× faster inference than conventional diffusion with conditioning

### Where MPC-like Planning Wins

**MPC advantages:**
- **Hard safety constraints:** Explicit constraint handling (collision avoidance, joint limits, workspace constraints)
- **Real-time control:** Classical MPC can operate at >100Hz for simple dynamics
- **Analytical dynamics:** When accurate dynamics models are available (e.g., industrial robots with known kinematics/dynamics)
- **Long-horizon planning:** Model-based planning can optimize over extended horizons using world model rollouts

**WAM + MPC hybrid advantages:**
- **WorldPlanner:** MCTS planning with world model rollouts significantly outperforms BC baselines
- **Diffusion warm-start MPC:** Diffusion provides good initial guesses for MPC optimization
- **ACID:** Action consistency constraints improve planning reliability with world models

**Trade-offs:**
- **Compute:** MPC requires solving optimization problems online; WAMs amortize computation into training
- **Generalization:** Classical MPC fails on novel objects/scenes without re-modeling; WAMs generalize from data
- **Latency:** Pure MPC is fastest; WAMs with video generation are slowest; hybrid approaches in middle

---

## 4. Open-Source Code and Weights

### Models Available for Download (Inference-Only)

| Model | HuggingFace | GitHub | Hardware Required | Inference Only? |
|-------|-------------|--------|------------------|-----------------|
| **π0** | `gs://openpi-assets/checkpoints/pi0_base` | https://github.com/Physical-Intelligence/openpi | GPU (4090+) | Yes (fine-tuning available) |
| **π0.5** | `gs://openpi-assets/checkpoints/pi05_base` | https://github.com/Physical-Intelligence/openpi | GPU (4090+) | Yes (fine-tuning available) |
| **Motus** | `motus-robotics/Motus_Wan2_2_5B_pretrain` | https://github.com/thu-ml/Motus | GPU (multi-GPU for 2.5B) | Yes (fine-tuning available) |
| **UnifoLM-WMA-0-Base** | `unitreerobotics/UnifoLM-WMA-0-Base` | https://github.com/unitreerobotics/unifolm-world-model-action | GPU | Yes (fine-tuning available) |
| **UnifoLM-WMA-0-Dual** | `unitreerobotics/UnifoLM-WMA-0-Dual` | https://github.com/unitreerobotics/unifolm-world-model-action | GPU | Yes (fine-tuning available) |
| **Octo-Base** | `rail-berkeley/octo-base` | https://github.com/octo-models/octo | GPU (1x 4090 @ 13 it/sec) | Yes (fine-tuning available) |
| **Octo-Small** | `rail-berkeley/octo-small` | https://github.com/octo-models/octo | GPU | Yes (fine-tuning available) |
| **VIMA** | `VIMA/VIMA` | https://github.com/vimalabs/VIMA | GPU | Yes |
| **DreamZero** | - | https://github.com/dreamzero0/dreamzero | GPU (multi-GPU for 14B) | Yes (inference server) |
| **Cosmos 3** | `nvidia/cosmos3` collection | https://github.com/NVIDIA/cosmos | NVIDIA GPU | Yes |
| **PointWorld** | `nvidia/PointWorld_models` | - | GPU | Yes |
| **τ₀-World Model** | `sii-research/tau-0-wm` | - | GPU | Yes |
| **WorldDiT** | `bageldotcom/worlddit` | - | GPU | Yes |
| **RIO-2** | `hoguai/RIO-2` | - | GPU | Yes |

### Models That Can Run on DGX Spark / GB10

**Recommended for DGX Spark / GB10 deployment:**

1. **Octo-Base (93M params):** Designed for consumer GPUs (1x 4090), will run efficiently on DGX
2. **Octo-Small (27M params):** Lightweight, suitable for embedded deployment
3. **VIMA:** Transformer-based, moderate compute requirements
4. **Motus (2.5B params):** Requires multi-GPU, suitable for DGX Spark
5. **DreamZero (14B params):** Requires multi-GPU, suitable for DGX Spark
6. **Cosmos 3:** Optimized for NVIDIA hardware, ideal for DGX

---

## 5. Gaps in Current Study

### Missing WAM Comparisons

The current MPC vs VLA vs Diffusion study does **not** include:

1. **World Action Models:** No WAM baselines (DreamZero, Motus, UnifoLM-WMA-0, Cosmos Policy)
2. **Joint video-action modeling:** No evaluation of models that predict both future states and actions
3. **World model rollouts for planning:** No comparison of imagine-then-execute paradigms
4. **Cross-embodiment transfer:** No evaluation of how well models transfer between different robot platforms
5. **Physical generalization metrics:** No measurement of generalization to novel physics/dynamics

### Missing Hybrid Approaches

1. **WAM + MPC:** Use world model for constraint-aware planning
2. **WAM warm-start:** Use WAM to initialize MPC optimization
3. **VLA + WAM:** Use VLM for semantic understanding, WAM for physical prediction
4. **Diffusion + WAM:** Use diffusion for action sampling, WAM for feasibility checking

### Research Questions to Add

**RQ9 (WAM Advantage):** Do WAMs provide measurable advantages over pure VLA/diffusion policies for physical generalization?
- **H9a:** WAMs outperform VLA/diffusion on OOD physics and novel environments
- **H9b:** WAM advantage comes from predictive supervision, not architecture

**RQ10 (World Model Fidelity):** How accurate must world model predictions be to provide action benefits?
- **H10:** Coarse future prediction suffices; precise pixel-level prediction not required

**RQ11 (Planning Horizon):** What is the optimal prediction horizon for WAM-based control?
- **H11:** Short-horizon prediction (1-5 steps) optimal for real-time control; long-horizon useful for high-level planning

---

## 6. References and URLs

- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://proceedings.mlr.press/v305/black25a.html
- Motus: https://arxiv.org/abs/2512.13030
- DreamZero: https://arxiv.org/abs/2602.15922
- Octo: https://arxiv.org/abs/2405.12213
- UnifoLM-WMA: https://github.com/unitreerobotics/unifolm-world-model-action
- WorldPlanner: https://arxiv.org/abs/2511.03077
- ACID: https://arxiv.org/abs/2607.02403
- OpenPi: https://github.com/Physical-Intelligence/openpi
- NVIDIA Cosmos: https://www.nvidia.com/en-us/ai/cosmos/
