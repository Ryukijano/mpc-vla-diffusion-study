# Real Robotic Vision and Sim-to-Real for Manipulation

**Study:** MPC vs VLA vs Diffusion Comparison  
**Date:** August 2026  
**Purpose:** Synthesize 2024-2026 research on real-world vision, sim-to-real, benchmarks, and deployment pipelines

---

## 1. Real Robot Vision Challenges

### 1.1 Camera Calibration

- **Hand-eye calibration** remains a bottleneck. Marker-based methods (AprilTags, checkerboards) are fragile under occlusion.
- **Kalib** (arXiv:2408.10562): Markerless calibration using visual foundation models and PnP.
- **EasyHeC++** (arXiv:2410.09293): Fully automatic, marker-free, training-free hand-eye calibration via differentiable rendering.
- **Key lesson:** Calibration drift, occlusion, and lighting are dominant failure modes in real-world deployment.

### 1.2 Lighting, Occlusion, Motion Blur

| Challenge | Impact | Mitigation |
|-----------|--------|------------|
| Lighting | Texture/colors shift, specular highlights | Domain randomization, PNDR, ISP-aware augmentation |
| Occlusion | Robot/object blocks target | Multi-view fusion, temporal persistence, active vision |
| Motion blur | Fast motion corrupts images | High frame rate, global shutter, short exposure |

### 1.3 Egocentric vs Third-Person vs Multi-View

- **Egocentric (wrist):** high precision, viewpoint-specific, needs hand-eye calibration.
- **Third-person (overhead):** scene context, less occlusion, less precision.
- **Multi-view fusion (Look Closer, 4Diff, 3D-MVP):** combines both; strong gains on hammer manipulation (75% vs 13% single-view).

### 1.4 RGB vs Depth vs Point Cloud vs Multi-View

- **RGB:** best for VLA/semantic understanding; fails on dark/reflective surfaces.
- **Depth:** lighting-invariant; noisy on transparent/reflective surfaces.
- **Point cloud:** full 3D; high compute.
- **Multi-view:** best for complex manipulation but requires calibration/sync.

---

## 2. Sim-to-Real Gap

### 2.1 Domain Randomization (DR)

Randomize textures, lighting, camera pose, friction, mass, contact stiffness. Recent work optimizes DR parameters using proxy tasks (arXiv:2307.15320). **Risk:** too narrow → overfitting; too wide → underfitting.

### 2.2 Photorealistic Rendering

- **PNDR** (arXiv:2210.12682): neural ray-tracing for efficient photorealistic augmentation.
- **3D Gaussian Splatting (ViserDex, arXiv:2604.11138):** domain randomization in Gaussian space.
- **SimWeaver-Sim** (CVPR 2026): measurement-backed deformable-object simulator.
- **Isaac Lab PPISP:** physically plausible ISP pipeline in simulation.

### 2.3 Physics-Realistic Modeling

- Rigid-rigid contact is mature; **rigid-deformable** contact remains hard.
- **TacEx** (arXiv:2411.04776): integrates GIPC soft-body simulator with Isaac Sim for GelSight tactile sim.
- **Recommendation:** standard MuJoCo/PhysX for rigid manipulation; TacEx for deformable/contact-rich tasks.

### 2.4 ISP-Aware Photometric Augmentation

Standard RGB augmentation after ISP is unrealistic. Apply augmentation **before** ISP for realistic pixel distributions.

- **Rawgment** (CVPR 2023): RAW image augmentation with noise accounting.
- **SimWeaver-Real:** ISP-aware photometric augmentation for deformable manipulation.
- **Pipeline:** responsivity → exposure → vignetting → color homography → CRF → noise → uint8.

### 2.5 Adaptation / Fine-Tuning on Real Data

- **Zero-shot:** SimWeaver (200 sim demos → 91% avg real success on deformable tasks).
- **Fine-tuning:** 10-100 real demos usually sufficient.
- **Sim-real co-training (RL-Co, arXiv:2602.12628):** +24% OpenVLA, +20% π0.5 over real-only.

---

## 3. Real-World Benchmarks and Datasets (2024-2026)

### 3.1 RoboDojo (arXiv:2607.04434)

- **42 sim + 18 real tasks**
- Five capability dimensions: generalization, memory, precision, long-horizon, open-vocabulary
- Heterogeneous parallel sim in Isaac Sim
- RoboDojo-RealEval: reproducible real-world evaluation with cloud access
- 30 integrated policies
- https://robodojo-benchmark.com

### 3.2 VLA-REPLICA (arXiv:2605.20774)

- Low-cost real-world benchmark using SO-101 manipulator
- ID and OOD evaluation protocols
- Supports {act, smolvla, dit, xvla, pi0, pi05}
- https://irvlutd.github.io/VLAReplica/

### 3.3 SimWeaver (arXiv:2606.15338)

- Zero-shot RGB sim-to-real for deformable manipulation
- 200 sim demos → >80% per-task, 91% avg real success
- 100% success on silk grasping under visual distribution shifts
- ISP-aware augmentation is key

### 3.4 RL-Co (arXiv:2602.12628)

- Two-stage: SFT warm-start on mixed sim+real, then sim RL with real-data anchoring loss.
- Gains: +24% OpenVLA, +20% π0.5.

### 3.5 Datasets

| Dataset | Size | Robots | Format | Best For |
|---------|------|--------|--------|----------|
| **BridgeData V2** | 53,896 traj | WidowX | Raw/RLDS | Multi-task, accessible |
| **DROID** | 76k traj, 350h | Various | RLDS | In-the-wild diversity |
| **RH20T** | 110k+ seq | 4 arms | MP4+NPY | Multi-modal (force, tactile, audio) |
| **Open X-Embodiment** | 1M+ | 22 types | RLDS | Large-scale pretraining |
| **G1-Dex** | 468 eps | Unitree G1 | LeRobot v3 | Humanoid manipulation |

**Common 7-DOF action space:** x, y, z, roll, pitch, yaw, gripper (absolute/delta/velocity).

---

## 4. Practical Deployment Pipeline

### 4.1 Image Preprocessing

1. Resize to model input (e.g., 224×224)
2. Normalize (ImageNet or dataset-specific)
3. Temporal stacking (if history-dependent)
4. Optional: ARRO masking (arXiv:2505.08627), ISP-aware augmentation

### 4.2 Action Normalization / Unnormalization

**Critical safety issue:** same checkpoint + wrong normalization = different physical actions (arXiv:2606.03724).

Use LeRobot `PolicyProcessorPipeline` with modes: `MEAN_STD`, `MIN_MAX`, `QUANTILES`, `QUANTILE10`.

```python
# Training
normalized_action = (action - mean) / std
# Inference
raw_action = model_output * std + mean
```

### 4.3 Control Frequency and Latency

| Method | Typical Latency | Max Control Freq | Strategy |
|--------|----------------|------------------|----------|
| Classical MPC | 1-10 ms | 100-1000 Hz | Direct control |
| Diffusion (T=16) | 10-50 ms | 20-100 Hz | Action chunking |
| VLA (standard) | 50-200 ms | 5-20 Hz | Open-loop chunking |
| VLA (Reflex) | 20-40 ms | 25-50 Hz | Streaming inference |
| VLA (FLASH) | 8-15 ms | 65-125 Hz | Speculative decoding |

Recent speedups:
- **Reflex** (arXiv:2607.14695): 2.58× speedup, 50Hz streaming
- **TIDAL** (arXiv:2601.14945): dual-frequency macro/micro loops
- **LAGO** (arXiv:2606.17982): latency-aware async diffusion + collision-free planning
- **FLASH** (2026): 58ms → 7.8ms for diffusion VLAs

### 4.4 Safety Layers

1. Hard velocity/joint limits
2. Collision prediction (SDF or learned)
3. **MPC fallback** for constraint satisfaction
4. HJ reachability safety filters (arXiv:2509.14758)
5. Emergency stop on failures

```
Observation → VLA/Diffusion → Action Candidate → Safety Filter (MPC/HJ) → Robot
```

### 4.5 End-to-End Pipeline

```
Camera → Preprocessing → Encoder → Policy → Postprocessing → Safety Filter → Robot
```

**Latency budget (typical VLA):** 28-102 ms total → ~15-35 Hz control.

---

## 5. Recommendations for the Study

### Phase 2 (Real-Physics Simulators)

- **MuJoCo + MJX** (DGX Spark aarch64 compatible)
- **ManiSkill3** (GPU-parallelized manipulation)
- **Isaac Lab** (photorealistic, DGX Spark supported)

### Datasets to Use

- **BridgeData V2** for initial experiments
- **DROID / Open X-Embodiment** for large-scale training
- **VLA-REPLICA** if real robot available

### Sim-to-Real Strategy

1. Domain randomization in sim
2. Photorealistic rendering (PNDR or 3DGS)
3. ISP-aware augmentation (Rawgment-style)
4. Fine-tune on real data or use RL-Co

### Evaluation Metrics

- Success rate, latency, data efficiency
- OOD robustness, manifold adherence
- Constraint violation rate
- Safety filter activation rate

---

## 6. Key References

- **Kalib:** arXiv:2408.10562
- **EasyHeC++:** arXiv:2410.09293
- **PNDR:** arXiv:2210.12682
- **Rawgment:** CVPR 2023
- **SimWeaver:** arXiv:2606.15338
- **RL-Co:** arXiv:2602.12628
- **RoboDojo:** arXiv:2607.04434
- **VLA-REPLICA:** arXiv:2605.20774
- **Reflex:** arXiv:2607.14695
- **TIDAL:** arXiv:2601.14945
- **LAGO:** arXiv:2606.17982
- **Same Weights, Different Robot:** arXiv:2606.03724
- **Safety Filters:** arXiv:2509.14758
- **LeRobot processors:** https://github.com/huggingface/lerobot
