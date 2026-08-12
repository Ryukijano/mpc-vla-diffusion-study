# Blog Narrative Plan: MPC vs VLA vs Diffusion

**Date:** August 2026  
**Purpose:** Narrative and structure for the Hugging Face blog post

---

## 1. Recommended Angle

**"MPC vs VLA vs Diffusion: an open-source controlled study on DGX Spark"**

Justification:
- Emphasizes open-source reproducibility and pre-registered methodology.
- Highlights DGX Spark / GB10 compute, which signals scale.
- Positions the work as a resource for practitioners choosing between control families.
- Leaves room to add World Action Models (WAMs) in Phase 2.

---

## 2. Target Audience

- Primary: HF open-source robotics community (researchers, engineers, VLA/diffusion practitioners).
- Secondary: Academic researchers in robot learning, controls, and sim-to-real.

---

## 3. Key Messages

1. No single family wins everywhere — MPC, VLA, and diffusion each dominate in different regimes.
2. The Simchowitz et al. hypothesis (mechanism is iterative compute + noise, not distribution fitting) is testable and we pre-registered an ablation for it.
3. Classical MPC still wins on latency, hard constraints, and accurate dynamics — but is model-dependent.
4. VLA excels at language conditioning and zero-shot generalization — but is high-latency.
5. Diffusion/flow policies are flexible and can capture multi-modal actions, but inference is expensive.
6. World Action Models (WAMs) are a fourth family we will test in Phase 2.
7. Sim-to-real visual robustness is a major unsolved gap.
8. Everything is open source and reproducible; the goal is evidence, not hype.

---

## 4. Story Arc

1. **Hook** (150 words): "Do we need diffusion in robotics?" — the Simchowitz question.
2. **Background** (200 words): three/four families, strengths/weaknesses.
3. **Study design** (250 words): pre-registered protocols, benchmarks, metrics, fairness controls.
4. **Quick test results** (200 words): smoke-test on 2-D reaching, caveated.
5. **Ablation results** (300 words): EXP-001 mechanism ablation, tiny-data results.
6. **Pareto analysis** (200 words): latency vs success trade-off.
7. **Roadmap to real robots** (250 words): WAMs, real MPC, real vision, sim-to-real.
8. **Reproducibility / artifacts** (200 words): GitHub, planned HF Hub assets.
9. **Call to action** (150 words): clone, run, contribute.

---

## 5. Figures and Tables

### Use existing quick-test assets
- `results/quick_test/report/figures/comparison_success_rate.png`
- `results/quick_test/report/figures/comparison_latency.png`
- `results/quick_test/report/figures/pareto_latency_vs_success.png`
- `results/quick_test/ablation/figures/ablation_success_rate.png`
- `results/quick_test/ablation/figures/ablation_latency.png`

### Existing tables
- `results/quick_test/report/master_comparison_table.csv`
- `results/quick_test/ablation/ablation_aggregated.csv`

### New visuals to create
- Architecture diagram: MPC vs VLA vs Diffusion vs WAM.
- Phase roadmap: toy sim → real-physics sim → real datasets → real hardware.
- Thumbnail: 1300x650 px, four-panel concept.

---

## 6. Code Snippets

1. **Clone and run quick test**:
   ```bash
   git clone https://github.com/Ryukijano/mpc-vla-diffusion-study.git
   cd mpc-vla-diffusion-study
   bash scripts/run_quick_test.sh
   ```

2. **Load MPC baseline** (from `mpc_baselines_repo/src`):
   ```python
   from src.linear_mpc import LinearMPC
   from src.utils import PointMass2D
   ```

3. **Run ablation**:
   ```bash
   conda run -n mpc_vla python run_ablation.py \
       --benchmark reaching --seeds 0 --episodes 5 --epochs 10 --num-demos 10
   ```

---

## 7. Titles (ranked)

1. "MPC vs VLA vs Diffusion: An Open-Source Study on the DGX Spark"
2. "Do You Need Diffusion? A Controlled Comparison of Robot Control Architectures"
3. "Beyond the Hype: MPC, VLA, Diffusion, and World Action Models Compared"
4. "The Robot Control Pareto Frontier: Latency, Generalization, and Safety"
5. "From Toy Sim to Real Robots: A Reproducible Study of Control Policies"

---

## 8. Authors

- Ryukijano (study lead)
- devin-ai (Cognition research agent)
- AIMS Group, University of Leeds (compute / affiliation)

---

## 9. Length

- 2,000–2,700 words.
- 8-10 sections.
- 6-8 code/figure blocks.

---

## 10. Publishing Path

1. Create HF Hub artifacts (model, dataset, optional Space).
2. Write final draft in `docs/blogging/hf_blog_draft.md`.
3. Upload images to `huggingface/documentation-images` or keep repo-relative paths for community blog.
4. Create 1300x650 thumbnail.
5. Publish via `huggingface.co/new-blog` (community blog).
6. Add repo README link and HF collection.

---

## 11. Accuracy guardrails

From `claim_audit.md`, the blog must avoid:
- Claiming MPC dominates based on quick test.
- Claiming MIP matches full DDPM (not proven in quick test).
- Listing benchmarks that are not implemented (RoboMimic, CALVIN, LIBERO, DMControl not yet in repo).
- Stating the full Pareto frontier without running EXP-004.
- Overstating real-robot results before EXP-005–008.

Use caveats: "in our smoke test," "pre-registered to test," "so far," "Phase 2 will add."
