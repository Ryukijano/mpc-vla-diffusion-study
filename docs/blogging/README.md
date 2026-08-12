# Hugging Face Blog Master Plan

**Project:** MPC vs VLA vs Diffusion comparison study  
**Repository:** https://github.com/Ryukijano/mpc-vla-diffusion-study  
**Date:** 2026-08-12  
**Goal:** Publish a high-quality Hugging Face blog post (community or official) that announces the open-source study and invites reproducible research.

---

## 1. Why a HF blog?

- Timely topic: comparing MPC, VLA, Diffusion, and now WAMs.
- Strong HF robotics traffic around LeRobot, π0, SmolVLA, diffusion policy.
- Blog can link to GitHub, and later to HF Hub models/datasets/Spaces.
- Establishes canonical write-up before an arXiv paper.
- Attracts contributors and real-robot collaborators for Phase 2.

---

## 2. Blog format and channel

**Recommended channel:** Hugging Face **Community Blog** (https://huggingface.co/new-blog)

- Fastest path for an external contributor.
- Appears on the main blog page alongside official posts.
- Can be updated as experiments progress.
- No PR review queue required.

**Alternative:** PR to `huggingface/blog` only if a direct HF collaboration develops.

**Article length:** 2,000–2,700 words.

**Frontmatter:**
```yaml
---
title: "MPC vs VLA vs Diffusion: An Open-Source Study of Robot Control Families"
thumbnail: /blog/assets/mpc-vla-diffusion-study/thumbnail.png
authors:
- user: Ryukijano
- user: devin-ai
---
```

---

## 3. Target audience and key messages

**Audience:**
- Open-source robot-learning researchers and engineers on HF.
- VLA/diffusion/MPC practitioners choosing an architecture.

**Key messages:**
1. We pre-registered 12 research questions and 8 experiments to compare control families fairly.
2. The quick smoke test is a harness check, not a conclusion: MPC solvers 100% on 2-D reaching; learned policies need more data.
3. The Simchowitz / "Much Ado About Noising" mechanism ablation (EXP-001) is the scientific core.
4. World Action Models (WAMs), real MPC, and real-robot vision are the next frontiers (EXP-005–008).
5. Everything is open source and runnable on a DGX Spark; the community can reproduce and extend it.

---

## 4. Required artifacts before publication

### Must have
- [x] GitHub repository public and documented.
- [x] `README.md` with setup and quick-test instructions.
- [x] `LICENSE` (MIT) at repo root.
- [x] Pre-registered research questions and experiment protocols.
- [x] Quick-test results and figures committed.
- [x] Draft blog post in `docs/blogging/hf_blog_draft.md`.

### Strongly recommended
- [ ] One HF Hub **model** repo with a trained SmallVLA or MIP checkpoint.
- [ ] One HF Hub **dataset** repo with quick-test expert demonstrations.
- [ ] One HF Hub **Space** showing the comparison figures or a small interactive demo.
- [ ] A 1300×650 **thumbnail** image (`.png` or `.gif`).
- [ ] All body figures uploaded to `huggingface/documentation-images` or served from the repo.

### Optional but high impact
- [ ] Colab notebook reproducing the quick test.
- [ ] Twitter/X thread and LinkedIn post linking to the blog.
- [ ] arXiv preprint later, then link to `hf.co/papers/<arxiv_id>`.

---

## 5. Suggested Hub artifact IDs

| Artifact | Suggested ID | Format | Status |
|----------|--------------|--------|--------|
| Collection | `Ryukijano/mpc-vla-diffusion-study` | HF Collection | planned |
| SmallVLA quick checkpoint | `Ryukijano/smallvla-mpc-vla-diffusion-quick` | PyTorch `.pt` (~340 MB) | needs training |
| Tiny DiffusionPolicy | `Ryukijano/diffusion-policy-mpc-vla-diffusion-quick` | PyTorch `.pt` (~5.6 MB) | needs training |
| MPC expert demos | `Ryukijano/mpc-expert-demos-quick-test` | `.npz` (LeRobot optional) | needs generation |
| Plot gallery Space | `Ryukijano/mpc-vla-diffusion-plot-gallery` | Gradio | planned |
| Interactive arena Space | `Ryukijano/mpc-vla-diffusion-arena` | Gradio | planned |

---

## 6. Blog post structure (from `hf_blog_draft.md`)

1. **TL;DR**
2. **The debate: do we need diffusion in robotics?**
3. **Three families, one shared harness** (+ WAMs as Phase 2)
4. **How we designed the study** (pre-registration, fairness controls)
5. **Quick smoke-test results** (caveated, n = 1 seed, 5 episodes)
6. **Ablation: what makes a diffusion policy work?**
7. **The latency-success Pareto frontier**
8. **From toy sim to real robots** (Phase 2 roadmap)
9. **Try it yourself** (clone, run quick test, code snippets)
10. **Artifacts and reproducibility**
11. **What is next?**
12. **Call to action**

---

## 7. Figures and tables to include

- `results/quick_test/report/figures/comparison_success_rate.png`
- `results/quick_test/report/figures/comparison_latency.png`
- `results/quick_test/report/figures/pareto_latency_vs_success.png`
- `results/quick_test/ablation/figures/ablation_success_rate.png`
- `results/quick_test/ablation/figures/ablation_latency.png`
- Master comparison table
- Ablation table

**To create:**
- 1300×650 thumbnail (four families or Pareto frontier).
- Architecture diagram for the four families.
- Phase roadmap diagram.

---

## 8. Claim-safety checklist (from `claim_audit.md`)

**Do say:**
- "In our smoke test, every MPC variant solved the 2-D reaching task."
- "We pre-registered X experiments to test Y."
- "The quick test is intentionally limited; the full protocol is ~80,000 episodes."
- "MIP is a 2-step regression with noise, designed to test Simchowitz's hypothesis."
- "Phase 2 will add WAMs, real-physics simulators, and real-robot vision."

**Do not say (without full results):**
- "MPC dominates robot control."
- "MIP matches full diffusion."
- "Our Pareto frontier shows X is best."
- "We tested on RoboMimic/CALVIN/LIBERO/DMControl" (not yet implemented).
- "We deployed on a real robot" (not yet done).

---

## 9. Pre-publication action checklist

### Immediate (this week)
- [x] Draft blog post and supporting plans.
- [x] Fix `run_experiments.py` OpenVLA import bug.
- [x] Add top-level `LICENSE`.
- [x] Refresh `experiments/environment.lock`.
- [x] Verify quick test still runs.
- [ ] Train and save a small model checkpoint to Hub.
- [ ] Generate and upload a 1300×650 thumbnail.
- [ ] Upload body images to `huggingface/documentation-images`.

### Short term (next 2–4 weeks)
- [ ] Run full EXP-001 ablation and EXP-002 comparison.
- [ ] Create HF Hub model and dataset repos.
- [ ] Create a Gradio Space (plot gallery or simple arena).
- [ ] Update blog draft with full results.

### Long term
- [ ] Publish on arXiv and claim HF paper page.
- [ ] Add real-robot results from Phase 2–4.
- [ ] Update HF blog with follow-up posts.

---

## 10. Publishing workflow

1. Log in to https://huggingface.co/new-blog.
2. Paste the final markdown from `docs/blogging/hf_blog_draft.md`.
3. Add authors, thumbnail, and tags (e.g. `robotics`, `vla`, `diffusion-policy`, `mpc`, `world-models`, `lerobot`, `reproducibility`).
4. Preview, then publish under `Ryukijano` namespace.
5. Add the blog URL to the repo `README.md`.
6. Tweet / toot / post a short summary.

---

## 11. Risk register

| Risk | Mitigation |
|------|------------|
| Blog overstates results | Use claim-safe language; run claim audit before publish. |
| VLA baselines still stub-like | Blog says "VLA baselines are early wrappers, contributions welcome." |
| No HF Hub artifacts yet | Create at least one model + dataset before publish; otherwise emphasize GitHub. |
| Thumbnail / figure quality | Use matplotlib outputs or simple Canva/Figma diagram; keep consistent color scheme. |
| Community blog hidden if low quality | Ensure >1500 words, original analysis, code snippets, and genuine insight. |

---

## 12. Useful links

- Draft blog: `docs/blogging/hf_blog_draft.md`
- Narrative plan: `docs/blogging/blog_narrative_plan.md`
- HF landscape research: `docs/blogging/hf_blog_research.md`
- Artifact plan: `docs/blogging/hf_artifact_plan.md`
- Claim audit: `docs/blogging/claim_audit.md`
- Study repo: https://github.com/Ryukijano/mpc-vla-diffusion-study
- HF new blog: https://huggingface.co/new-blog
- HF blog repo guide: https://github.com/huggingface/blog
- HF paper pages: https://huggingface.co/docs/hub/paper-pages
