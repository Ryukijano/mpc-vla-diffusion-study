# Publish Decision Analysis: MPC vs VLA vs Diffusion HF Community Blog

**Date:** 2026-08-13  
**Study:** MPC vs VLA vs Diffusion comparison study  
**Repository:** https://github.com/Ryukijano/mpc-vla-diffusion-study  
**Purpose:** Strategic analysis of whether to publish the HF community blog now or later

---

## 1. Should we publish now?

**Answer: NO — Wait 2-4 weeks and strengthen the blog first.**

### Five reasons to wait

1. **HF Community Blog quality gate is strict.** HF explicitly states *"Low-quality, non-technical, non-original, and LLM-generated blog posts will be hidden from the blog main page"* ([huggingface.co/blog-explorers](https://huggingface.co/blog-explorers)). The current draft risks being perceived as LLM-generated, partly because the prose is polished but light on results.

2. **No HF Hub artifacts yet.** HF blog guidelines recommend two categories: (a) *"Explore an AI science or engineering concept"* or (b) *"Announce the release of an open source artifact"* ([docs/hub/blog-articles](https://huggingface.co/docs/hub/main/en/blog-articles)). The current blog has no HF Hub models, datasets, or Spaces — only GitHub. This reduces discoverability and risks the blog being hidden.

3. **Empirical results are too weak.** The quick test (1 seed, 5 episodes, tiny networks, toy 2-D reaching) is a sanity check, not a result. The claim audit shows multiple claims that would need to be removed or heavily caveated. Without EXP-001/002/004 results, the blog lacks the empirical core to be a successful "concept exploration" post.

4. **VLA baselines are not functional end-to-end.** The quick test only runs `mpc` and `diffusion` families. Publishing a blog about *"MPC vs VLA vs Diffusion"* without working VLA results is misleading.

5. **Successful work-in-progress examples have stronger artifacts.** Examples like [Proof of Time](https://huggingface.co/blog/shanchen/pot), [PrediBench](https://huggingface.co/blog/charles-azam/predibench), and [Open DeepResearch](https://huggingface.co/blog/open-deep-research) all published with (a) HF Hub datasets/models, (b) clear prototype results, or (c) explicit "work in progress" framing with concrete next steps.

---

## 2. Main risks of publishing now

### LLM-generated perception
- The prose is polished but result-light, and a `devin-ai` co-author may reinforce the impression that the post was LLM-generated.
- HF blog-explorers explicitly bans *"LLM-generated blog posts"*.

### Weak results damage credibility
- Quick-test numbers (MPC 100%, MIP 40%, DDPM 0%) are from n=5 episodes on a toy task.
- Building a family-level Pareto frontier around this is premature.

### No Hub artifacts reduces discoverability
- Blog only links to GitHub, not to any model/dataset/Space.
- HF articles mentioning a repo's own models/datasets get automatic sidebar links; without artifacts, the blog is an island.

### VLA stubs mislead readers
- The title promises a VLA comparison, but the current quick test does not produce VLA results.

---

## 3. What would make the blog much stronger? (Prioritized)

**Priority 1 — must have before any publish**
1. Run EXP-001 mechanism ablation (5 conditions × 2+ benchmarks × 5 seeds).
2. Run EXP-002 three-family comparison (MPC, VLA, diffusion/flow on multiple tasks × 5 seeds).
3. Create at least one HF Hub model checkpoint (SmallVLA or tiny DiffusionPolicy).
4. Create at least one HF Hub dataset repo (MPC expert demos).
5. Fix VLA baseline API so quick test actually runs VLA.

**Priority 2 — strongly recommended**
6. Run EXP-004 latency Pareto sweep.
7. Create a HF Space (plot gallery or simple leaderboard).
8. Add `env_info.json` and data manifests for reproducibility.

**Priority 3 — nice to have**
9. 1300×650 thumbnail.
10. Upload body images to `huggingface/documentation-images`.

---

## 4. Minimum viable version safe to publish now

If there is external pressure to publish now (not recommended):

- **Retitle** to *"MPC vs Diffusion: An Open-Source Pre-Registered Study Design"* (remove VLA).
- **Remove VLA** from the blog and frame it as a two-family pre-registered study.
- **Add a "Work in Progress" banner** at the top.
- **Add explicit caveats** on every result table: *n = 1 seed, 5 episodes, toy benchmark — not a conclusion*.
- **Remove the `devin-ai` co-author** to reduce LLM-perception risk.
- **Fix the VLA import bug** or remove all VLA references from the repo for consistency.

Even this reduced version is risky and likely to get low engagement.

---

## 5. Path comparison

### Path A — Publish a WIP blog now (1-2 days)
- **Pros:** fast visibility, may attract early contributors.
- **Cons:** high risk of being hidden, no Hub cross-links, misleading VLA claims, low expected views (50–200), possible negative feedback.
- **Risk level:** HIGH

### Path B — Wait 2-4 weeks and run experiments (recommended)
- **Pros:** real empirical core, HF Hub artifacts, auto-sidebar cross-links, lower risk of being hidden, one definitive post, stronger credibility.
- **Cons:** delayed publication, ~16 hours of compute on DGX Spark, more upfront engineering.
- **Expected views:** 500–2,000 (5–10× higher)
- **Risk level:** LOW

---

## 6. Recommendation

**Wait and follow Path B.** Publish after EXP-001/002/004, at least one HF Hub model, one HF Hub dataset, and a working VLA baseline. The extra 2–4 weeks produce a blog that can survive HF's quality gate and drive real engagement.

---

## 7. Concrete next steps

1. Fix the VLA baseline API (`SmallVLA.__init__` and `OpenVLAInference` train/predict interface).
2. Add `env_info.json` generation to `run_experiments.py`.
3. Run and commit EXP-001, EXP-002, and EXP-004 results.
4. Train SmallVLA and tiny DiffusionPolicy and upload them to HF Hub.
5. Upload MPC expert demos as a HF Hub dataset.
6. Create a Gradio Space plot gallery.
7. Rewrite the blog draft with real results, remove `devin-ai` co-author, and add Hub artifact links.
8. Create 1300×650 thumbnail and upload body images to `huggingface/documentation-images`.
9. Publish via `huggingface.co/new-blog`.

---

## 8. Sources

- [HF Blog Articles Docs](https://huggingface.co/docs/hub/main/en/blog-articles)
- [HF Blog Explorers Guidelines](https://huggingface.co/blog-explorers)
- [Proof of Time (WIP blog example)](https://huggingface.co/blog/shanchen/pot)
- [PrediBench (prototype blog example)](https://huggingface.co/blog/charles-azam/predibench)
- [Open DeepResearch (WIP blog example)](https://huggingface.co/blog/open-deep-research)
- `docs/blogging/claim_audit.md`
- `docs/blogging/hf_blog_draft.md`
