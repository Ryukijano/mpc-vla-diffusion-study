# Bugbot Review — MPC vs VLA vs Diffusion Study

**Review date:** 2026-08-14  
**Commit reviewed:** `e90229f9af5d82672d770b0f508868682c7db85a`  
**Repository:** `/home/aimsgroupuol/AIMSgeneral/Gyanateet/mpc_vla_diffusion_study/`  
**Scope:** Last commit (`e90229f`) plus working tree. No uncommitted changes were present at review time (`git status --short` was empty).

---

## 1. Executive Summary

The `e90229f` commit makes useful progress — it adds the Hugging Face blog plan, fixes the `OpenVLAWrapper` → `OpenVLAInference` import, adds a top-level `LICENSE`, and refreshes the environment-lock header. However, the **VLA training path in `run_experiments.py` is still an API mismatch**, the **HF blog draft contains a command that does not produce the file path it points to**, and the **environment lock is one commit stale**. Several `claim_audit.md` findings are also now out of date because the same commit resolved them (top-level `LICENSE`, `OpenVLAInference` import).

**Bottom line:** the repo is not yet safe to publish as a “runnable end-to-end for all three families” blog post. The MPC and MIP paths work, but VLA is an import-only stub and the quick-test documentation is a footgun.

---

## 2. Severity | Location | Finding

| Severity | Location | Finding |
|----------|----------|---------|
| **High** | `run_experiments.py` lines 104–108, 535–565 | VLA import was renamed (`OpenVLAInference`), but the **training/evaluation calls are still using the wrong API**. `SmallVLA.__init__` does not accept `state_dim` or `seed` (`vla_baselines/small_vla.py` lines 222–232), and `OpenVLAInference` has no `.train()` method and no `.predict(x)` method (`vla_baselines/openvla_wrapper.py` lines 60–346). The exceptions are swallowed, so VLA is silently skipped. |
| **High** | `docs/blogging/hf_blog_draft.md` lines 61, 64 | The blog tells readers to run `conda run -n mpc_vla python run_experiments.py --quick` and then points them to `results/quick_test/report/master_comparison_table.csv`. Because `run_experiments.py` line 904 defaults to `--output-dir <STUDY_ROOT>/results`, the command writes `results/tables/master_comparison.csv`, **not** `results/quick_test/...`. This is a reproducibility footgun that will break the next section’s file references. |
| **Medium** | `experiments/environment.lock` lines 32–37 | The lock was “refreshed” to commit `0867a71d9f...`, but `git rev-parse HEAD` is `e90229f...`. The lock is one commit behind and still contains no runtime `env_info.json` proof. Note also the lock says Python 3.11.15 while `configs/system_config.yaml` line 29 says Python 3.12. |
| **Medium** | `docs/blogging/hf_blog_draft.md` lines 76, 83, 110, 123, plus thumbnail line 3 | Image links use repo-relative paths (`results/quick_test/...`) and the thumbnail uses `/blog/assets/...`. These will not render on the Hugging Face blog unless the assets are explicitly uploaded and absolute HF URLs are used. |
| **Medium** | `docs/blogging/hf_blog_draft.md` line 219 | Claims “every result is traceable to a config hash, a git commit, and the CSV/JSON files in `results/`.” No `env_info.json`, config hash, or commit hash is written by `run_experiments.py` / `run_ablation.py` in the current code. |
| **Medium** | `docs/blogging/claim_audit.md` lines 339–342, 369–370, 416–417 | The audit is now partially stale. It still lists the `OpenVLAWrapper` import as broken and the top-level `LICENSE` as missing, but `e90229f` fixed both. This will mislead anyone using the audit as a pre-publish gate. |
| **Low** | `docs/blogging/hf_blog_draft.md` line 61 / `run_experiments.py` line 895 | `--quick` hard-codes `controller_families = ["mpc", "diffusion"]` and `bench_names = ["reaching"]`, so the smoke test never exercises VLA even when the API is fixed. The blog does not mention this. |
| **Low** | `results/quick_test/` | No `env_info.json`, `nvidia-smi` log, or per-condition commit hash is committed, undermining the DGX Spark / reproducibility claims in `claim_audit.md` Category 3.2 and 5.7. |
| **Low** | `data/` | The directory is empty. `claim_audit.md` 5.4 notes the missing `data/eval_seeds_exp*.json` and data manifests. |

---

## 3. Detailed Findings

### 3.1 `run_experiments.py` VLA import change

**What changed in `e90229f`:**

```diff
-from vla_baselines import OpenVLAWrapper, SmallVLA
+from vla_baselines import OpenVLAInference, SmallVLA
+OpenVLAWrapper = OpenVLAInference  # expose under expected alias
```

This part is correct: `vla_baselines/__init__.py` only exports `OpenVLAInference`, so the old import would fail. **However, the downstream code was not updated to match the real VLA interfaces:**

- `SmallVLA.__init__` (`vla_baselines/small_vla.py` lines 222–232) expects `(action_dim, horizon, hidden_dim, num_layers, img_size, text_backend, text_model_name, device)`. `run_experiments.py` line 538 calls `SmallVLA(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim, seed=seed)`.
- `SmallVLA.train()` (`vla_baselines/small_vla.py` lines 379–484) expects a list of dicts with keys `image`, `instruction`, `action`. `run_experiments.py` passes `demos`, a list of `(state, action_sequence)` tuples collected from MPC.
- `SmallVLA.predict_action()` (`vla_baselines/small_vla.py` line 359) expects `(image, instruction)`. `run_experiments.py` line 543 calls `_v.predict(x)` with a state vector.
- `OpenVLAInference` (`vla_baselines/openvla_wrapper.py`) has **no** `train()` method and its inference entry point is `predict_action(image, instruction)`. `run_experiments.py` lines 555–560 call `.train(demos)` and `.predict(x)`.

Because `train_learning_controllers` wraps everything in `try/except`, the failure is silent: a `[SKIP] SmallVLA: ...` or `[SKIP] OpenVLA: ...` message. This means the runner can never actually train VLA baselines in its current form.

### 3.2 `hf_blog_draft.md` — inaccurate command / path mismatch

The blog section “4. Quick smoke-test results” gives this command:

```bash
conda run -n mpc_vla python run_experiments.py --quick
```

Then immediately says:

> The aggregated master comparison table is at `results/quick_test/report/master_comparison_table.csv`

This is wrong. The runner’s default output directory is `results/` (`run_experiments.py` line 904):

```python
output_dir = args.output_dir or os.path.join(STUDY_ROOT, "results")
```

Running the above command creates:

```
results/metrics/full_results.json
results/tables/master_comparison.csv
results/tables/aggregated_comparison.csv
```

It does **not** create `results/quick_test/...`. The committed `results/quick_test/` directory was produced by `scripts/run_quick_test.sh`, which explicitly passes `--output-dir results/quick_test` (`scripts/run_quick_test.sh` line 113).

**Verification:** I ran `python3 run_experiments.py --quick` in a clean tree. Output dir was `/home/.../mpc_vla_diffusion_study/results`, and it wrote `results/tables/master_comparison.csv` and `results/metrics/full_results.json`. I then removed those files to keep the tree clean.

### 3.3 `experiments/environment.lock` refresh

The commit message says “Refresh `experiments/environment.lock` to current HEAD and EXP-001..008.” The diff shows the header was updated, but the recorded commit is `0867a71d9f1dd413e0fe04a97bca584ef26306c3` (the parent), not `e90229f...` (the current HEAD):

```text
# Commit:       0867a71d9f1dd413e0fe04a97bca584ef26306c3
```

`git rev-parse HEAD` returns `e90229f9af5d82672d770b0f508868682c7db85a`. So the lock was not refreshed to the current HEAD. Additionally, the lock’s claim that “The actual commit at run time is logged in each experiment's outputs/<condition>/seed_<s>/env_info.json” is unfulfilled — no `env_info.json` files exist in `results/`.

### 3.4 Open issues from `docs/blogging/claim_audit.md`

The audit already flags most of the serious concerns. The most important unresolved items are:

1. **VLA end-to-end not runnable** (Claim 5.6 / Category 1.6) — import is fixed, but training/evaluation API is still broken.
2. **No DGX Spark runtime provenance** (Claim 3.2) — `results/quick_test/` lacks `env_info.json` or `nvidia-smi` logs.
3. **No Pareto / EXP-004 result** (Claims 1.7, 2.6) — only a toy 5-point plot exists.
4. **No MIP-matches-DDPM evidence** (Claim 2.2) — quick ablation shows MIP 0.2 vs DDPM 0.0, not a match.
5. **No evaluation seeds / data manifests** (Claim 5.4) — `data/` is empty.
6. **HF Hub artifacts not created** (Claim 5.8) — only planning documents.
7. **Result traceability not implemented** (Claim 5.7) — no config hash or per-run commit in outputs.

Two audit items are **now resolved by `e90229f`** and the audit should be refreshed before publish:

- `OpenVLAWrapper` import issue (`claim_audit.md` lines 339–342, 417) — fixed by the import change.
- Missing top-level `LICENSE` (`claim_audit.md` lines 369–370, 418) — `LICENSE` now exists at repo root.

### 3.5 Known quick-test issues

**SmallVLA `__init__` gets unexpected `state_dim` and OpenVLA lacks `.train()`:**

As detailed in §3.1, both are true. The right fix is either:

- **Short-term:** Document VLA as a stub in `run_experiments.py` and `hf_blog_draft.md`, and do not attempt training in state-only mode. Catch the known API mismatch explicitly and print a clearer message.
- **Long-term:** Add a small adapter that renders the 2-D state as a trivial image (e.g. a 96×96 canvas with the point-mass and goal) and a default instruction string, then calls `SmallVLA.predict_action(image, instruction)`. For OpenVLA, either skip training entirely or use `VLATrainer` / a custom training wrapper.

**`run_experiments.py --quick` uses `results/` instead of `results/quick_test/`:**

Yes. This is a genuine footgun. The default `--output-dir` is `None` and resolves to `results/` for all modes, including `--quick`. Only `scripts/run_quick_test.sh` overrides it to `results/quick_test`. A user copy-pasting the blog command will not find the files the blog points to and may overwrite other `results/` outputs.

---

## 4. Prioritized Fix List

| Priority | Fix | Estimated Effort | File(s) |
|----------|-----|------------------|---------|
| **P0** | Decide whether VLA is in-scope for the quick test. If it stays a stub, make the skip **explicit and documented** in `hf_blog_draft.md` and `run_experiments.py`. If it must run, write a state→image/instruction adapter and use `SmallVLA.predict_action()` / `OpenVLAInference.predict_action()`. | 1–2 days | `run_experiments.py`, `vla_baselines/small_vla.py` (or adapter), `docs/blogging/hf_blog_draft.md` |
| **P0** | Fix the quick-test command/path mismatch. Either (a) make `run_experiments.py --quick` default to `results/quick_test` when `--output-dir` is unset, or (b) update `hf_blog_draft.md` and `run_experiments.py --help` to include `--output-dir results/quick_test`. | 10 min–1 hr | `run_experiments.py` and/or `docs/blogging/hf_blog_draft.md` |
| **P1** | Regenerate `experiments/environment.lock` from the actual `mpc_vla` environment and set the commit hash to the current HEAD. Add an `env_info.json` writer to `run_experiments.py` and `run_ablation.py`, then rerun the quick test on DGX Spark and commit the outputs. | 2–4 hrs (if DGX env is available; longer if not) | `experiments/environment.lock`, `run_experiments.py`, `run_ablation.py` |
| **P1** | Fix HF blog image paths. Upload the four PNGs in `results/quick_test/report/figures/` and `results/quick_test/ablation/figures/` to the HF blog asset store and replace the repo-relative markdown links. | 1–2 hrs | `docs/blogging/hf_blog_draft.md` |
| **P1** | Refresh `docs/blogging/claim_audit.md` to reflect the resolved items (import fixed, `LICENSE` added) and the remaining open issues. | 1 hr | `docs/blogging/claim_audit.md` |
| **P2** | Generate and commit `data/eval_seeds_exp*.json` and data manifests. | 2–4 hrs | `data/`, `scripts/run_quick_test.sh` |
| **P2** | Add a `--quick` caveat to the blog: `--quick` overrides controllers to `["mpc", "diffusion"]` and benchmarks to `["reaching"]`, so it does not exercise VLA. | 30 min | `docs/blogging/hf_blog_draft.md` |
| **P3** | Run `EXP-001`–`EXP-004` before making family-comparison, Pareto, or Simchowitz-replication claims in the blog. | Days–weeks | `experiments/EXP-*/` |

---

## 5. Recommended Blog-Safe Edits (can land now)

If a short-term publish is desired, the following wording/paths should be changed in `docs/blogging/hf_blog_draft.md`:

1. **Line 61:** Add `--output-dir results/quick_test` to the command, or add a parenthetical note that the paths below assume `scripts/run_quick_test.sh` was used.
2. **Lines 76, 83, 110, 123:** Replace `![...](results/quick_test/...)` with HF asset URLs (or at least note “paths are repo-relative; upload the corresponding PNGs from `results/quick_test/...`).
3. **Line 219:** Change “every result is traceable to a config hash, a git commit, and the CSV/JSON files” to “we *plan* to make every result traceable …; the quick test currently provides the raw CSV/JSON/PNG outputs only.”
4. Add a one-sentence caveat that the quick test is MPC + MIP only; VLA baselines are imported but not yet trained/evaluated end-to-end.

---

## 6. Verification Evidence

All findings above are based on the current commit and direct checks:

- `git status --short` and `git log --oneline -5` confirmed no uncommitted changes and `e90229f` as HEAD.
- `git rev-parse HEAD` returned `e90229f9af5d82672d770b0f508868682c7db85a`.
- `git show e90229f -- run_experiments.py experiments/environment.lock` confirmed the VLA import change and lock header diff.
- `python3 run_experiments.py --quick --output-dir /tmp/bugbot_quick_test` passed (18.3s, MPC success 1.0, MIP 0.4) and wrote the expected CSV/JSON.
- `python3 run_experiments.py --quick` (no `--output-dir`) confirmed the default output is `results/` and produced `results/tables/master_comparison.csv`; those files were removed after verification.
- `python3 run_ablation.py --benchmark reaching --seeds 0 --episodes 5 --epochs 10 --num-demos 10 --output-dir /tmp/bugbot_ablation` passed and produced the expected ablation CSVs and figures.
- Code review of `vla_baselines/small_vla.py`, `vla_baselines/openvla_wrapper.py`, and `run_experiments.py` lines 535–565 confirmed the API mismatch.
- `ls -la data/` and `find results/quick_test -name env_info.json` confirmed missing data seeds and env provenance.
- `ls -la LICENSE` confirmed a top-level `LICENSE` now exists.

---

## 7. Summary for Parent Agent

**Key files and line numbers:**

- `run_experiments.py` VLA import fix: **lines 104–108** (correct) and VLA usage bug: **lines 535–565**.
- `docs/blogging/hf_blog_draft.md` command/path mismatch: **lines 61 and 64**.
- `run_experiments.py` default output dir: **line 904** and `--help` text **lines 1058–1060**.
- `experiments/environment.lock` stale commit: **line 32**.
- `docs/blogging/claim_audit.md` stale items: **lines 339–342, 369–370, 416–418**.

**Most important actions:**

1. Fix or clearly stub the VLA training path in `run_experiments.py`.
2. Fix the `hf_blog_draft.md` quick-test command to include `--output-dir results/quick_test`, or change the runner default.
3. Refresh `experiments/environment.lock` to the current HEAD and add `env_info.json` generation.
4. Refresh `docs/blogging/claim_audit.md` and resolve the stale findings.
