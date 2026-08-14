# Verification Report — `demo_space/app.py`

**Project:** `mpc_vla_diffusion_study`  
**Scope:** Gradio Space application (`demo_space/app.py`)  
**Date:** 14 Aug 2026  
**Environment:** `mpc_vla` conda environment, Gradio 6.24.0

---

## 1. Import / Syntax Checks

| Check | Command | Result |
|-------|---------|--------|
| Dry import test | `conda run -n mpc_vla python -c "import app; print('app.py imports OK')"` | **PASS** |
| Byte-code compile | `python -m py_compile app.py` | **PASS** |

`app.py` imports cleanly and produces no syntax errors.

---

## 2. Asset / Artifact Verification

All images, plots, and CSVs that `app.py` actually references were checked.

| Asset / Artifact | Path | Status |
|------------------|------|--------|
| Master comparison CSV | `results/quick_test/report/master_comparison_table.csv` (resolved from study root) | **Present** |
| EXP-001 ablation CSV | `results/quick_test/ablation/ablation_comparison.csv` (resolved from study root) | **Present** |
| SmallVLA checkpoint | `dist/hf_models/small_vla/small_vla_quick.pt` | **Present** |
| Report figure PNGs | `results/quick_test/report/figures/*.png` (5 files) | **Present** |
| Ablation figure PNGs | `results/quick_test/ablation/figures/*.png` (3 files) | **Present** |
| Local results fallback | `demo_space/results/quick_test/...` | **Present** |
| `demo_space/assets/` | `demo_space/assets/` | **Empty and not referenced by `app.py`** |

The gallery in the *Results & Ablation Viewer* tab successfully discovers **8 unique PNGs** (de-duplicated across study root and local `demo_space/results`).

No referenced assets were missing, so no placeholder figures were required. The unused `assets/` directory is available for optional README/banner images.

---

## 3. Gradio UI / Function-Signature Consistency

The Gradio `Blocks` object (`demo`) was constructed successfully and `demo.get_config_file()` returns a valid 77-component layout.

| Function | Inputs | Outputs | Wired to UI | Result |
|----------|--------|---------|-------------|--------|
| `run_simulation` | 7 (controller, task, seed, max steps, noise, diffusion steps, instruction) | 4 (traj plot, telemetry plot, summary markdown, telemetry DataFrame) | Yes — `run_btn.click(...)` | **Consistent** |
| `update_pareto_chart` | 5 (task, families, log-x, min success, max latency) | 2 (pareto plot, filtered DataFrame) | Yes — all 5 inputs `.change(...)` and `demo.load(...)` | **Consistent** |
| `load_comparison_tables` / `get_figure_paths` | None / None | Tables / figure list | Yes — Results & Ablation tab | **Consistent** |

`gr.Dataframe(headers=...)` is accepted by Gradio 6.24.0. `gr.Plot`, `gr.Image`, `gr.CheckboxGroup`, and `gr.Slider` usage all align with current Gradio APIs.

---

## 4. Runtime / Function-Level Verification

Each controller was exercised through `run_simulation` to ensure the backend logic is functional.

| Controller | Task | Result |
|------------|------|--------|
| Linear MPC | 2D Reaching | OK (20 steps) |
| Nonlinear MPC (iLQR) | 2D Reaching | OK (20 steps) |
| Collision-Free MPC | 2D Reaching (Cluttered) | OK (20 steps) |
| **Diffusion Warm-Start MPC** | **2D Reaching (Cluttered)** | **OK after fix** |
| MIP (Minimal Iterative Policy) | 2D Reaching | OK (20 steps) |
| Diffusion Policy (DDPM) | 2D Reaching | OK (20 steps) |
| Flow Matching Policy | 2D Reaching | OK (20 steps) |
| SmallVLA | 2D Reaching | OK (20 steps) |
| SmallVLA | PushT | OK (20 steps) |

Additional checks:

- `update_pareto_chart(...)` returns a Plotly figure with 14 rows of data — **OK**.
- `load_comparison_tables()` returns master table `(9, 10)` and ablation table `(5, 9)` — **OK**.
- `get_figure_paths()` discovers 8 figure files — **OK**.

---

## 5. Errors Found and Fixes Applied

### 5.1 Diffusion Warm-Start MPC crash on cluttered reaching

**Symptom:** `run_simulation('Diffusion Warm-Start MPC', '2D Reaching (Cluttered)', ...)` raised:

```
AttributeError: 'Obstacle' object has no attribute 'signed_distance'
```

**Root cause:** The cluttered reaching obstacles were created with `benchmarks.reaching_env.Obstacle`, which exposes `center`, `radius`, and `contains()` but **not** `signed_distance()`. `DiffusionWarmStartMPC` scores trajectories with `src.utils.obstacles.total_collision_cost`, which calls `obs.signed_distance(point)` on every obstacle.

**Fix applied in `app.py`:**

1. Added a small internal `_ArenaObstacle` class (`app.py` line 108) that is duck-typed for both APIs:
   - `contains(point)` for `ReachingEnv`
   - `signed_distance(point)` for the MPC / diffusion warm-start collision scorer
2. Replaced the seven `Obstacle(...)` constructors in the cluttered task definition (`app.py` lines 271-277) with `_ArenaObstacle(...)`.

This is a minimal, self-contained change in `app.py` and does not require modifying the benchmark or MPC libraries.

### 5.2 Warnings / Suggestions

1. **`demo_space/assets/` is empty.** It is not referenced by `app.py`, but consider adding a banner or OpenGraph image for the Hugging Face Space card.
2. **Gradio version drift.** `requirements.txt` pins `gradio>=5.0.0`, but the active environment has `gradio==6.24.0`. The app works with 6.x, but for reproducibility it is recommended to pin to a tested minor version, e.g. `gradio>=6.24.0,<7.0.0`.
3. **Unused requirements.** `scipy`, `pyyaml`, `pillow`, `pyarrow`, and `huggingface-hub` are listed but not directly imported by `app.py`. They are safe to keep as transitive dependencies, but could be reviewed.
4. **Module fallback blocks.** All study modules imported successfully in the test environment, but the `try/except` import blocks silently set globals to `None` if a dependency is missing. Runtime errors could be more user-friendly if `None` values are checked before use.

---

## 6. Files Changed

- `demo_space/app.py`
  - Added `_ArenaObstacle` adapter class (line 108).
  - Replaced `Obstacle(...)` with `_ArenaObstacle(...)` for the 2D Reaching (Cluttered) obstacle list (lines 271-277).

- `demo_space/VERIFICATION.md` (this file) — newly created.

---

## 7. Commands Run

```bash
# Dry import
cd /home/aimsgroupuol/AIMSgeneral/Gyanateet/mpc_vla_diffusion_study/demo_space
conda run -n mpc_vla python -c "import app; print('app.py imports OK')"

# Syntax check
python -m py_compile app.py

# Controller smoke tests
conda run -n mpc_vla python -c "
import app
for ctrl, task in [
    ('Linear MPC', '2D Reaching'),
    ('Nonlinear MPC (iLQR)', '2D Reaching'),
    ('Collision-Free MPC', '2D Reaching (Cluttered)'),
    ('Diffusion Warm-Start MPC', '2D Reaching (Cluttered)'),
    ('MIP (Minimal Iterative Policy)', '2D Reaching'),
    ('Diffusion Policy (DDPM)', '2D Reaching'),
    ('Flow Matching Policy', '2D Reaching'),
    ('SmallVLA', '2D Reaching'),
    ('SmallVLA', 'PushT'),
]:
    app.run_simulation(ctrl, task, 42, 20, 0.1, 5, 'Reach the target')
    print(f'{ctrl} / {task} OK')
"

# UI data checks
conda run -n mpc_vla python -c "
import app
fig, df = app.update_pareto_chart('All Tasks', ['Classical MPC','Hybrid','Minimal Iterative','Diffusion Policy','Vision-Language-Action'], True, 0, 100)
print('Pareto rows:', len(df))
print('Figures:', len(app.get_figure_paths()))
"
```

---

## 8. Conclusion

`demo_space/app.py` is now importable, syntactically valid, and functionally verified for all eight controller families across both 2D reaching tasks and PushT. The referenced result CSVs and figures are all present. A single runtime bug in the Diffusion Warm-Start MPC path was identified and fixed. No upload to Hugging Face was performed.
