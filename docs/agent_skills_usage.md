# Agent-Skills Usage Guide for MPC vs VLA vs Diffusion Study

This document maps the tools from `agent-skills/` (Ryukijano/agent-skills) to specific
tasks in the comparison study.

## Skills Location

```
agent-skills/
├── .devin/skills/      # 147 Devin skills (SKILL.md each)
├── .devin/workflows/   # 133 Devin workflows (.md each)
├── .cursor/skills/     # 162 Cursor skills
├── .cursor/commands/   # 143 Cursor commands
├── science_skills/     # 34 science/bioinformatics skills
├── mcp_servers/        # 7 MCP servers (72 tools)
└── workflows/devin/    # Devin workflow definitions
```

## Skills Used in This Study

### 1. deep-research (`.devin/skills/deep-research/SKILL.md`)
**When:** Phase 1 literature review, and any time we need to research an unfamiliar method.

**How:** Follows the search loop: DECOMPOSE → SEARCH → EVALUATE → IDENTIFY GAPS → REFINE → REPEAT → SYNTHESIZE.

**Usage for this study:**
```
Sub-queries for "MPC vs VLA vs Diffusion":
1. "classical MPC limitations robotic manipulation generalization"
2. "diffusion policy multi-modal action distribution evidence"
3. "VLA inference latency real-time robot control"
4. "iterative compute noise injection policy learning"
5. "minimal iterative policy regression vs flow matching"
```

### 2. autoresearch (`.devin/skills/autoresearch/SKILL.md`)
**When:** Automated literature search and synthesis across large paper sets.

**Usage:** Feed seed papers from `literature/seed_papers/`, auto-discover related work,
generate structured notes in `literature/notes/`.

### 3. ablation-study (`.devin/skills/ablation-study/SKILL.md`)
**When:** Phase 4 — designing the GCP component ablations.

**Process:**
1. Identify components: noise injection, iterative steps, distribution fitting, architecture
2. Baseline = full flow GCP
3. Variants: no-noise, single-step, MIP (2-step), pure regression
4. Same seed, same data, same budget
5. Collect into comparison table
6. Bar chart + table for report

**Config location:** `configs/ablations/`

### 4. literature_search_arxiv (`.science_skills/literature_search_arxiv/`)
**When:** Finding papers on arXiv by keyword, category, or citation.

**Usage:**
```python
# Search for diffusion MPC papers
# Search for VLA benchmark papers
# Search for iterative policy / regression policy papers
```

### 5. literature_search_openalex (`.science_skills/literature_search_openalex/`)
**When:** Citation graph analysis — finding papers that cite or are cited by seed papers.

**Usage:** Start from "Much Ado About Noising" (2512.01809) and "Diffusion Policy" (2303.04137),
trace citation network to map the debate.

### 6. experiment-tracking (`.devin/skills/experiment-tracking/`)
**When:** Phase 5 — logging all comparison runs.

**Usage:** Structured logging of:
- Method name (MPC-linear, MPC-nonlinear, OpenVLA, DiffusionPolicy, MIP, ...)
- Benchmark name
- Seed
- Success rate, return, latency, mode coverage
- Config hash for reproducibility

### 7. experiment-reproducibility (`.devin/skills/experiment-reproducibility/`)
**When:** Ensuring fair comparison across methods.

**Checklist:**
- Fixed random seeds (5 seeds: 0, 1, 2, 42, 123)
- Same observation preprocessing
- Same evaluation episodes (fixed episode seeds)
- Same action space discretization
- Same success criterion
- Config files committed for every run

### 8. data-visualization (`.devin/skills/data-visualization/`)
**When:** Phase 6 — generating comparison plots.

**Plots to generate:**
- Success rate bar chart (per task, per method)
- Latency vs. success scatter (all methods overlaid)
- Ablation bar chart (GCP component contributions)
- Mode coverage heatmap (method × task)
- OOD robustness curve (perturbation level vs. manifold adherence)

### 9. academic-plotting (`.devin/skills/academic-plotting/`)
**When:** Publication-quality figures for the position paper.

### 10. collaborative-research (`.devin/skills/collaborative-research/`)
**When:** If multiple contributors join the study.

## MCP Servers

### research-workflow (`mcp_servers/research_workflow/server.py`)
**Tools:** Live research workflow tools — can be installed and run as an MCP server.

**Install:**
```bash
cd agent-skills/mcp_servers/research_workflow
# Follow server.py setup instructions
```

### Other MCP servers available:
- `cloud_gpu_ssh` — SSH to cloud GPU instances
- `cuda_profiling` — Profile GPU code
- `dgx_monitor` — Monitor DGX Spark resources
- `distributed_training` — Multi-GPU training
- `tpu_jax` — TPU/JAX workflows
- `endosight_pipeline` — Endosight 3D pipeline (not relevant here)

## Workflow Files

Pre-defined workflows in `.devin/workflows/`:
- `ablation-study.md` — Step-by-step ablation procedure
- `deep-research.md` — Step-by-step deep research procedure
- `experiment-tracking.md` — Experiment logging setup
- `data-visualization.md` — Visualization workflow

## How to Invoke

### In Devin CLI:
```
/deep-research "Compare classical MPC vs diffusion policy for robotic manipulation"
/ablation-study "Design ablation for GCP components: noise, iterative steps, distribution fitting"
/autoresearch "Find papers on iterative policy regression vs flow matching robotics"
```

### In Cursor:
Use the corresponding commands in `.cursor/commands/`.

### As MCP tools:
Install the MCP servers and call tools programmatically from Python scripts.
