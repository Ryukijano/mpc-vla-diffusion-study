# MPC vs VLA vs Diffusion MPC: A Comparative Study

## Motivation

This study is motivated by Max Simchowitz's talk **"Do we need diffusion in robotics?"**
(Simons Institute for the Theory of Computing, Aug 7, 2026) and the accompanying paper
**"Much Ado About Noising: Dispelling the Myths of Generative Robotic Control"**
(arXiv: [2512.01809](https://arxiv.org/abs/2512.01809)).

The central question: **Do generative control policies (GCPs) — diffusion/flow-based action
generators — actually provide benefits over classical MPC and VLA-style policies, and if so,
what is the mechanism?**

Simchowitz et al. find that GCPs' success is **not** due to multi-modal distribution fitting
(the prevailing wisdom), but rather to **supervised iterative compute + stochasticity injection**,
which improves manifold adherence under out-of-distribution observations. A minimal iterative
policy (MIP) — a lightweight two-step regression — matches flow GCPs.

This study plans a systematic comparison across three families of robot control:

1. **Classical MPC** — optimization-based, explicit dynamics, constraint-aware
2. **VLA (Vision-Language-Action)** — VLM backbone + action head, language-conditioned
3. **Diffusion MPC / Diffusion Policy (GCPs)** — diffusion/flow-based action generation

## Folder Structure

```
mpc_vla_diffusion_study/
├── agent-skills/              # Cloned Ryukijano/agent-skills repo (tooling)
├── docs/                      # Study documents, plans, methodology
├── literature/                # Literature review
│   ├── seed_papers/           # Core papers to build from
│   ├── notes/                 # Per-paper reading notes
│   └── bibtex/                # Bibliography
├── configs/                   # Experiment configurations
│   ├── mpc/                   # Classical MPC configs
│   ├── vla/                   # VLA model configs
│   ├── diffusion/             # Diffusion policy configs
│   └── ablations/             # Ablation study configs
├── src/                       # Source code
│   ├── mpc_baselines/         # Classical MPC implementations
│   ├── vla_baselines/         # VLA model wrappers
│   ├── diffusion_policies/    # Diffusion/flow policy implementations
│   ├── eval/                  # Evaluation harnesses and metrics
│   └── utils/                 # Shared utilities
├── scripts/                   # Run scripts
├── notebooks/                 # Exploratory analysis
├── data/                      # Dataset links/manifests
├── results/                   # Experiment outputs
│   ├── tables/                # Comparison tables
│   ├── logs/                  # Run logs
│   └── metrics/               # Aggregated metrics
└── figures/                   # Plots and visualizations
```

## Agent-Skills Tooling

The `agent-skills/` repo (from [Ryukijano/agent-skills](https://github.com/Ryukijano/agent-skills))
provides the following tools used in this study:

| Tool | Type | Use in this study |
|------|------|-------------------|
| `deep-research` | Devin skill | Systematic web research on MPC/VLA/diffusion literature |
| `autoresearch` | Devin skill | Automated literature search and synthesis |
| `ablation-study` | Devin skill | Design ablation experiments isolating GCP components |
| `collaborative-research` | Devin skill | Multi-author project coordination |
| `experiment-tracking` | Devin skill | Structured logging of comparison runs |
| `experiment-reproducibility` | Devin skill | Seeds, config capture, deterministic runs |
| `literature_search_arxiv` | Science skill | arXiv paper discovery |
| `literature_search_openalex` | Science skill | OpenAlex citation graph analysis |
| `literature_search_europepmc` | Science skill | Europe PMC full-text search |
| `research-workflow` | MCP server | Live research workflow tools |
| `data-visualization` | Devin skill | Comparison plots and charts |
| `academic-plotting` | Devin skill | Publication-quality figures |

See `docs/agent_skills_usage.md` for detailed usage patterns.
