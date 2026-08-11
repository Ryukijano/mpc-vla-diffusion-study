"""Evaluation utilities for benchmarking controllers.

This module provides :class:`Evaluator`, which runs controllers on
:class:`benchmarks.base_env.BaseEnv` instances and collects a standard set of
metrics for the MPC-vs-VLA-vs-Diffusion comparison study.

Supported controller types
--------------------------
* ``"mpc"``       -- ``controller.solve(state, ref) -> action``
* ``"vla"``       -- ``controller.predict_action(image, instruction) -> action``
* ``"diffusion"`` -- ``controller.sample(observation) -> action``

Collected metrics
-----------------
* ``success_rate``           -- fraction of episodes that reached the goal.
* ``mean_return``            -- average episode return.
* ``path_length``            -- average total action-norm over an episode.
* ``collision_rate``         -- fraction of episodes that ended in collision.
* ``mean_inference_latency`` -- average per-step inference time in seconds.
"""

from __future__ import annotations

import csv
import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .base_env import BaseEnv

__all__ = ["Evaluator"]

# Valid controller type strings.
_VALID_CONTROLLER_TYPES = ("mpc", "vla", "diffusion")


class Evaluator:
    """Run controllers on environments and collect standardised metrics.

    Parameters
    ----------
    verbose : bool, optional
        If ``True``, print per-episode progress.
    """

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = bool(verbose)

    # ------------------------------------------------------------------
    # Internal: controller dispatch
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_action(
        controller: Any,
        env: BaseEnv,
        controller_type: str,
    ) -> Tuple[np.ndarray, float]:
        """Query *controller* for an action and measure inference latency.

        Returns
        -------
        action : np.ndarray
            The action produced by the controller.
        latency : float
            Wall-clock inference time in seconds.
        """
        t0 = time.perf_counter()
        if controller_type == "mpc":
            state = env.get_state()
            ref = env.get_target() if hasattr(env, "get_target") else None
            action = controller.solve(state, ref)
        elif controller_type == "vla":
            image = env.get_image()
            instruction = env.get_language_instruction()
            action = controller.predict_action(image, instruction)
        elif controller_type == "diffusion":
            obs = env.get_observation()
            action = controller.sample(obs)
        else:
            raise ValueError(
                f"Unknown controller_type '{controller_type}'. "
                f"Must be one of {_VALID_CONTROLLER_TYPES}."
            )
        latency = time.perf_counter() - t0
        return np.asarray(action, dtype=np.float32), latency

    # ------------------------------------------------------------------
    # Internal: single-episode rollout
    # ------------------------------------------------------------------
    def _run_episode(
        self,
        controller: Any,
        env: BaseEnv,
        controller_type: str,
        seed: Optional[int],
        max_steps: Optional[int] = None,
    ) -> Dict[str, float]:
        """Run a single evaluation episode and return per-episode metrics."""
        max_steps = max_steps if max_steps is not None else env.max_steps
        env.reset(seed=seed)

        total_reward = 0.0
        path_length = 0.0
        latencies: List[float] = []
        collided = False
        succeeded = False

        for _ in range(max_steps):
            action, latency = self._compute_action(
                controller, env, controller_type
            )
            latencies.append(latency)
            _, reward, done, info = env.step(action)
            total_reward += float(reward)
            path_length += float(np.linalg.norm(action))
            if env.is_collision():
                collided = True
            if env.is_success():
                succeeded = True
            if done:
                break

        episode_metrics: Dict[str, float] = {
            "return": total_reward,
            "path_length": path_length,
            "success": float(succeeded),
            "collision": float(collided),
            "mean_latency": float(np.mean(latencies)) if latencies else 0.0,
        }
        if self.verbose:
            print(
                f"  seed={seed} | return={total_reward:.3f} "
                f"success={succeeded} collision={collided} "
                f"steps={env.step_count}"
            )
        return episode_metrics

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate(
        self,
        controller: Any,
        env: BaseEnv,
        n_episodes: int = 10,
        seeds: Optional[Sequence[int]] = None,
        controller_type: str = "mpc",
        max_steps: Optional[int] = None,
    ) -> Dict[str, float]:
        """Evaluate a single controller on *env*.

        Parameters
        ----------
        controller : object
            The controller to evaluate.  Its interface must match
            *controller_type*.
        env : BaseEnv
            The benchmark environment.
        n_episodes : int
            Number of evaluation episodes.
        seeds : sequence of int, optional
            Per-episode seeds.  If ``None``, seeds ``0..n_episodes-1`` are used.
        controller_type : str
            One of ``"mpc"``, ``"vla"``, ``"diffusion"``.
        max_steps : int, optional
            Override the environment's default horizon.

        Returns
        -------
        dict
            Aggregated metrics: ``success_rate``, ``mean_return``,
            ``path_length``, ``collision_rate``, ``mean_inference_latency``.
        """
        if controller_type not in _VALID_CONTROLLER_TYPES:
            raise ValueError(
                f"controller_type must be one of {_VALID_CONTROLLER_TYPES}, "
                f"got '{controller_type}'."
            )
        if seeds is None:
            seeds = list(range(n_episodes))
        elif len(seeds) < n_episodes:
            seeds = list(seeds) + [
                len(seeds) + i for i in range(n_episodes - len(seeds))
            ]

        all_metrics: List[Dict[str, float]] = []
        for ep in range(n_episodes):
            ep_metrics = self._run_episode(
                controller, env, controller_type,
                seed=seeds[ep], max_steps=max_steps,
            )
            all_metrics.append(ep_metrics)

        return self._aggregate(all_metrics)

    def evaluate_all(
        self,
        controllers: Dict[str, Dict[str, Any]],
        env: BaseEnv,
        n_episodes: int = 10,
        seeds: Optional[Sequence[int]] = None,
        max_steps: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate multiple controllers on the same environment.

        Parameters
        ----------
        controllers : dict
            Mapping ``name -> {"controller": obj, "type": str}`` where
            ``type`` is one of ``"mpc"``, ``"vla"``, ``"diffusion"``.
        env : BaseEnv
            The benchmark environment.
        n_episodes : int
            Number of evaluation episodes per controller.
        seeds : sequence of int, optional
            Per-episode seeds (shared across controllers for fair comparison).
        max_steps : int, optional
            Override the environment's default horizon.

        Returns
        -------
        dict
            Mapping ``controller_name -> metrics_dict``.
        """
        results: Dict[str, Dict[str, float]] = {}
        for name, spec in controllers.items():
            controller = spec["controller"]
            ctype = spec.get("type", "mpc")
            if self.verbose:
                print(f"Evaluating '{name}' ({ctype}) ...")
            results[name] = self.evaluate(
                controller, env,
                n_episodes=n_episodes,
                seeds=seeds,
                controller_type=ctype,
                max_steps=max_steps,
            )
        return results

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    @staticmethod
    def _aggregate(
        episodes: List[Dict[str, float]],
    ) -> Dict[str, float]:
        """Aggregate per-episode metrics into a summary dict."""
        n = len(episodes)
        if n == 0:
            return {
                "success_rate": 0.0,
                "mean_return": 0.0,
                "path_length": 0.0,
                "collision_rate": 0.0,
                "mean_inference_latency": 0.0,
            }
        return {
            "success_rate": float(np.mean([e["success"] for e in episodes])),
            "mean_return": float(np.mean([e["return"] for e in episodes])),
            "path_length": float(np.mean([e["path_length"] for e in episodes])),
            "collision_rate": float(np.mean([e["collision"] for e in episodes])),
            "mean_inference_latency": float(
                np.mean([e["mean_latency"] for e in episodes])
            ),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    @staticmethod
    def save_results(
        results: Dict[str, Dict[str, float]],
        path: str,
    ) -> Tuple[str, str]:
        """Save evaluation results to CSV and JSON files.

        Parameters
        ----------
        results : dict
            Output of :meth:`evaluate` or :meth:`evaluate_all`.
        path : str
            Base file path (without extension).  ``.csv`` and ``.json``
            extensions are appended.

        Returns
        -------
        tuple of str
            ``(csv_path, json_path)`` -- the actual file paths written.
        """
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        csv_path = path + ".csv"
        json_path = path + ".json"

        # --- JSON ---
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

        # --- CSV ---
        metric_keys: List[str] = []
        for metrics in results.values():
            for k in metrics:
                if k not in metric_keys:
                    metric_keys.append(k)

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["controller"] + metric_keys)
            for name, metrics in results.items():
                row = [name] + [metrics.get(k, "") for k in metric_keys]
                writer.writerow(row)

        return csv_path, json_path

    # ------------------------------------------------------------------
    # Pretty-printing
    # ------------------------------------------------------------------
    @staticmethod
    def compare_table(
        results: Dict[str, Dict[str, float]],
    ) -> str:
        """Return a formatted ASCII table comparing controllers.

        Parameters
        ----------
        results : dict
            Output of :meth:`evaluate_all` (or a manually constructed dict).

        Returns
        -------
        str
            A multi-line string containing the comparison table.
        """
        columns = [
            "controller",
            "success_rate",
            "mean_return",
            "path_length",
            "collision_rate",
            "mean_inference_latency",
        ]
        # Determine column widths.
        widths = {c: len(c) for c in columns}
        rows: List[List[str]] = []
        for name, metrics in results.items():
            row = [
                name,
                f"{metrics.get('success_rate', 0.0):.3f}",
                f"{metrics.get('mean_return', 0.0):.3f}",
                f"{metrics.get('path_length', 0.0):.3f}",
                f"{metrics.get('collision_rate', 0.0):.3f}",
                f"{metrics.get('mean_inference_latency', 0.0):.6f}",
            ]
            rows.append(row)
            for i, c in enumerate(columns):
                widths[c] = max(widths[c], len(row[i]))

        # Build separator and header.
        def _fmt_row(cells: List[str]) -> str:
            parts = []
            for i, c in enumerate(columns):
                if c == "controller":
                    parts.append(cells[i].ljust(widths[c]))
                else:
                    parts.append(cells[i].rjust(widths[c]))
            return "| " + " | ".join(parts) + " |"

        separator = "+" + "+".join(
            "-" * (widths[c] + 2) for c in columns
        ) + "+"

        lines = [separator, _fmt_row(columns), separator]
        for row in rows:
            lines.append(_fmt_row(row))
        lines.append(separator)
        return "\n".join(lines)
