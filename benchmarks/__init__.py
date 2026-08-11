"""Benchmark environments for the MPC vs VLA vs Diffusion study.

This package provides a unified, framework-agnostic environment interface
(:class:`BaseEnv`) and a set of concrete benchmark tasks that can be used to
evaluate and compare Model Predictive Control (MPC), Vision-Language-Action
(VLA) models, and Diffusion Policies.

Environments
------------
* :class:`ReachingEnv` -- 2-D / 3-D point-mass reaching with optional obstacles.
* :class:`PushTEnv` -- T-shaped block pushing task.
* :class:`MetaWorldWrapper` -- wrapper for MetaWorld manipulation tasks
  (optional dependency).

Utilities
---------
* :class:`EnvSpec` -- lightweight observation/action space descriptor.
* :class:`Obstacle` -- circular/spherical obstacle used by ``ReachingEnv``.
* :class:`DemonstrationCollector` -- collect (obs, action, next_obs, reward,
  done) transitions from any environment.
* :class:`Evaluator` -- run controllers on environments and collect metrics.
"""

from __future__ import annotations

from .base_env import BaseEnv, EnvSpec
from .reaching_env import Obstacle, ReachingEnv
from .pusht_env import PushTEnv
from .demonstration_collector import DemonstrationCollector
from .evaluation import Evaluator

# MetaWorld is an optional dependency; import lazily so that the package can be
# imported even when MetaWorld is not installed.
try:
    from .metaworld_wrapper import (
        SUPPORTED_METAWORLD_TASKS,
        MetaWorldWrapper,
    )
except ImportError:  # pragma: no cover - MetaWorld not installed
    SUPPORTED_METAWORLD_TASKS = ()
    MetaWorldWrapper = None  # type: ignore[assignment, misc]

__all__ = [
    "BaseEnv",
    "EnvSpec",
    "Obstacle",
    "ReachingEnv",
    "PushTEnv",
    "MetaWorldWrapper",
    "SUPPORTED_METAWORLD_TASKS",
    "DemonstrationCollector",
    "Evaluator",
]
