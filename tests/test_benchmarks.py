"""Unit tests for the benchmarks package.

Run with:
    conda run -n mpc_vla pytest tests/test_benchmarks.py -v

These tests exercise the ReachingEnv, PushTEnv, DemonstrationCollector, and
Evaluator classes without requiring optional dependencies (MetaWorld, torch).
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest

# Ensure the study root is on the path so `benchmarks` can be imported
# regardless of where pytest is invoked from.
_STUDY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _STUDY_ROOT not in sys.path:
    sys.path.insert(0, _STUDY_ROOT)

from benchmarks import (
    BaseEnv,
    DemonstrationCollector,
    EnvSpec,
    Evaluator,
    Obstacle,
    PushTEnv,
    ReachingEnv,
)
from benchmarks.evaluation import Evaluator as EvaluatorDirect


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def reaching_env():
    """A simple 2-D reaching environment without obstacles."""
    return ReachingEnv(dim=2, max_steps=50, image_size=32, seed=42)


@pytest.fixture
def reaching_env_cluttered():
    """A 2-D reaching environment with a single obstacle."""
    obs = [Obstacle(center=[0.0, 0.0], radius=0.5)]
    return ReachingEnv(
        dim=2, max_steps=50, image_size=32, obstacles=obs, seed=42
    )


@pytest.fixture
def pusht_env():
    """A PushT environment with a small image size for speed."""
    return PushTEnv(max_steps=50, image_size=32, seed=42)


# ---------------------------------------------------------------------------
# ReachingEnv tests
# ---------------------------------------------------------------------------
class TestReachingEnv:
    """Tests for the 2-D ReachingEnv."""

    def test_reset_returns_observation(self, reaching_env):
        """reset() should return an observation of the correct shape."""
        obs = reaching_env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (4,)  # [x, y, vx, vy]
        assert obs.dtype == np.float32

    def test_reset_with_seed_reproducible(self, reaching_env):
        """reset(seed) should be reproducible."""
        obs1 = reaching_env.reset(seed=123)
        obs2 = ReachingEnv(dim=2, max_steps=50, image_size=32, seed=123).reset()
        np.testing.assert_allclose(obs1, obs2, atol=1e-6)

    def test_step_returns_valid_tuple(self, reaching_env):
        """step() should return (obs, reward, done, info)."""
        reaching_env.reset()
        action = np.array([0.1, 0.0], dtype=np.float32)
        obs, reward, done, info = reaching_env.step(action)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (4,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert "distance" in info
        assert "step" in info

    def test_step_increments_count(self, reaching_env):
        reaching_env.reset()
        assert reaching_env.step_count == 0
        reaching_env.step(np.zeros(2))
        assert reaching_env.step_count == 1
        reaching_env.step(np.zeros(2))
        assert reaching_env.step_count == 2

    def test_success_detection(self):
        """is_success() should be True when agent is at the target."""
        env = ReachingEnv(dim=2, success_threshold=0.5, max_steps=50, seed=0)
        env.reset(seed=0)
        # Manually place agent on target.
        env._state[:2] = env._target.copy()
        env._state[2:] = 0.0
        assert env.is_success()

    def test_success_false_when_far(self, reaching_env):
        reaching_env.reset()
        # Move agent far from target.
        reaching_env._state[:2] = reaching_env._target + np.array([10.0, 10.0])
        assert not reaching_env.is_success()

    def test_collision_detection(self, reaching_env_cluttered):
        """is_collision() should be True when agent is inside an obstacle."""
        env = reaching_env_cluttered
        env.reset(seed=0)
        # Place agent at the obstacle centre.
        env._state[:2] = np.array([0.0, 0.0])
        assert env.is_collision()

    def test_no_collision_outside_obstacle(self, reaching_env_cluttered):
        env = reaching_env_cluttered
        env.reset(seed=0)
        # Place agent far from the obstacle.
        env._state[:2] = np.array([4.5, 4.5])
        assert not env.is_collision()

    def test_collision_false_no_obstacles(self, reaching_env):
        reaching_env.reset()
        assert not reaching_env.is_collision()

    def test_render_produces_image(self, reaching_env):
        """render() should produce an (H, W, 3) uint8 image."""
        reaching_env.reset()
        img = reaching_env.render(mode="rgb_array")
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3
        assert img.shape[2] == 3
        assert img.shape[0] == reaching_env.image_size
        assert img.shape[1] == reaching_env.image_size
        assert img.dtype == np.uint8

    def test_get_image_same_as_render(self, reaching_env):
        reaching_env.reset()
        img1 = reaching_env.get_image()
        img2 = reaching_env.render()
        np.testing.assert_array_equal(img1, img2)

    def test_get_state(self, reaching_env):
        reaching_env.reset()
        state = reaching_env.get_state()
        assert state.shape == (4,)
        assert state.dtype == np.float32

    def test_get_target(self, reaching_env):
        reaching_env.reset()
        target = reaching_env.get_target()
        assert target.shape == (2,)

    def test_language_instruction(self, reaching_env):
        reaching_env.reset()
        instr = reaching_env.get_language_instruction()
        assert isinstance(instr, str)
        assert len(instr) > 0

    def test_done_on_timeout(self):
        env = ReachingEnv(dim=2, max_steps=3, seed=0)
        env.reset(seed=0)
        # Ensure we don't accidentally succeed.
        env._target = env._state[:2] + np.array([100.0, 100.0])
        env.step(np.zeros(2))
        env.step(np.zeros(2))
        _, _, done, _ = env.step(np.zeros(2))
        assert done  # timeout after 3 steps

    def test_action_space(self, reaching_env):
        assert reaching_env.action_space.shape == (2,)
        sample = reaching_env.action_space.sample()
        assert sample.shape == (2,)

    def test_observation_space(self, reaching_env):
        assert reaching_env.observation_space.shape == (4,)

    def test_3d_env(self):
        env = ReachingEnv(dim=3, max_steps=20, image_size=16, seed=1)
        obs = env.reset()
        assert obs.shape == (6,)
        img = env.get_image()
        assert img.shape == (16, 16, 3)

    def test_invalid_dim_raises(self):
        with pytest.raises(ValueError):
            ReachingEnv(dim=4)


# ---------------------------------------------------------------------------
# PushTEnv tests
# ---------------------------------------------------------------------------
class TestPushTEnv:
    """Tests for the PushTEnv."""

    def test_reset_returns_observation(self, pusht_env):
        obs = pusht_env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (5,)  # [block_x, block_y, block_angle, agent_x, agent_y]
        assert obs.dtype == np.float32

    def test_step_returns_valid_tuple(self, pusht_env):
        pusht_env.reset()
        action = np.array([0.1, 0.0], dtype=np.float32)
        obs, reward, done, info = pusht_env.step(action)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (5,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert "iou" in info

    def test_success_detection(self):
        """is_success() should be True when IoU exceeds threshold."""
        env = PushTEnv(success_iou_threshold=0.1, max_steps=50, seed=0)
        env.reset(seed=0)
        # Align block with target.
        env._block = env._target.copy()
        assert env.is_success()

    def test_success_false_when_misaligned(self, pusht_env):
        pusht_env.reset()
        # Move block far from target.
        pusht_env._block = np.array(
            [-pusht_env.workspace / 2, -pusht_env.workspace / 2, 0.0]
        )
        pusht_env._target = np.array(
            [pusht_env.workspace / 2, pusht_env.workspace / 2, 0.0]
        )
        assert not pusht_env.is_success()

    def test_no_collision(self, pusht_env):
        pusht_env.reset()
        assert not pusht_env.is_collision()

    def test_render_produces_image(self, pusht_env):
        pusht_env.reset()
        img = pusht_env.render(mode="rgb_array")
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3
        assert img.shape[2] == 3
        assert img.shape[0] == pusht_env.image_size
        assert img.shape[1] == pusht_env.image_size
        assert img.dtype == np.uint8

    def test_get_state(self, pusht_env):
        pusht_env.reset()
        state = pusht_env.get_state()
        assert state.shape == (5,)

    def test_language_instruction(self, pusht_env):
        pusht_env.reset()
        instr = pusht_env.get_language_instruction()
        assert isinstance(instr, str)
        assert "T" in instr or "block" in instr.lower()

    def test_action_space(self, pusht_env):
        assert pusht_env.action_space.shape == (2,)

    def test_done_on_timeout(self):
        env = PushTEnv(max_steps=3, image_size=16, seed=0)
        env.reset(seed=0)
        env._block = np.array([-5.0, -5.0, 0.0])
        env._target = np.array([5.0, 5.0, 0.0])
        env.step(np.zeros(2))
        env.step(np.zeros(2))
        _, _, done, _ = env.step(np.zeros(2))
        assert done


# ---------------------------------------------------------------------------
# DemonstrationCollector tests
# ---------------------------------------------------------------------------
class TestDemonstrationCollector:
    """Tests for the DemonstrationCollector."""

    def test_collect_from_reaching_random_policy(self, reaching_env):
        """Collect transitions using a random policy."""
        collector = DemonstrationCollector()

        def random_policy(obs):
            return reaching_env.action_space.sample()

        dataset = collector.from_env(
            reaching_env, random_policy, n_episodes=2, max_steps=10
        )
        assert "observations" in dataset
        assert "actions" in dataset
        assert "next_observations" in dataset
        assert "rewards" in dataset
        assert "dones" in dataset
        # At least some transitions collected.
        n = len(dataset["observations"])
        assert n > 0
        assert dataset["observations"].shape[0] == n
        assert dataset["actions"].shape[0] == n
        assert dataset["rewards"].shape == (n,)
        assert dataset["dones"].dtype == bool

    def test_observation_shape(self, reaching_env):
        collector = DemonstrationCollector()
        dataset = collector.from_env(
            reaching_env, lambda o: np.zeros(2), n_episodes=1, max_steps=5
        )
        assert dataset["observations"].shape[1:] == (4,)
        assert dataset["actions"].shape[1:] == (2,)

    def test_collect_from_expert(self, reaching_env):
        """Collect using a custom expert function with env access."""
        collector = DemonstrationCollector()

        def expert_fn(env):
            # Simple proportional controller towards target.
            pos = env._state[:2]
            vel = env._state[2:]
            err = env.get_target() - pos
            return np.clip(5.0 * err - 2.0 * vel, -1.0, 1.0)

        dataset = collector.from_expert(
            reaching_env, expert_fn, n_episodes=2, max_steps=10
        )
        assert len(dataset["observations"]) > 0

    def test_collect_from_mpc(self, reaching_env):
        """Collect using a mock MPC controller with solve(state, ref)."""

        class MockMPC:
            def solve(self, state, ref):
                pos = state[:2]
                vel = state[2:]
                err = ref - pos
                return np.clip(5.0 * err - 2.0 * vel, -1.0, 1.0)

        collector = DemonstrationCollector()
        dataset = collector.from_mpc(
            reaching_env, MockMPC(), n_episodes=2, max_steps=10
        )
        assert len(dataset["observations"]) > 0

    def test_save_load_roundtrip(self, reaching_env, tmp_path):
        """Save and reload a dataset; data should match."""
        collector = DemonstrationCollector()

        def policy(obs):
            return np.zeros(2, dtype=np.float32)

        collector.from_env(reaching_env, policy, n_episodes=1, max_steps=5)
        dataset_before = collector.get_dataset()

        path = str(tmp_path / "demos")
        collector.save(path)

        # File should exist with .npz extension.
        assert os.path.exists(path + ".npz")

        # Load into a fresh collector.
        collector2 = DemonstrationCollector()
        dataset_after = collector2.load(path)

        for key in dataset_before:
            np.testing.assert_array_equal(dataset_before[key], dataset_after[key])

    def test_get_batch(self, reaching_env):
        collector = DemonstrationCollector()
        collector.from_env(
            reaching_env, lambda o: np.zeros(2), n_episodes=1, max_steps=5
        )
        batch = collector.get_batch(batch_size=3)
        assert batch["observations"].shape[0] == 3
        assert batch["actions"].shape[0] == 3

    def test_get_batch_empty_raises(self):
        collector = DemonstrationCollector()
        with pytest.raises(RuntimeError):
            collector.get_batch(1)

    def test_image_mode(self, reaching_env):
        """Image mode should store image observations."""
        collector = DemonstrationCollector(image_mode=True)
        dataset = collector.from_env(
            reaching_env, lambda o: np.zeros(2), n_episodes=1, max_steps=3
        )
        # Observations should be images (H, W, 3).
        assert dataset["observations"].ndim == 4
        assert dataset["observations"].shape[1:] == (
            reaching_env.image_size,
            reaching_env.image_size,
            3,
        )

    def test_len(self, reaching_env):
        collector = DemonstrationCollector()
        collector.from_env(
            reaching_env, lambda o: np.zeros(2), n_episodes=1, max_steps=5
        )
        assert len(collector) == 5


# ---------------------------------------------------------------------------
# Evaluator tests
# ---------------------------------------------------------------------------
class TestEvaluator:
    """Tests for the Evaluator."""

    def test_evaluate_random_mpc_controller(self, reaching_env):
        """Evaluate a random 'MPC' controller on the reaching env."""

        class RandomMPC:
            def solve(self, state, ref):
                return np.random.uniform(-1, 1, size=2).astype(np.float32)

        evaluator = Evaluator()
        results = evaluator.evaluate(
            RandomMPC(), reaching_env,
            n_episodes=2, seeds=[0, 1], controller_type="mpc",
        )
        assert "success_rate" in results
        assert "mean_return" in results
        assert "path_length" in results
        assert "collision_rate" in results
        assert "mean_inference_latency" in results
        assert 0.0 <= results["success_rate"] <= 1.0
        assert results["mean_inference_latency"] >= 0.0

    def test_evaluate_diffusion_controller(self, reaching_env):
        """Evaluate a mock diffusion controller (controller.sample(obs))."""

        class MockDiffusion:
            def sample(self, obs):
                return np.zeros(2, dtype=np.float32)

        evaluator = Evaluator()
        results = evaluator.evaluate(
            MockDiffusion(), reaching_env,
            n_episodes=2, seeds=[0, 1], controller_type="diffusion",
        )
        assert "success_rate" in results

    def test_evaluate_vla_controller(self, reaching_env):
        """Evaluate a mock VLA controller (predict_action(image, instr))."""

        class MockVLA:
            def predict_action(self, image, instruction):
                return np.zeros(2, dtype=np.float32)

        evaluator = Evaluator()
        results = evaluator.evaluate(
            MockVLA(), reaching_env,
            n_episodes=2, seeds=[0, 1], controller_type="vla",
        )
        assert "success_rate" in results

    def test_invalid_controller_type_raises(self, reaching_env):
        evaluator = Evaluator()
        with pytest.raises(ValueError):
            evaluator.evaluate(
                None, reaching_env, n_episodes=1, controller_type="invalid",
            )

    def test_evaluate_all(self, reaching_env):
        """evaluate_all should return results for every controller."""

        class ZeroController:
            def solve(self, state, ref):
                return np.zeros(2, dtype=np.float32)

            def sample(self, obs):
                return np.zeros(2, dtype=np.float32)

        controllers = {
            "mpc_zero": {"controller": ZeroController(), "type": "mpc"},
            "diff_zero": {"controller": ZeroController(), "type": "diffusion"},
        }
        evaluator = Evaluator()
        results = evaluator.evaluate_all(
            controllers, reaching_env, n_episodes=2, seeds=[0, 1]
        )
        assert "mpc_zero" in results
        assert "diff_zero" in results
        for name in results:
            assert "success_rate" in results[name]

    def test_save_results(self, tmp_path):
        """save_results should write CSV and JSON files."""
        results = {
            "ctrl_a": {
                "success_rate": 0.8,
                "mean_return": 1.5,
                "path_length": 10.0,
                "collision_rate": 0.1,
                "mean_inference_latency": 0.001,
            },
        }
        csv_path, json_path = Evaluator.save_results(
            results, str(tmp_path / "results")
        )
        assert os.path.exists(csv_path)
        assert os.path.exists(json_path)
        assert csv_path.endswith(".csv")
        assert json_path.endswith(".json")

        # Verify JSON content.
        import json
        with open(json_path) as f:
            loaded = json.load(f)
        assert "ctrl_a" in loaded
        assert loaded["ctrl_a"]["success_rate"] == 0.8

    def test_compare_table(self):
        """compare_table should return a formatted string table."""
        results = {
            "ctrl_a": {
                "success_rate": 0.8,
                "mean_return": 1.5,
                "path_length": 10.0,
                "collision_rate": 0.1,
                "mean_inference_latency": 0.001,
            },
            "ctrl_b": {
                "success_rate": 0.5,
                "mean_return": 0.7,
                "path_length": 15.0,
                "collision_rate": 0.2,
                "mean_inference_latency": 0.002,
            },
        }
        table = Evaluator.compare_table(results)
        assert isinstance(table, str)
        assert "ctrl_a" in table
        assert "ctrl_b" in table
        assert "success_rate" in table
        # Should contain table border characters.
        assert "|" in table
        assert "+" in table

    def test_compare_table_empty(self):
        """compare_table should handle empty results gracefully."""
        table = Evaluator.compare_table({})
        assert isinstance(table, str)
        assert "success_rate" in table

    def test_evaluator_direct_import(self):
        """Evaluator should be importable from both paths."""
        from benchmarks.evaluation import Evaluator as Ev1
        from benchmarks import Evaluator as Ev2
        assert Ev1 is Ev2


# ---------------------------------------------------------------------------
# EnvSpec tests
# ---------------------------------------------------------------------------
class TestEnvSpec:
    """Tests for the EnvSpec helper."""

    def test_sample_shape(self):
        spec = EnvSpec(shape=(3,), low=-1.0, high=1.0)
        sample = spec.sample()
        assert sample.shape == (3,)

    def test_broadcast_bounds(self):
        spec = EnvSpec(shape=(2,), low=0.0, high=1.0)
        assert spec.low.shape == (2,)
        assert spec.high.shape == (2,)
        assert np.all(spec.low == 0.0)
        assert np.all(spec.high == 1.0)

    def test_inf_bounds_sample(self):
        spec = EnvSpec(shape=(2,), low=-np.inf, high=np.inf)
        sample = spec.sample()
        assert sample.shape == (2,)


# ---------------------------------------------------------------------------
# Package __init__ tests
# ---------------------------------------------------------------------------
class TestPackageInit:
    """Verify that the package exports the expected names."""

    def test_imports(self):
        import benchmarks
        assert hasattr(benchmarks, "BaseEnv")
        assert hasattr(benchmarks, "EnvSpec")
        assert hasattr(benchmarks, "ReachingEnv")
        assert hasattr(benchmarks, "PushTEnv")
        assert hasattr(benchmarks, "Obstacle")
        assert hasattr(benchmarks, "DemonstrationCollector")
        assert hasattr(benchmarks, "Evaluator")

    def test_all_list(self):
        import benchmarks
        for name in [
            "BaseEnv", "EnvSpec", "Obstacle", "ReachingEnv",
            "PushTEnv", "DemonstrationCollector", "Evaluator",
        ]:
            assert name in benchmarks.__all__
