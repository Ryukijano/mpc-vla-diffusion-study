"""Unit tests for the diffusion_baselines package.

Run with::

    conda run -n mpc_vla pytest tests/test_diffusion_baselines.py -v

All tests use SMALL networks (hidden_dim=32, num_layers=2) so the full
suite completes in a few seconds on CPU.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

# Ensure the project root is on the path so ``diffusion_baselines`` resolves
# regardless of the pytest invocation directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from diffusion_baselines import (
    ConditionalUnet1D,
    CosineSchedule,
    DiffusionPolicy,
    FlowMatchingPolicy,
    FlowSchedule,
    IterativeRegressionPolicy,
    LinearSchedule,
    NoiseSchedule,
    RegressionPolicy,
)

# ---------------------------------------------------------------------------
# Shared test configuration (small networks for speed)
# ---------------------------------------------------------------------------
ACTION_DIM = 2
HORIZON = 8
OBS_DIM = 4
HIDDEN_DIM = 32
NUM_LAYERS = 2
NUM_DIFFUSION_STEPS = 10  # short for fast tests
NUM_FLOW_STEPS = 5
BATCH = 4


def _make_demos(n: int = 16, seed: int = 42):
    """Generate synthetic (obs, action_seq) demonstration pairs."""
    rng = np.random.RandomState(seed)
    demos = []
    for _ in range(n):
        obs = rng.randn(OBS_DIM).astype(np.float32)
        act = rng.randn(HORIZON, ACTION_DIM).astype(np.float32)
        demos.append((obs, act))
    return demos


# ===========================================================================
# ConditionalUnet1D
# ===========================================================================
class TestConditionalUnet1D:
    """Tests for the 1D temporal U-Net denoising network."""

    def test_forward_shape(self):
        """Forward pass produces output with the correct shape."""
        net = ConditionalUnet1D(
            input_dim=ACTION_DIM,
            global_cond_dim=OBS_DIM,
            down_dims=[HIDDEN_DIM * (2 ** i) for i in range(NUM_LAYERS)],
        )
        B = BATCH
        sample = torch.randn(B, HORIZON, ACTION_DIM)
        timestep = torch.randint(0, NUM_DIFFUSION_STEPS, (B,))
        global_cond = torch.randn(B, OBS_DIM)
        out = net(sample, timestep, global_cond=global_cond)
        assert out.shape == (B, HORIZON, ACTION_DIM), (
            f"Expected {(B, HORIZON, ACTION_DIM)}, got {tuple(out.shape)}"
        )

    def test_forward_unconditional(self):
        """Forward pass works without global conditioning."""
        net = ConditionalUnet1D(
            input_dim=ACTION_DIM,
            global_cond_dim=None,
            down_dims=[HIDDEN_DIM * (2 ** i) for i in range(NUM_LAYERS)],
        )
        B = BATCH
        sample = torch.randn(B, HORIZON, ACTION_DIM)
        timestep = torch.randint(0, NUM_DIFFUSION_STEPS, (B,))
        out = net(sample, timestep, global_cond=None)
        assert out.shape == (B, HORIZON, ACTION_DIM)

    def test_forward_scalar_timestep(self):
        """Forward pass accepts a scalar timestep."""
        net = ConditionalUnet1D(
            input_dim=ACTION_DIM,
            global_cond_dim=OBS_DIM,
            down_dims=[HIDDEN_DIM * (2 ** i) for i in range(NUM_LAYERS)],
        )
        B = BATCH
        sample = torch.randn(B, HORIZON, ACTION_DIM)
        global_cond = torch.randn(B, OBS_DIM)
        out = net(sample, 5, global_cond=global_cond)
        assert out.shape == (B, HORIZON, ACTION_DIM)


# ===========================================================================
# Noise schedules
# ===========================================================================
class TestNoiseSchedules:
    """Tests for the noise schedule classes."""

    def test_linear_add_noise_shape(self):
        """LinearSchedule.add_noise produces the correct shape."""
        sched = LinearSchedule(num_steps=NUM_DIFFUSION_STEPS)
        B = BATCH
        x = torch.randn(B, HORIZON, ACTION_DIM)
        t = torch.randint(0, NUM_DIFFUSION_STEPS, (B,))
        noise = torch.randn_like(x)
        x_t = sched.add_noise(x, t, noise)
        assert x_t.shape == x.shape

    def test_linear_step_shape(self):
        """LinearSchedule.step produces the correct shape."""
        sched = LinearSchedule(num_steps=NUM_DIFFUSION_STEPS)
        B = BATCH
        x_t = torch.randn(B, HORIZON, ACTION_DIM)
        t = torch.randint(1, NUM_DIFFUSION_STEPS, (B,))
        pred = torch.randn_like(x_t)
        x_prev = sched.step(x_t, t, pred)
        assert x_prev.shape == x_t.shape

    def test_cosine_add_noise_shape(self):
        """CosineSchedule.add_noise produces the correct shape."""
        sched = CosineSchedule(num_steps=NUM_DIFFUSION_STEPS)
        B = BATCH
        x = torch.randn(B, HORIZON, ACTION_DIM)
        t = torch.randint(0, NUM_DIFFUSION_STEPS, (B,))
        noise = torch.randn_like(x)
        x_t = sched.add_noise(x, t, noise)
        assert x_t.shape == x.shape

    def test_cosine_step_shape(self):
        """CosineSchedule.step produces the correct shape."""
        sched = CosineSchedule(num_steps=NUM_DIFFUSION_STEPS)
        B = BATCH
        x_t = torch.randn(B, HORIZON, ACTION_DIM)
        t = torch.randint(1, NUM_DIFFUSION_STEPS, (B,))
        pred = torch.randn_like(x_t)
        x_prev = sched.step(x_t, t, pred)
        assert x_prev.shape == x_t.shape

    def test_flow_add_noise_shape(self):
        """FlowSchedule.add_noise produces the correct shape."""
        sched = FlowSchedule(num_steps=NUM_FLOW_STEPS)
        B = BATCH
        x = torch.randn(B, HORIZON, ACTION_DIM)
        t = torch.rand(B)  # continuous in [0, 1)
        noise = torch.randn_like(x)
        x_t = sched.add_noise(x, t, noise)
        assert x_t.shape == x.shape

    def test_flow_step_shape(self):
        """FlowSchedule.step produces the correct shape."""
        sched = FlowSchedule(num_steps=NUM_FLOW_STEPS)
        B = BATCH
        x_t = torch.randn(B, HORIZON, ACTION_DIM)
        t = torch.rand(B)
        pred = torch.randn_like(x_t)
        x_next = sched.step(x_t, t, pred)
        assert x_next.shape == x_t.shape

    def test_compute_alpha_shape(self):
        """compute_alpha returns a 1-D tensor of length B."""
        sched = LinearSchedule(num_steps=NUM_DIFFUSION_STEPS)
        B = BATCH
        t = torch.randint(0, NUM_DIFFUSION_STEPS, (B,))
        alpha = sched.compute_alpha(t)
        assert alpha.shape == (B,)

    def test_base_schedule_step_not_implemented(self):
        """Base NoiseSchedule.step raises NotImplementedError."""
        sched = NoiseSchedule(num_steps=NUM_DIFFUSION_STEPS)
        x_t = torch.randn(BATCH, HORIZON, ACTION_DIM)
        t = torch.zeros(BATCH, dtype=torch.long)
        pred = torch.randn_like(x_t)
        with pytest.raises(NotImplementedError):
            sched.step(x_t, t, pred)


# ===========================================================================
# DiffusionPolicy (DDPM)
# ===========================================================================
class TestDiffusionPolicy:
    """Tests for the DDPM Diffusion Policy."""

    def test_init(self):
        """Policy initialises with correct attributes."""
        policy = DiffusionPolicy(
            action_dim=ACTION_DIM,
            horizon=HORIZON,
            obs_dim=OBS_DIM,
            num_diffusion_steps=NUM_DIFFUSION_STEPS,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            device="cpu",
        )
        assert policy.action_dim == ACTION_DIM
        assert policy.horizon == HORIZON
        assert policy.obs_dim == OBS_DIM
        assert policy.num_diffusion_steps == NUM_DIFFUSION_STEPS
        assert policy.net is not None

    def test_training_runs(self):
        """Training runs for a few epochs and returns losses."""
        policy = DiffusionPolicy(
            action_dim=ACTION_DIM,
            horizon=HORIZON,
            obs_dim=OBS_DIM,
            num_diffusion_steps=NUM_DIFFUSION_STEPS,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            device="cpu",
        )
        demos = _make_demos(n=16)
        losses = policy.train(demos, epochs=3, batch_size=8, lr=1e-3)
        assert len(losses) == 3
        assert all(isinstance(l, float) for l in losses)
        assert policy._trained is True

    def test_sampling_shape(self):
        """Sampling produces an action sequence of the correct shape."""
        policy = DiffusionPolicy(
            action_dim=ACTION_DIM,
            horizon=HORIZON,
            obs_dim=OBS_DIM,
            num_diffusion_steps=NUM_DIFFUSION_STEPS,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            device="cpu",
        )
        demos = _make_demos(n=8)
        policy.train(demos, epochs=2, batch_size=8, lr=1e-3)
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        actions = policy.sample(obs, num_samples=1)
        assert actions.shape == (1, HORIZON, ACTION_DIM), (
            f"Expected {(1, HORIZON, ACTION_DIM)}, got {tuple(actions.shape)}"
        )

    def test_sampling_multiple(self):
        """Sampling with num_samples > 1 produces the correct shape."""
        policy = DiffusionPolicy(
            action_dim=ACTION_DIM,
            horizon=HORIZON,
            obs_dim=OBS_DIM,
            num_diffusion_steps=NUM_DIFFUSION_STEPS,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            device="cpu",
        )
        demos = _make_demos(n=8)
        policy.train(demos, epochs=1, batch_size=8, lr=1e-3)
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        actions = policy.sample(obs, num_samples=3)
        assert actions.shape == (3, HORIZON, ACTION_DIM)

    def test_save_load(self, tmp_path):
        """Save and load round-trips correctly."""
        policy = DiffusionPolicy(
            action_dim=ACTION_DIM,
            horizon=HORIZON,
            obs_dim=OBS_DIM,
            num_diffusion_steps=NUM_DIFFUSION_STEPS,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            device="cpu",
        )
        demos = _make_demos(n=8)
        policy.train(demos, epochs=1, batch_size=8, lr=1e-3)
        path = str(tmp_path / "ddpm.pt")
        policy.save(path)
        loaded = DiffusionPolicy.from_checkpoint(path, device="cpu")
        assert loaded.action_dim == ACTION_DIM
        assert loaded.horizon == HORIZON
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        out = loaded.sample(obs, num_samples=1)
        assert out.shape == (1, HORIZON, ACTION_DIM)


# ===========================================================================
# FlowMatchingPolicy
# ===========================================================================
class TestFlowMatchingPolicy:
    """Tests for the Flow Matching Policy."""

    def test_init(self):
        """Policy initialises with correct attributes."""
        policy = FlowMatchingPolicy(
            action_dim=ACTION_DIM,
            horizon=HORIZON,
            obs_dim=OBS_DIM,
            num_flow_steps=NUM_FLOW_STEPS,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            device="cpu",
        )
        assert policy.action_dim == ACTION_DIM
        assert policy.horizon == HORIZON
        assert policy.obs_dim == OBS_DIM
        assert policy.num_flow_steps == NUM_FLOW_STEPS
        assert policy.net is not None

    def test_training_runs(self):
        """Training runs for a few epochs and returns losses."""
        policy = FlowMatchingPolicy(
            action_dim=ACTION_DIM,
            horizon=HORIZON,
            obs_dim=OBS_DIM,
            num_flow_steps=NUM_FLOW_STEPS,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            device="cpu",
        )
        demos = _make_demos(n=16)
        losses = policy.train(demos, epochs=3, batch_size=8, lr=1e-3)
        assert len(losses) == 3
        assert all(isinstance(l, float) for l in losses)
        assert policy._trained is True

    def test_sampling_shape(self):
        """Sampling produces an action sequence of the correct shape."""
        policy = FlowMatchingPolicy(
            action_dim=ACTION_DIM,
            horizon=HORIZON,
            obs_dim=OBS_DIM,
            num_flow_steps=NUM_FLOW_STEPS,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            device="cpu",
        )
        demos = _make_demos(n=8)
        policy.train(demos, epochs=2, batch_size=8, lr=1e-3)
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        actions = policy.sample(obs, num_samples=1)
        assert actions.shape == (1, HORIZON, ACTION_DIM), (
            f"Expected {(1, HORIZON, ACTION_DIM)}, got {tuple(actions.shape)}"
        )

    def test_save_load(self, tmp_path):
        """Save and load round-trips correctly."""
        policy = FlowMatchingPolicy(
            action_dim=ACTION_DIM,
            horizon=HORIZON,
            obs_dim=OBS_DIM,
            num_flow_steps=NUM_FLOW_STEPS,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            device="cpu",
        )
        demos = _make_demos(n=8)
        policy.train(demos, epochs=1, batch_size=8, lr=1e-3)
        path = str(tmp_path / "flow.pt")
        policy.save(path)
        loaded = FlowMatchingPolicy.from_checkpoint(path, device="cpu")
        assert loaded.action_dim == ACTION_DIM
        assert loaded.num_flow_steps == NUM_FLOW_STEPS
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        out = loaded.sample(obs, num_samples=1)
        assert out.shape == (1, HORIZON, ACTION_DIM)


# ===========================================================================
# RegressionPolicy (RCP)
# ===========================================================================
class TestRegressionPolicy:
    """Tests for the Regression Policy (RCP baseline)."""

    def test_init(self):
        """Policy initialises with correct attributes."""
        policy = RegressionPolicy(
            action_dim=ACTION_DIM,
            horizon=HORIZON,
            obs_dim=OBS_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            device="cpu",
        )
        assert policy.action_dim == ACTION_DIM
        assert policy.horizon == HORIZON
        assert policy.obs_dim == OBS_DIM
        assert policy.net is not None

    def test_training_runs(self):
        """Training runs for a few epochs and returns losses."""
        policy = RegressionPolicy(
            action_dim=ACTION_DIM,
            horizon=HORIZON,
            obs_dim=OBS_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            device="cpu",
        )
        demos = _make_demos(n=16)
        losses = policy.train(demos, num_epochs=5, batch_size=8, lr=1e-3)
        assert len(losses) == 5
        assert all(isinstance(l, float) for l in losses)
        assert policy._trained is True

    def test_prediction_shape(self):
        """Prediction produces an action sequence of the correct shape."""
        policy = RegressionPolicy(
            action_dim=ACTION_DIM,
            horizon=HORIZON,
            obs_dim=OBS_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            device="cpu",
        )
        demos = _make_demos(n=8)
        policy.train(demos, num_epochs=3, batch_size=8, lr=1e-3)
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action = policy.predict(obs)
        assert action.shape == (HORIZON, ACTION_DIM), (
            f"Expected {(HORIZON, ACTION_DIM)}, got {tuple(action.shape)}"
        )

    def test_prediction_batch(self):
        """Prediction handles a batch of observations."""
        policy = RegressionPolicy(
            action_dim=ACTION_DIM,
            horizon=HORIZON,
            obs_dim=OBS_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            device="cpu",
        )
        demos = _make_demos(n=8)
        policy.train(demos, num_epochs=2, batch_size=8, lr=1e-3)
        obs = np.random.randn(3, OBS_DIM).astype(np.float32)
        actions = policy.predict(obs)
        assert actions.shape == (3, HORIZON, ACTION_DIM)

    def test_save_load(self, tmp_path):
        """Save and load round-trips correctly."""
        policy = RegressionPolicy(
            action_dim=ACTION_DIM,
            horizon=HORIZON,
            obs_dim=OBS_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            device="cpu",
        )
        demos = _make_demos(n=8)
        policy.train(demos, num_epochs=2, batch_size=8, lr=1e-3)
        path = str(tmp_path / "rcp.pt")
        policy.save(path)
        loaded = RegressionPolicy.from_checkpoint(path, device="cpu")
        assert loaded.action_dim == ACTION_DIM
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        out = loaded.predict(obs)
        assert out.shape == (HORIZON, ACTION_DIM)


# ===========================================================================
# IterativeRegressionPolicy (MIP)
# ===========================================================================
class TestIterativeRegressionPolicy:
    """Tests for the Iterative Regression Policy (MIP baseline)."""

    def test_init(self):
        """Policy initialises with correct attributes."""
        policy = IterativeRegressionPolicy(
            action_dim=ACTION_DIM,
            horizon=HORIZON,
            obs_dim=OBS_DIM,
            num_iterations=2,
            hidden_dim=HIDDEN_DIM,
            noise_std=0.1,
            device="cpu",
        )
        assert policy.action_dim == ACTION_DIM
        assert policy.horizon == HORIZON
        assert policy.obs_dim == OBS_DIM
        assert policy.num_iterations == 2
        assert policy.noise_std == 0.1
        assert len(policy.nets) == 2

    def test_training_runs(self):
        """Training runs for a few epochs and returns losses."""
        policy = IterativeRegressionPolicy(
            action_dim=ACTION_DIM,
            horizon=HORIZON,
            obs_dim=OBS_DIM,
            num_iterations=2,
            hidden_dim=HIDDEN_DIM,
            noise_std=0.1,
            device="cpu",
        )
        demos = _make_demos(n=16)
        losses = policy.train(demos, num_epochs=3, batch_size=8, lr=1e-3)
        # num_epochs per step * num_iterations steps
        assert len(losses) == 3 * 2
        assert all(isinstance(l, float) for l in losses)
        assert policy._trained is True

    def test_prediction_shape(self):
        """Prediction produces an action sequence of the correct shape."""
        policy = IterativeRegressionPolicy(
            action_dim=ACTION_DIM,
            horizon=HORIZON,
            obs_dim=OBS_DIM,
            num_iterations=2,
            hidden_dim=HIDDEN_DIM,
            noise_std=0.1,
            device="cpu",
        )
        demos = _make_demos(n=8)
        policy.train(demos, num_epochs=2, batch_size=8, lr=1e-3)
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action = policy.predict(obs)
        assert action.shape == (HORIZON, ACTION_DIM), (
            f"Expected {(HORIZON, ACTION_DIM)}, got {tuple(action.shape)}"
        )

    def test_prediction_batch(self):
        """Prediction handles a batch of observations."""
        policy = IterativeRegressionPolicy(
            action_dim=ACTION_DIM,
            horizon=HORIZON,
            obs_dim=OBS_DIM,
            num_iterations=2,
            hidden_dim=HIDDEN_DIM,
            noise_std=0.1,
            device="cpu",
        )
        demos = _make_demos(n=8)
        policy.train(demos, num_epochs=2, batch_size=8, lr=1e-3)
        obs = np.random.randn(3, OBS_DIM).astype(np.float32)
        actions = policy.predict(obs)
        assert actions.shape == (3, HORIZON, ACTION_DIM)

    def test_mip_config(self):
        """Verify MIP config: 2 iterations with noise_std > 0."""
        policy = IterativeRegressionPolicy(
            action_dim=ACTION_DIM,
            horizon=HORIZON,
            obs_dim=OBS_DIM,
            num_iterations=2,
            hidden_dim=HIDDEN_DIM,
            noise_std=0.1,
            device="cpu",
        )
        # MIP is defined as num_iterations=2 and noise_std > 0
        assert policy.num_iterations == 2
        assert policy.noise_std > 0
        assert len(policy.nets) == 2

    def test_single_iteration_no_noise(self):
        """With num_iterations=1 and noise_std=0, behaves like plain regression."""
        policy = IterativeRegressionPolicy(
            action_dim=ACTION_DIM,
            horizon=HORIZON,
            obs_dim=OBS_DIM,
            num_iterations=1,
            hidden_dim=HIDDEN_DIM,
            noise_std=0.0,
            device="cpu",
        )
        demos = _make_demos(n=8)
        losses = policy.train(demos, num_epochs=3, batch_size=8, lr=1e-3)
        assert len(losses) == 3
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action = policy.predict(obs)
        assert action.shape == (HORIZON, ACTION_DIM)

    def test_save_load(self, tmp_path):
        """Save and load round-trips correctly."""
        policy = IterativeRegressionPolicy(
            action_dim=ACTION_DIM,
            horizon=HORIZON,
            obs_dim=OBS_DIM,
            num_iterations=2,
            hidden_dim=HIDDEN_DIM,
            noise_std=0.1,
            device="cpu",
        )
        demos = _make_demos(n=8)
        policy.train(demos, num_epochs=2, batch_size=8, lr=1e-3)
        path = str(tmp_path / "mip.pt")
        policy.save(path)
        loaded = IterativeRegressionPolicy.from_checkpoint(path, device="cpu")
        assert loaded.action_dim == ACTION_DIM
        assert loaded.num_iterations == 2
        assert loaded.noise_std == 0.1
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        out = loaded.predict(obs)
        assert out.shape == (HORIZON, ACTION_DIM)


# ===========================================================================
# Package __init__ exports
# ===========================================================================
class TestPackageExports:
    """Verify that all expected classes are importable from the package."""

    def test_all_exports_present(self):
        import diffusion_baselines

        expected = [
            "ConditionalUnet1D",
            "ConditionalResidualBlock1D",
            "Conv1dBlock",
            "Downsample1d",
            "SinusoidalPosEmb",
            "Upsample1d",
            "DiffusionPolicy",
            "FlowMatchingPolicy",
            "RegressionPolicy",
            "IterativeRegressionPolicy",
            "NoiseSchedule",
            "LinearSchedule",
            "CosineSchedule",
            "FlowSchedule",
        ]
        for name in expected:
            assert hasattr(diffusion_baselines, name), f"Missing export: {name}"
            assert name in diffusion_baselines.__all__, f"Missing from __all__: {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
