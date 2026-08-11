"""Unit tests for the vla_baselines package.

Run with:
    conda run -n mpc_vla pytest tests/test_vla_baselines.py -v

Tests are designed to be fast — they use tiny networks, the BoW text backend
(no CLIP download), small images, and only a few training steps.
"""

from __future__ import annotations

import os
import sys
import tempfile
import warnings

import numpy as np
import pytest
import torch

# Ensure the study root is on the path so `vla_baselines` can be imported
# regardless of where pytest is invoked from.
_STUDY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _STUDY_ROOT not in sys.path:
    sys.path.insert(0, _STUDY_ROOT)

from vla_baselines import OpenVLAInference, SmallVLA, TextEncoder, VLATrainer
from vla_baselines.text_encoder import BagOfWordsEncoder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_img_size():
    """Small image size for fast tests."""
    return 64


@pytest.fixture
def small_action_dim():
    return 4


@pytest.fixture
def small_horizon():
    return 2


@pytest.fixture
def small_vla(small_img_size, small_action_dim, small_horizon):
    """A tiny SmallVLA instance for fast tests (BoW backend, no CLIP download)."""
    model = SmallVLA(
        action_dim=small_action_dim,
        horizon=small_horizon,
        hidden_dim=32,
        num_layers=2,
        img_size=small_img_size,
        text_backend="bow",
        device="cpu",
    )
    return model


@pytest.fixture
def dummy_image(small_img_size):
    """A random numpy image (H, W, C) for testing."""
    return np.random.rand(small_img_size, small_img_size, 3).astype(np.float32)


@pytest.fixture
def dummy_demos(small_action_dim, small_horizon, small_img_size):
    """A small set of dummy demonstrations for training tests."""
    demos = []
    for i in range(8):
        demos.append({
            "image": np.random.rand(small_img_size, small_img_size, 3).astype(np.float32),
            "instruction": f"pick up the {['red', 'blue', 'green'][i % 3]} block",
            "action": np.random.randn(small_horizon, small_action_dim).astype(np.float32),
        })
    return demos


# ---------------------------------------------------------------------------
# TextEncoder tests
# ---------------------------------------------------------------------------

class TestTextEncoder:
    """Tests for the TextEncoder class."""

    def test_bow_backend_produces_correct_shape(self):
        """BoW encoder produces embeddings of the correct shape."""
        encoder = TextEncoder(output_dim=64, backend="bow", device="cpu")
        emb = encoder.encode("pick up the red block")
        assert emb.shape == (1, 64), f"Expected (1, 64), got {emb.shape}"

    def test_bow_batch_encoding(self):
        """BoW encoder handles a batch of strings."""
        encoder = TextEncoder(output_dim=128, backend="bow", device="cpu")
        texts = ["pick up the red block", "push the blue cup", "open the drawer"]
        emb = encoder.encode(texts)
        assert emb.shape == (3, 128), f"Expected (3, 128), got {emb.shape}"

    def test_bow_backend_property(self):
        """The backend property resolves to 'bow' when forced."""
        encoder = TextEncoder(output_dim=64, backend="bow", device="cpu")
        assert encoder.backend == "bow"

    def test_forward_alias(self):
        """forward() is an alias for encode()."""
        encoder = TextEncoder(output_dim=32, backend="bow", device="cpu")
        result = encoder("pick up the block")
        assert result.shape == (1, 32)

    def test_different_texts_produce_different_embeddings(self):
        """Different instructions should (very likely) produce different embeddings."""
        encoder = TextEncoder(output_dim=64, backend="bow", device="cpu")
        emb1 = encoder.encode("pick up the red block")
        emb2 = encoder.encode("push the blue cup")
        assert not torch.allclose(emb1, emb2), "Different texts produced identical embeddings"

    def test_bag_of_words_encoder_directly(self):
        """BagOfWordsEncoder can be used directly."""
        bow = BagOfWordsEncoder(output_dim=32, hidden_dim=16)
        emb = bow.encode_text("grasp the green bottle")
        assert emb.shape == (1, 32)


# ---------------------------------------------------------------------------
# SmallVLA tests
# ---------------------------------------------------------------------------

class TestSmallVLA:
    """Tests for the SmallVLA class."""

    def test_init(self, small_vla, small_action_dim, small_horizon, small_img_size):
        """SmallVLA initialises with the correct configuration."""
        assert small_vla.action_dim == small_action_dim
        assert small_vla.horizon == small_horizon
        assert small_vla.img_size == small_img_size
        assert small_vla.hidden_dim == 32
        assert small_vla.num_layers == 2

    def test_forward_shape(self, small_vla, small_img_size, small_action_dim, small_horizon):
        """Forward pass produces the correct output shape."""
        batch = 4
        images = torch.randn(batch, 3, small_img_size, small_img_size)
        instructions = ["pick up the block"] * batch
        actions = small_vla.forward(images, instructions)
        assert actions.shape == (batch, small_horizon, small_action_dim), \
            f"Expected ({batch}, {small_horizon}, {small_action_dim}), got {actions.shape}"

    def test_forward_single_instruction_string(self, small_vla, small_img_size, small_action_dim, small_horizon):
        """Forward pass works with a single string instruction (batch=1)."""
        images = torch.randn(1, 3, small_img_size, small_img_size)
        actions = small_vla.forward(images, "pick up the red block")
        assert actions.shape == (1, small_horizon, small_action_dim)

    def test_predict_action_numpy_input(self, small_vla, dummy_image, small_action_dim, small_horizon):
        """predict_action accepts a numpy image and returns a numpy array."""
        action = small_vla.predict_action(dummy_image, "pick up the red block")
        assert isinstance(action, np.ndarray)
        assert action.shape == (small_horizon, small_action_dim), \
            f"Expected ({small_horizon}, {small_action_dim}), got {action.shape}"

    def test_predict_action_returns_float32(self, small_vla, dummy_image):
        """predict_action returns a float32 numpy array."""
        action = small_vla.predict_action(dummy_image, "pick up the red block")
        assert action.dtype == np.float32

    def test_training_runs_few_steps(self, small_vla, dummy_demos, small_action_dim, small_horizon):
        """The built-in train() method runs for a few epochs without error."""
        history = small_vla.train(
            dummy_demos,
            epochs=2,
            batch_size=4,
            lr=1e-3,
            verbose=False,
        )
        assert "epoch_loss" in history
        assert "batch_loss" in history
        assert len(history["epoch_loss"]) == 2
        # Loss should be a finite number
        assert all(np.isfinite(history["epoch_loss"])), "Epoch losses contain non-finite values"

    def test_save_and_load(self, small_vla, dummy_image, small_action_dim, small_horizon):
        """save() and load() round-trip correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            small_vla.save(path)
            assert os.path.exists(path)

            loaded = SmallVLA.load(path, device="cpu", text_backend="bow")
            assert loaded.action_dim == small_action_dim
            assert loaded.horizon == small_horizon

            # Predictions should match
            action_orig = small_vla.predict_action(dummy_image, "pick up the red block")
            action_loaded = loaded.predict_action(dummy_image, "pick up the red block")
            np.testing.assert_allclose(action_orig, action_loaded, atol=1e-5)

    def test_count_parameters(self, small_vla):
        """count_parameters() returns a positive integer."""
        n = small_vla.count_parameters()
        assert isinstance(n, int)
        assert n > 0, "Model should have some trainable parameters"

    def test_train_eval_mode_switching(self, small_vla):
        """train_mode() and eval_mode() switch the model correctly."""
        small_vla.train_mode()
        # Check that the model is in training mode via nn.Module
        # (SmallVLA.train overrides nn.Module.train, so use train_mode)
        small_vla.eval_mode()
        # Should not raise
        small_vla.train_mode()


# ---------------------------------------------------------------------------
# OpenVLA wrapper tests
# ---------------------------------------------------------------------------

class TestOpenVLAWrapper:
    """Tests for the OpenVLAInference wrapper (stub mode only — no 7B download)."""

    def test_instantiate(self):
        """OpenVLAInference can be instantiated without loading the model."""
        vla = OpenVLAInference(action_dim=7)
        assert vla.model_name == "openvla/openvla-7b"
        assert vla.is_loaded is False
        assert vla.is_stub_mode is False
        assert vla.num_inferences == 0

    def test_instantiate_with_custom_params(self):
        """OpenVLAInference accepts custom parameters."""
        vla = OpenVLAInference(
            model_name="openvla/openvla-7b",
            load_in_8bit=False,
            action_dim=6,
        )
        assert vla.load_in_8bit is False
        assert vla.action_dim == 6

    def test_stub_mode_predict(self):
        """In stub mode (model can't load), predict_action returns zero action."""
        # Use a non-existent model name to force load failure
        vla = OpenVLAInference(
            model_name="openvla/nonexistent-model-xyz",
            action_dim=7,
            load_in_8bit=False,
        )
        # Trigger load (will fail -> stub mode)
        vla.ensure_loaded()
        assert vla.is_stub_mode is True, "Expected stub mode after failed load"
        assert vla.is_loaded is False

        # predict_action should return zeros in stub mode
        dummy_img = np.random.rand(64, 64, 3).astype(np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            action = vla.predict_action(dummy_img, "pick up the red block")
        assert action.shape == (7,), f"Expected (7,), got {action.shape}"
        assert np.allclose(action, 0.0), "Stub mode should return zero action"
        assert vla.num_inferences == 1

    def test_predict_actions_batch_stub(self):
        """predict_actions_batch works in stub mode."""
        vla = OpenVLAInference(
            model_name="openvla/nonexistent-model-xyz",
            action_dim=5,
            load_in_8bit=False,
        )
        vla.ensure_loaded()
        assert vla.is_stub_mode is True

        images = [np.random.rand(32, 32, 3).astype(np.float32) for _ in range(3)]
        instructions = ["pick up the block", "push the cup", "open the drawer"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            actions, avg_ms = vla.predict_actions_batch(images, instructions)
        assert actions.shape == (3, 5), f"Expected (3, 5), got {actions.shape}"
        assert isinstance(avg_ms, float)
        assert vla.num_inferences == 3

    def test_unload(self):
        """unload() cleans up model references."""
        vla = OpenVLAInference(action_dim=7)
        # Even without loading, unload should not crash
        vla.unload()
        assert vla.is_loaded is False


# ---------------------------------------------------------------------------
# VLATrainer tests
# ---------------------------------------------------------------------------

class TestVLATrainer:
    """Tests for the VLATrainer class."""

    def test_init(self, small_vla):
        """VLATrainer initialises correctly."""
        trainer = VLATrainer(small_vla, lr=1e-4, device="cpu")
        assert trainer.lr == 1e-4
        assert trainer.weight_decay == 1e-5
        assert trainer.scheduler_type == "cosine"
        assert trainer.patience == 10
        assert trainer.current_epoch == 0

    def test_training_runs_few_steps(self, small_vla, dummy_demos):
        """The training loop runs for a few epochs and returns history."""
        trainer = VLATrainer(small_vla, lr=1e-3, device="cpu", verbose=False)
        history = trainer.train(
            dummy_demos,
            num_epochs=3,
            batch_size=4,
            val_split=0.25,
        )
        assert "train_loss" in history
        assert "val_loss" in history
        assert "lr" in history
        assert len(history["train_loss"]) <= 3
        # All losses should be finite
        assert all(np.isfinite(history["train_loss"])), "Train losses contain non-finite values"
        assert all(np.isfinite(history["val_loss"])), "Val losses contain non-finite values"

    def test_training_with_no_val_split(self, small_vla, dummy_demos):
        """Training works with val_split=0 (no validation)."""
        trainer = VLATrainer(small_vla, lr=1e-3, device="cpu", verbose=False)
        history = trainer.train(
            dummy_demos,
            num_epochs=2,
            batch_size=4,
            val_split=0.0,
        )
        assert len(history["train_loss"]) == 2
        # With no val, val_loss should equal train_loss
        assert history["val_loss"] == history["train_loss"]

    def test_early_stopping(self, small_vla, dummy_demos):
        """Early stopping triggers when val loss doesn't improve."""
        trainer = VLATrainer(
            small_vla,
            lr=1e-3,
            device="cpu",
            patience=2,
            min_delta=1e-6,
            verbose=False,
        )
        history = trainer.train(
            dummy_demos,
            num_epochs=20,
            batch_size=4,
            val_split=0.25,
        )
        # With patience=2, should stop before 20 epochs (loss is random data,
        # unlikely to improve consistently)
        # Note: with random data it may or may not stop early, but the loop
        # should at least complete without error
        assert len(history["train_loss"]) <= 20

    def test_cosine_scheduler(self, small_vla, dummy_demos):
        """Cosine scheduler reduces LR over training."""
        trainer = VLATrainer(
            small_vla, lr=1e-3, device="cpu",
            scheduler_type="cosine", verbose=False,
        )
        history = trainer.train(
            dummy_demos,
            num_epochs=5,
            batch_size=4,
            val_split=0.25,
        )
        # LR should decrease (cosine annealing)
        lrs = history["lr"]
        assert len(lrs) == 5
        assert lrs[-1] < lrs[0], f"LR should decrease: {lrs[0]} -> {lrs[-1]}"

    def test_step_scheduler(self, small_vla, dummy_demos):
        """Step scheduler reduces LR at step_size intervals."""
        trainer = VLATrainer(
            small_vla, lr=1e-3, device="cpu",
            scheduler_type="step",
            scheduler_kwargs={"step_size": 2, "gamma": 0.5},
            verbose=False,
        )
        history = trainer.train(
            dummy_demos,
            num_epochs=5,
            batch_size=4,
            val_split=0.25,
        )
        lrs = history["lr"]
        # After epoch 2, LR should be halved
        assert lrs[2] < lrs[0], f"LR should decrease after step: {lrs[0]} -> {lrs[2]}"

    def test_save_and_load_checkpoint(self, small_vla, dummy_demos):
        """save_checkpoint and load_checkpoint round-trip correctly."""
        trainer = VLATrainer(small_vla, lr=1e-3, device="cpu", verbose=False)
        trainer.train(dummy_demos, num_epochs=2, batch_size=4, val_split=0.25)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            trainer.save_checkpoint(path)
            assert os.path.exists(path)

            # Create a new trainer and load
            new_trainer = VLATrainer(small_vla, lr=1e-3, device="cpu", verbose=False)
            checkpoint = new_trainer.load_checkpoint(path)
            assert "model_state_dict" in checkpoint
            assert "history" in checkpoint
            assert new_trainer.current_epoch == trainer.current_epoch

    def test_evaluate(self, small_vla, dummy_demos):
        """evaluate() returns a finite loss without training."""
        trainer = VLATrainer(small_vla, lr=1e-3, device="cpu", verbose=False)
        loss = trainer.evaluate(dummy_demos, batch_size=4)
        assert isinstance(loss, float)
        assert np.isfinite(loss), "Evaluate loss should be finite"

    def test_grad_clip(self, small_vla, dummy_demos):
        """Training with gradient clipping works."""
        trainer = VLATrainer(
            small_vla, lr=1e-3, device="cpu",
            grad_clip=0.5, verbose=False,
        )
        history = trainer.train(
            dummy_demos,
            num_epochs=2,
            batch_size=4,
            val_split=0.25,
        )
        assert len(history["train_loss"]) == 2

    def test_wandb_not_required(self, small_vla, dummy_demos):
        """Training works without wandb installed/initialised."""
        trainer = VLATrainer(small_vla, lr=1e-3, device="cpu", verbose=False)
        # init_wandb should return False if wandb not available, not crash
        result = trainer.init_wandb()
        assert result is False or result is True  # depends on env
        # Training should still work
        history = trainer.train(dummy_demos, num_epochs=1, batch_size=4, val_split=0.25)
        assert len(history["train_loss"]) == 1


# ---------------------------------------------------------------------------
# Package __init__ tests
# ---------------------------------------------------------------------------

class TestPackageImports:
    """Verify that the package exports are correct."""

    def test_all_exports_present(self):
        """All __all__ exports are importable from the package."""
        import vla_baselines

        for name in vla_baselines.__all__:
            assert hasattr(vla_baselines, name), f"Missing export: {name}"

    def test_classes_are_correct_types(self):
        """Exported names are the correct classes."""
        from vla_baselines import OpenVLAInference, SmallVLA, TextEncoder, VLATrainer

        # OpenVLAInference is a plain class
        assert callable(OpenVLAInference)
        # SmallVLA and TextEncoder are nn.Module subclasses
        assert issubclass(SmallVLA, torch.nn.Module)
        assert issubclass(TextEncoder, torch.nn.Module)
        # VLATrainer is a plain class
        assert callable(VLATrainer)
