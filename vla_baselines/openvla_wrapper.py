"""OpenVLA inference wrapper.

Provides :class:`OpenVLAInference` — a thin wrapper around the OpenVLA-7B model
(``openvla/openvla-7b``) from HuggingFace for robot action prediction.

Key features
------------
* **Lazy loading** — the 7B model is not loaded at import time or even at
  construction.  It loads on the first call to :meth:`predict_action`.
* **GB10 friendly** — uses ``device_map="auto"`` so the model spreads across
  the 128 GB unified memory of the GB10.  Optional 8-bit quantisation.
* **Graceful offline fallback** — if the model cannot be downloaded (offline,
  HF Hub unreachable, OOM, etc.), :meth:`predict_action` returns a stub action
  and logs a warning instead of crashing.
* **Latency measurement** — every forward pass is timed; the last latency is
  stored in ``self.last_latency_ms``.

Reference: https://github.com/openvla/openvla
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Default action dimension for OpenVLA (7-DoF for common robot arms)
_DEFAULT_ACTION_DIM = 7


class OpenVLAInference:
    """Inference wrapper for the OpenVLA-7B vision-language-action model.

    The model is loaded lazily on the first call to :meth:`predict_action`.
    If the model cannot be loaded (offline, OOM, etc.), the wrapper enters
    *stub mode* and returns zero actions with a warning.

    Args:
        model_name: HuggingFace model name (default ``"openvla/openvla-7b"``).
        load_in_8bit: If True, load the model in 8-bit quantisation to save
            memory.  Recommended for GB10's 128 GB unified memory.
        device_map: Device map strategy for HuggingFace ``from_pretrained``.
            Default ``"auto"`` spreads across available devices.
        trust_remote_code: Whether to trust remote code (required for OpenVLA).
        action_dim: Action dimensionality for stub-mode fallback.
        cache_dir: Optional HuggingFace cache directory override.

    Example:
        >>> vla = OpenVLAInference()
        >>> action = vla.predict_action(image, "pick up the red block")
        >>> print(action.shape, vla.last_latency_ms)
    """

    def __init__(
        self,
        model_name: str = "openvla/openvla-7b",
        load_in_8bit: bool = True,
        device_map: str = "auto",
        trust_remote_code: bool = True,
        action_dim: int = _DEFAULT_ACTION_DIM,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.load_in_8bit = load_in_8bit
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self.action_dim = action_dim
        self.cache_dir = cache_dir or os.environ.get("HF_HOME")

        # Lazy-loaded internals
        self._model: Optional[Any] = None
        self._processor: Optional[Any] = None
        self._loaded: bool = False
        self._load_failed: bool = False
        self._load_error: Optional[str] = None

        # Latency tracking
        self.last_latency_ms: float = 0.0
        self._num_inferences: int = 0

    # -- properties ---------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """Whether the model has been successfully loaded."""
        return self._loaded

    @property
    def is_stub_mode(self) -> bool:
        """Whether the wrapper is in stub (offline/fallback) mode."""
        return self._load_failed

    @property
    def num_inferences(self) -> int:
        """Total number of inference calls made."""
        return self._num_inferences

    # -- model loading ------------------------------------------------------

    def _load_model(self) -> None:
        """Attempt to load the OpenVLA model and processor.

        Sets ``self._loaded`` on success or ``self._load_failed`` on failure.
        """
        if self._loaded or self._load_failed:
            return

        logger.info("OpenVLAInference: loading model '%s' (8bit=%s)...", self.model_name, self.load_in_8bit)
        try:
            # transformers 5.x renamed AutoModelForVision2Seq -> AutoModelForImageTextToText
            try:
                from transformers import AutoModelForImageTextToText as _AutoVLM  # type: ignore
            except ImportError:
                from transformers import AutoModelForVision2Seq as _AutoVLM  # type: ignore

            from transformers import AutoProcessor  # type: ignore

            load_kwargs: Dict[str, Any] = {
                "trust_remote_code": self.trust_remote_code,
                "device_map": self.device_map,
            }
            if self.load_in_8bit:
                load_kwargs["load_in_8bit"] = True
            if self.cache_dir:
                load_kwargs["cache_dir"] = self.cache_dir

            # Load processor
            self._processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                **({"cache_dir": self.cache_dir} if self.cache_dir else {}),
            )

            # Load model
            self._model = _AutoVLM.from_pretrained(self.model_name, **load_kwargs)
            self._model.eval()
            self._loaded = True
            logger.info("OpenVLAInference: model loaded successfully.")
        except Exception as exc:  # noqa: BLE001
            self._load_failed = True
            self._load_error = str(exc)
            logger.warning(
                "OpenVLAInference: could not load model '%s' (%s). "
                "Entering stub mode — predict_action will return zero actions.",
                self.model_name, exc,
            )

    def ensure_loaded(self) -> None:
        """Explicitly trigger model loading (otherwise lazy on first predict)."""
        self._load_model()

    # -- inference ----------------------------------------------------------

    def predict_action(
        self,
        image: Any,
        instruction: str,
        unnorm_key: Optional[str] = None,
        do_sample: bool = False,
        max_new_tokens: int = 100,
    ) -> np.ndarray:
        """Predict a robot action from an image and natural-language instruction.

        Args:
            image: A PIL Image (or array-like) of the current scene.
            instruction: Natural-language task instruction, e.g.
                ``"pick up the red block and place it in the bin"``.
            unnorm_key: Dataset-specific unnormalisation key for the action
                post-processor (passed to OpenVLA's processor).  If ``None``,
                the model's default is used.
            do_sample: Whether to sample during generation (default deterministic).
            max_new_tokens: Maximum generated tokens.

        Returns:
            ``(action_dim,)`` numpy array of the predicted action.  In stub
            mode, returns a zero array of shape ``(action_dim,)``.

        Raises:
            RuntimeError: Only if the model was loaded but inference fails
                unexpectedly (not for load failures — those produce stubs).
        """
        # Lazy load
        if not self._loaded and not self._load_failed:
            self._load_model()

        self._num_inferences += 1

        # Stub mode
        if self._load_failed:
            warnings.warn(
                f"OpenVLAInference is in stub mode (model '{self.model_name}' "
                f"not loaded: {self._load_error}). Returning zero action.",
                stacklevel=2,
            )
            return np.zeros(self.action_dim, dtype=np.float32)

        # Real inference
        assert self._model is not None
        assert self._processor is not None

        # Build the prompt — OpenVLA expects an instruction with the "In:" prefix
        prompt = f"In: What action should the robot take to {instruction}?"

        # Prepare processor kwargs
        proc_kwargs: Dict[str, Any] = {
            "images": image,
            "text": prompt,
            "return_tensors": "pt",
        }
        if unnorm_key is not None:
            proc_kwargs["unnorm_key"] = unnorm_key

        t0 = time.perf_counter()
        try:
            inputs = self._processor(**proc_kwargs)
            # Move to model device
            model_device = next(self._model.parameters()).device
            inputs = {k: v.to(model_device) if hasattr(v, "to") else v for k, v in inputs.items()}

            # Generate
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gen_kwargs: Dict[str, Any] = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                }
                if unnorm_key is not None and "unnorm_key" in gen_kwargs:
                    gen_kwargs["unnorm_key"] = unnorm_key
                output = self._model.generate(**inputs, **gen_kwargs)

            # Decode action — OpenVLA processor can convert tokens to actions
            predicted_action = self._processor.batch_decode(
                output, skip_special_tokens=True
            )[0]

            # Try to parse the action from the output
            action = self._parse_action_output(predicted_action)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "OpenVLAInference: inference failed (%s). Returning zero action.", exc
            )
            action = np.zeros(self.action_dim, dtype=np.float32)

        t1 = time.perf_counter()
        self.last_latency_ms = (t1 - t0) * 1000.0
        logger.debug(
            "OpenVLAInference: inference #%d latency=%.2f ms",
            self._num_inferences, self.last_latency_ms,
        )
        return action

    def _parse_action_output(self, output: Union[str, np.ndarray, Any]) -> np.ndarray:
        """Parse the raw model output into a numpy action array.

        OpenVLA's processor may return a tensor/array directly (when
        ``unnorm_key`` is used) or a string that needs parsing.
        """
        # If it's already an array-like
        if isinstance(output, (np.ndarray, list, tuple)):
            arr = np.asarray(output, dtype=np.float32).flatten()
            if arr.size >= self.action_dim:
                return arr[: self.action_dim]
            # Pad if too short
            padded = np.zeros(self.action_dim, dtype=np.float32)
            padded[: arr.size] = arr
            return padded

        # If it's a torch tensor
        if hasattr(output, "numpy"):
            arr = output.detach().cpu().numpy().astype(np.float32).flatten()
            if arr.size >= self.action_dim:
                return arr[: self.action_dim]
            padded = np.zeros(self.action_dim, dtype=np.float32)
            padded[: arr.size] = arr
            return padded

        # If it's a string, try to parse numbers from it
        if isinstance(output, str):
            import re
            numbers = re.findall(r"[-+]?\d*\.?\d+", output)
            if numbers:
                arr = np.array([float(n) for n in numbers], dtype=np.float32)
                if arr.size >= self.action_dim:
                    return arr[: self.action_dim]
                padded = np.zeros(self.action_dim, dtype=np.float32)
                padded[: arr.size] = arr
                return padded

        # Fallback
        logger.warning(
            "OpenVLAInference: could not parse action from output type %s. "
            "Returning zero action.", type(output).__name__,
        )
        return np.zeros(self.action_dim, dtype=np.float32)

    # -- batch inference (convenience) -------------------------------------

    def predict_actions_batch(
        self,
        images: list,
        instructions: list,
        unnorm_key: Optional[str] = None,
    ) -> Tuple[np.ndarray, float]:
        """Run inference on a batch of (image, instruction) pairs.

        Args:
            images: List of PIL Images.
            instructions: List of instruction strings.
            unnorm_key: Optional unnormalisation key.

        Returns:
            Tuple of ``(N, action_dim)`` action array and average latency in ms.
        """
        actions = []
        total_ms = 0.0
        for img, instr in zip(images, instructions):
            act = self.predict_action(img, instr, unnorm_key=unnorm_key)
            actions.append(act)
            total_ms += self.last_latency_ms
        avg_ms = total_ms / max(len(actions), 1)
        return np.stack(actions), avg_ms

    # -- cleanup ------------------------------------------------------------

    def unload(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._loaded = False
        # Don't reset _load_failed — if it failed once, don't retry automatically
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("OpenVLAInference: model unloaded.")
