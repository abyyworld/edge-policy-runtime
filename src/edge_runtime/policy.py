"""The policy artifact, and the contract it carries with it.

A deployable policy is weights *plus* the names of the columns it expects and
produces. Shipping weights alone is how a device ends up feeding joint velocity
into the slot the model trained as gripper state: the shapes match, nothing
raises, and the robot behaves strangely for a week.

So the artifact is self-describing, and :meth:`Policy.bind` resolves the
device's observation dictionary against the recorded column names — failing
loudly, with the missing column's name, rather than silently mis-slotting.

The forward pass is numpy. Inference on the device must not import torch: it
costs seconds of startup, hundreds of megabytes of image, and a class of
version-skew failure that is entirely avoidable for a network this size.
:mod:`edge_runtime.convert` handles torch, on the workstation, once.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

# The action contract: 7 joint deltas and a gripper target, in that order.
# Change it here and the manifest carries the change to every device.
ACTION_COLUMNS: tuple[str, ...] = (
    *(f"joint_{i}_delta" for i in range(7)),
    "gripper_target",
)


class PolicyError(RuntimeError):
    pass


class Policy:
    """A feed-forward policy with a named observation contract.

    Attributes
    ----------
    weights, biases
        Per-layer parameters. All layers but the last use tanh; the last is
        linear, because a squashed action head silently clips large corrections
        and those are exactly the ones that matter.
    """

    def __init__(
        self,
        weights: list[np.ndarray],
        biases: list[np.ndarray],
        obs_columns: list[str],
        action_columns: list[str] | None = None,
        name: str = "policy",
    ) -> None:
        if len(weights) != len(biases):
            raise PolicyError("weights and biases disagree on layer count")
        if not weights:
            raise PolicyError("a policy with no layers is not a policy")
        if weights[0].shape[0] != len(obs_columns):
            raise PolicyError(
                f"first layer takes {weights[0].shape[0]} inputs but the contract "
                f"names {len(obs_columns)} observation columns"
            )
        self.weights = [np.asarray(w, dtype=np.float32) for w in weights]
        self.biases = [np.asarray(b, dtype=np.float32) for b in biases]
        self.obs_columns = list(obs_columns)
        self.action_columns = list(action_columns or ACTION_COLUMNS)
        self.name = name

    # -- shape ---------------------------------------------------------------

    @property
    def obs_dim(self) -> int:
        return self.weights[0].shape[0]

    @property
    def action_dim(self) -> int:
        return self.weights[-1].shape[1]

    @property
    def n_parameters(self) -> int:
        return sum(w.size + b.size for w, b in zip(self.weights, self.biases, strict=True))

    # -- inference -----------------------------------------------------------

    def forward(self, obs: np.ndarray) -> np.ndarray:
        """Run the network. Accepts a single observation or a batch."""
        x = np.asarray(obs, dtype=np.float32)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        if x.shape[1] != self.obs_dim:
            raise PolicyError(f"expected {self.obs_dim} observation values, got {x.shape[1]}")

        last = len(self.weights) - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            x = x @ w + b
            if i != last:
                x = np.tanh(x)
        return x[0] if single else x

    def bind(self, observation: dict[str, float | np.ndarray]) -> np.ndarray:
        """Assemble the input vector from a named observation.

        Raises with the missing names rather than reindexing whatever is there.
        """
        missing = [c for c in self.obs_columns if c not in observation]
        if missing:
            raise PolicyError(
                f"observation is missing {len(missing)} column(s) the policy was "
                f"trained on: {', '.join(missing[:6])}" + (" …" if len(missing) > 6 else "")
            )
        return np.array([float(observation[c]) for c in self.obs_columns], dtype=np.float32)

    # -- identity and persistence -------------------------------------------

    def content_hash(self) -> str:
        """SHA-256 over the parameters and the contract.

        Two policies with identical bytes are the same policy regardless of what
        the file was called or which pipeline run produced it — the same rule the
        checkpoint registry upstream uses.
        """
        h = hashlib.sha256()
        for w, b in zip(self.weights, self.biases, strict=True):
            h.update(np.ascontiguousarray(w, dtype=np.float32).tobytes())
            h.update(np.ascontiguousarray(b, dtype=np.float32).tobytes())
        h.update(json.dumps([self.obs_columns, self.action_columns]).encode())
        return h.hexdigest()

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            arrays[f"w{i}"] = w
            arrays[f"b{i}"] = b
        arrays["__contract__"] = np.frombuffer(
            json.dumps(
                {
                    "obs_columns": self.obs_columns,
                    "action_columns": self.action_columns,
                    "name": self.name,
                    "n_layers": len(self.weights),
                }
            ).encode(),
            dtype=np.uint8,
        )
        np.savez(path, **arrays)
        return path

    @classmethod
    def load(cls, path: Path | str) -> Policy:
        with np.load(Path(path), allow_pickle=False) as data:
            contract = json.loads(bytes(data["__contract__"]).decode())
            n = contract["n_layers"]
            weights = [data[f"w{i}"] for i in range(n)]
            biases = [data[f"b{i}"] for i in range(n)]
        return cls(
            weights,
            biases,
            contract["obs_columns"],
            contract["action_columns"],
            contract.get("name", "policy"),
        )


def reference_policy(
    obs_columns: list[str] | None = None,
    hidden: tuple[int, ...] = (128, 128),
    seed: int = 0,
    skill: float = 1.0,
) -> Policy:
    """A stand-in policy, for exercising the runtime without a training run.

    It is not trained on anything. `skill` scales the goal-tracking term, which
    is the knob the OTA demo turns to publish a release that is genuinely worse
    than the one it replaces — an update the health gate has to catch.
    """
    obs_columns = list(obs_columns or default_obs_columns())
    rng = np.random.default_rng(seed)

    dims = [len(obs_columns), *hidden, len(ACTION_COLUMNS)]
    weights, biases = [], []
    for fan_in, fan_out in zip(dims[:-1], dims[1:], strict=True):
        scale = np.sqrt(2.0 / fan_in)
        weights.append(rng.normal(0, scale, (fan_in, fan_out)).astype(np.float32))
        biases.append(np.zeros(fan_out, dtype=np.float32))

    # Wire the goal-error channels straight through to the joint deltas so the
    # policy does something sensible rather than something random.
    error_idx = [i for i, c in enumerate(obs_columns) if c.startswith("goal_error_")]
    for slot, idx in enumerate(error_idx[:7]):
        weights[0][idx, :] *= 0.1
        weights[0][idx, slot] = 2.0 * skill
    for w in weights[1:]:
        w *= 0.35

    # The action head's output channels carry different physical quantities:
    # joint deltas are bounded by a rate cap of a tenth of a radian, while the
    # gripper target is a normalised [0, 1] command. A real trained head has
    # exactly this spread, and it is the reason a single quantization scale for
    # the whole matrix is a poor choice — see edge_runtime.quantize.
    head = weights[-1]
    head[:, :7] *= 0.08
    head[:, 7] *= 1.0

    return Policy(weights, biases, obs_columns, name=f"reference-s{seed}-k{skill:g}")


def default_obs_columns() -> list[str]:
    """The observation contract this repo demonstrates against.

    Seven joint positions, seven goal errors, gripper state, and three
    proprioceptive extras — 18 columns, named, because the names are the part
    that prevents a silent mis-binding.
    """
    return [
        *(f"joint_{i}_pos" for i in range(7)),
        *(f"goal_error_{i}" for i in range(7)),
        "gripper_state",
        "steps_elapsed_norm",
        "goal_distance",
        "last_action_norm",
    ]
