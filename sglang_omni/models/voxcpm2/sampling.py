# SPDX-License-Identifier: Apache-2.0
"""Validated inference settings for the VoxCPM2 flow sampler.

These settings describe inference, not a promise that a checkpoint was trained
for a particular step count. In particular, ordinary voice LoRAs do not imply
support for one-step inference.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from math import isfinite
from typing import Any


@dataclass(frozen=True)
class VoxCPM2Sampling:
    inference_timesteps: int = 10
    cfg_value: float = 2.0
    sway_sampling_coef: float = 1.0
    use_cfg_zero_star: bool = True

    def __post_init__(self) -> None:
        steps = self.inference_timesteps
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("inference_timesteps must be a positive integer")
        for name in ("cfg_value", "sway_sampling_coef"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
        if self.cfg_value < 0:
            raise ValueError("cfg_value must be nonnegative")
        if not 0 <= self.sway_sampling_coef <= 1:
            raise ValueError("sway_sampling_coef must be between 0 and 1")
        if not isinstance(self.use_cfg_zero_star, bool):
            raise TypeError("use_cfg_zero_star must be a boolean")
        if steps == 1 and self.use_cfg_zero_star:
            raise ValueError(
                "one-step sampling with use_cfg_zero_star=True skips the only "
                "DiT evaluation; use the checkpoint's inference recipe and "
                "explicitly set use_cfg_zero_star=False for one-step inference"
            )

    @classmethod
    def from_sources(cls, *sources: Mapping[str, Any]) -> VoxCPM2Sampling:
        """Resolve highest-priority source first, preserving explicit False/0."""
        values = {}
        for field in fields(cls):
            for source in sources:
                if source.get(field.name) is not None:
                    values[field.name] = source[field.name]
                    break
        return cls(**values)

    @classmethod
    def from_state(cls, state: Any) -> VoxCPM2Sampling:
        return cls(**{field.name: getattr(state, field.name) for field in fields(cls)})

    @classmethod
    def for_batch(cls, states: Sequence[Any]) -> VoxCPM2Sampling:
        if not states:
            raise ValueError("cannot resolve sampling for an empty batch")
        first = cls.from_state(states[0])
        for state in states[1:]:
            other = cls.from_state(state)
            for field in fields(cls):
                if getattr(first, field.name) != getattr(other, field.name):
                    raise ValueError(
                        f"VoxCPM2 cannot batch requests with different {field.name}"
                    )
        return first
