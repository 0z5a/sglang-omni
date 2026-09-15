# SPDX-License-Identifier: Apache-2.0
"""Few-step solver boundary conditions and request isolation."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.voxcpm2.components.cfm import CfmConfig, UnifiedCFM
from sglang_omni.models.voxcpm2.sampling import VoxCPM2Sampling


class ConstantVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, x, mu, t, cond, dt):
        self.calls.append((t.clone(), dt.clone()))
        return torch.ones_like(x)


@pytest.mark.parametrize("steps", [1, 2, 4, 10])
@pytest.mark.parametrize("sway", [0.0, 1.0])
def test_constant_velocity_integrates_full_interval(steps, sway):
    estimator = ConstantVelocity()
    sampler = UnifiedCFM(2, CfmConfig(), estimator, mean_mode=True)
    mu = torch.zeros(1, 4)
    cond = torch.zeros(1, 2, 3)
    torch.manual_seed(42)
    noise = torch.randn_like(cond)
    torch.manual_seed(42)
    result = sampler(
        mu, steps, 3, cond, sway_sampling_coef=sway, use_cfg_zero_star=False
    )
    torch.testing.assert_close(result, noise - 1)
    assert len(estimator.calls) == steps
    assert all(torch.all(dt > 0) for _, dt in estimator.calls)


def test_default_recipe_preserves_zero_initialization():
    estimator = ConstantVelocity()
    sampler = UnifiedCFM(2, CfmConfig(), estimator)
    cond = torch.zeros(1, 2, 3)
    torch.manual_seed(42)
    noise = torch.randn_like(cond)
    torch.manual_seed(42)
    result = sampler(torch.zeros(1, 4), 10, 3, cond)
    # The default recipe skips the first interval. A constant velocity then
    # integrates over [warped(0.9), 0], independently of internal loop details.
    end_of_first_interval = 0.9 + torch.cos(torch.tensor(torch.pi / 2 * 0.9)) - 1 + 0.9
    torch.testing.assert_close(result, noise - end_of_first_interval)
    assert len(estimator.calls) == 9
    assert all(torch.count_nonzero(dt) == 0 for _, dt in estimator.calls)


def test_invalid_one_step_fails_before_rng_or_estimator():
    estimator = ConstantVelocity()
    sampler = UnifiedCFM(2, CfmConfig(), estimator)
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="skips the only"):
        sampler(torch.zeros(1, 4), 1, 3, torch.zeros(1, 2, 3))
    assert torch.equal(torch.get_rng_state(), rng)
    assert estimator.calls == []


@pytest.mark.parametrize(
    "values",
    [
        {"inference_timesteps": 0},
        {"inference_timesteps": -1},
        {"inference_timesteps": 1.5},
        {"inference_timesteps": True},
        {"cfg_value": float("nan")},
        {"cfg_value": float("inf")},
        {"cfg_value": -1},
        {"sway_sampling_coef": -0.1},
        {"sway_sampling_coef": 1.1},
        {"use_cfg_zero_star": "false"},
    ],
)
def test_invalid_recipe(values):
    with pytest.raises((ValueError, TypeError)):
        VoxCPM2Sampling(**values)


def test_request_precedence_preserves_false_and_zero():
    sampling = VoxCPM2Sampling.from_sources(
        {"inference_timesteps": 1, "use_cfg_zero_star": False, "sway_sampling_coef": 0},
        {"inference_timesteps": 10, "use_cfg_zero_star": True, "cfg_value": 0},
    )
    assert sampling == VoxCPM2Sampling(1, 0, 0, False)


@pytest.mark.parametrize(
    "change",
    [
        {"inference_timesteps": 4},
        {"cfg_value": 1},
        {"sway_sampling_coef": 0},
        {"use_cfg_zero_star": False},
    ],
)
def test_batch_cannot_silently_share_another_requests_recipe(change):
    original = asdict(VoxCPM2Sampling())
    states = [SimpleNamespace(**original), SimpleNamespace(**(original | change))]
    with pytest.raises(ValueError, match=next(iter(change))):
        VoxCPM2Sampling.for_batch(states)


def test_identical_recipes_can_batch():
    recipe = VoxCPM2Sampling(1, 1, 0, False)
    assert VoxCPM2Sampling.for_batch([recipe, recipe]) == recipe


def test_unsupported_solver_is_explicit():
    with pytest.raises(ValueError, match="euler"):
        UnifiedCFM(2, CfmConfig(solver="heun"), ConstantVelocity())
