# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from sglang_omni.models.qwen3_omni.hf_config import Qwen3OmniMoeVisionEncoderConfig


def test_vision_config_defaults_are_not_shared() -> None:
    first = Qwen3OmniMoeVisionEncoderConfig()
    second = Qwen3OmniMoeVisionEncoderConfig()
    first.deepstack_visual_indexes.append(32)

    assert second.deepstack_visual_indexes == [8, 16, 24]


@pytest.mark.parametrize("indexes", [None, [], [4, 8]])
def test_vision_config_preserves_explicit_indexes(indexes: list[int] | None) -> None:
    config = Qwen3OmniMoeVisionEncoderConfig(deepstack_visual_indexes=indexes)

    assert config.deepstack_visual_indexes is indexes
    assert config.to_dict()["deepstack_visual_indexes"] == indexes
