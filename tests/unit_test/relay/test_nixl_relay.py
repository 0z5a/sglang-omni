# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang_omni.relay import nixl


@pytest.mark.parametrize(
    "failure", [RuntimeError("backend failure"), KeyboardInterrupt()]
)
def test_close_reports_backend_errors_and_preserves_interrupts(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure: BaseException,
) -> None:
    deregister = Mock(side_effect=failure)
    relay = object.__new__(nixl.NixlRelay)
    relay.connection = SimpleNamespace(
        _nixl=SimpleNamespace(deregister_memory=deregister)
    )
    relay.pool_handle = 42
    monkeypatch.setattr(nixl, "NIXL_AVAILABLE", True)

    if isinstance(failure, KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            relay.close()
    else:
        relay.close()
        assert "Failed to deregister NIXL memory pool" in caplog.text
        assert caplog.records[-1].exc_info is not None

    deregister.assert_called_once_with(42)
