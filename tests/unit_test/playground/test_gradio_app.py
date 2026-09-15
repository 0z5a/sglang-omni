# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import base64
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("gradio")

from playground.gradio import app


@pytest.mark.parametrize("finish", [False, True])
def test_audio_file_lives_until_chat_generator_finishes(
    monkeypatch: pytest.MonkeyPatch, finish: bool
) -> None:
    audio = b"generated audio"

    def chunks(api_base: str, payload: dict[str, object]) -> Iterator[dict[str, str]]:
        yield {"type": "audio", "value": base64.b64encode(audio).decode()}

    monkeypatch.setattr(app, "stream_chat_completion", chunks)
    results = app.make_chat_handler("http://localhost:8000")(
        message="hello",
        chat_display=[],
        api_history=[],
        model="test",
        system_prompt="",
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        output_mode="audio",
        media_files=None,
    )
    try:
        audio_path = next(results)[2]
        assert audio_path is not None
        path = Path(audio_path)
        assert path.read_bytes() == audio
        if finish:
            assert next(results)[2] == audio_path
            assert path.read_bytes() == audio
            with pytest.raises(StopIteration):
                next(results)
    finally:
        results.close()
    assert not path.exists()
