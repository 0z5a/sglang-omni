# SPDX-License-Identifier: Apache-2.0
"""VoxCPM2 preprocessing: prompt text assembly and request parameter resolution."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang_omni.models.voxcpm2 import constants as C
from sglang_omni.models.voxcpm2.hf_config import VoxCPM2RuntimeConfig
from sglang_omni.models.voxcpm2.payload_types import VoxCPM2State
from sglang_omni.models.voxcpm2.sampling import VoxCPM2Sampling
from sglang_omni.preprocessing.cache_key import hash_bytes
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.pipeline_state import load_state, store_state
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData
from sglang_omni.utils.audio_payload import audio_data_uri_from_reference


@dataclass
class VoxCPM2PreprocessingContext:
    config: VoxCPM2RuntimeConfig
    tokenizer: Any
    multichar_chinese_tokens: set[str] = field(init=False)

    def __post_init__(self) -> None:
        self.multichar_chinese_tokens = {
            token
            for token in self.tokenizer.get_vocab()
            if len(token) >= 2 and all("\u4e00" <= char <= "\u9fff" for char in token)
        }

    def encode_text(self, text: str) -> list[int]:
        # Match VoxCPM's character-level Chinese input and omit tokenizer-added
        # BOS/EOS tokens. Mixed-language and non-Chinese pieces stay intact.
        pieces = []
        for token in self.tokenizer.tokenize(text):
            clean = token.replace("▁", "")
            if clean in self.multichar_chinese_tokens:
                pieces.extend(clean)
            else:
                pieces.append(token)
        return self.tokenizer.convert_tokens_to_ids(pieces)


_CONTEXT: VoxCPM2PreprocessingContext | None = None


def set_voxcpm2_preprocessing_context(context: VoxCPM2PreprocessingContext) -> None:
    global _CONTEXT
    _CONTEXT = context


def _get_context() -> VoxCPM2PreprocessingContext:
    if _CONTEXT is None:
        raise RuntimeError("VoxCPM2 preprocessing context is not initialized")
    return _CONTEXT


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _first(*values: Any, default: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def _reference_source(reference: dict[str, Any]) -> str | None:
    for key in ("audio_path", "path", "url"):
        value = reference.get(key)
        if value:
            return str(value)
    return audio_data_uri_from_reference(reference)


def build_voxcpm2_state(
    payload: StagePayload, context: VoxCPM2PreprocessingContext
) -> VoxCPM2State:
    """Build the VoxCPM2 state from an incoming request."""
    raw_inputs = payload.request.inputs
    inputs = {"text": raw_inputs} if isinstance(raw_inputs, str) else _dict(raw_inputs)
    params = _dict(payload.request.params)
    tts_params = _dict(_dict(payload.request.metadata).get("tts_params"))
    engine_params = _dict(_dict(params.get("stage_params")).get("tts_engine"))

    target_text = str(
        _first(inputs.get("input"), inputs.get("text"), default="")
    ).strip()
    if not target_text:
        raise ValueError("VoxCPM2 requires nonempty input text")

    references = inputs.get("references")
    if isinstance(references, list) and len(references) > 1:
        raise ValueError("VoxCPM2 accepts at most one reference audio")
    reference = (
        references[0]
        if isinstance(references, list)
        and references
        and isinstance(references[0], dict)
        else {}
    )
    source = _reference_source(reference)
    reference_text = str(
        reference.get("text") or tts_params.get("ref_text") or ""
    ).strip()

    # note (Xinhao Tan): upstream picks the cloning mode based on whether the
    # reference audio has a transcript. With one, the model continues from the
    # reference audio; without one, it only copies the voice.
    # So we check for a transcript instead of making the caller pick a mode.
    prompt_audio = source if (source and reference_text) else ""
    reference_audio = source if (source and not reference_text) else ""
    prompt_text = reference_text if prompt_audio else ""

    text = prompt_text + target_text if prompt_text else target_text
    tokenizer = context.tokenizer
    audio_start_id = int(tokenizer.convert_tokens_to_ids(C.AUDIO_START_TOKEN))
    text_ids = context.encode_text(text) + [audio_start_id]

    config = context.config
    sampling = VoxCPM2Sampling.from_sources(engine_params, tts_params, params)
    return VoxCPM2State(
        sample_rate=config.sample_rate,
        out_sample_rate=config.out_sample_rate,
        prompt_text=prompt_text,
        prompt_audio=prompt_audio,
        reference_audio=reference_audio,
        text_token=torch.tensor(text_ids, dtype=torch.int32),
        target_text_length=len(context.encode_text(target_text)),
        patch_size=config.patch_size,
        feat_dim=config.feat_dim,
        inference_timesteps=sampling.inference_timesteps,
        cfg_value=sampling.cfg_value,
        sway_sampling_coef=sampling.sway_sampling_coef,
        use_cfg_zero_star=sampling.use_cfg_zero_star,
        min_len=int(_first(engine_params.get("min_len"), default=C.DEFAULT_MIN_LEN)),
        max_len=int(
            _first(
                engine_params.get("max_len"),
                tts_params.get("max_len"),
                params.get("max_len"),
                default=C.DEFAULT_MAX_LEN,
            )
        ),
        seed=_first(tts_params.get("seed"), params.get("seed"), default=None),
        stream=bool(params.get("stream")),
    )


@dataclass
class VoxCPM2PrefillInputs:
    """The full AR prefix: tokens, latent patches, and the two span masks."""

    text_token: torch.Tensor
    audio_feat: torch.Tensor
    text_mask: torch.Tensor
    audio_mask: torch.Tensor


def _ref_prefix(
    ref_latents: torch.Tensor, *, start_id: int, end_id: int, feat_dim: int
) -> VoxCPM2PrefillInputs:
    """Wrap reference latents in their start/end tokens, padded at both edges."""
    length = ref_latents.shape[0]
    patch_size = ref_latents.shape[1]
    pad = torch.zeros((1, patch_size, feat_dim), dtype=ref_latents.dtype)
    return VoxCPM2PrefillInputs(
        text_token=torch.cat(
            [
                torch.tensor([start_id], dtype=torch.int32),
                torch.zeros(length, dtype=torch.int32),
                torch.tensor([end_id], dtype=torch.int32),
            ]
        ),
        audio_feat=torch.cat([pad, ref_latents, pad], dim=0),
        text_mask=torch.cat(
            [
                torch.tensor([1], dtype=torch.int32),
                torch.zeros(length, dtype=torch.int32),
                torch.tensor([1], dtype=torch.int32),
            ]
        ),
        audio_mask=torch.cat(
            [
                torch.tensor([0], dtype=torch.int32),
                torch.ones(length, dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
            ]
        ),
    )


def build_prefill_inputs(
    state: VoxCPM2State, *, tokenizer: Any, patch_size: int, feat_dim: int
) -> VoxCPM2PrefillInputs:
    """Lay out the AR prefix: reference prefix, then text, then prompt audio.

    The masks say which positions the base stack reads as text embeddings and
    which it reads as latent patches; every position carries exactly one.
    """
    text_token = torch.as_tensor(state.text_token, dtype=torch.int32)
    text_length = int(text_token.shape[0])
    empty_feat = torch.zeros((text_length, patch_size, feat_dim), dtype=torch.float32)

    tokens = [text_token]
    feats = [empty_feat]
    text_masks = [torch.ones(text_length, dtype=torch.int32)]
    audio_masks = [torch.zeros(text_length, dtype=torch.int32)]

    if state.ref_latents is not None:
        prefix = _ref_prefix(
            torch.as_tensor(state.ref_latents),
            start_id=int(tokenizer.convert_tokens_to_ids(C.AUDIO_PROMPT_START_TOKEN)),
            end_id=int(tokenizer.convert_tokens_to_ids(C.AUDIO_PROMPT_END_TOKEN)),
            feat_dim=feat_dim,
        )
        tokens.insert(0, prefix.text_token)
        feats.insert(0, prefix.audio_feat)
        text_masks.insert(0, prefix.text_mask)
        audio_masks.insert(0, prefix.audio_mask)

    if state.prompt_latents is not None:
        prompt = torch.as_tensor(state.prompt_latents)
        prompt_length = int(prompt.shape[0])
        tokens.append(torch.zeros(prompt_length, dtype=torch.int32))
        feats.append(prompt)
        text_masks.append(torch.zeros(prompt_length, dtype=torch.int32))
        audio_masks.append(torch.ones(prompt_length, dtype=torch.int32))

    return VoxCPM2PrefillInputs(
        text_token=torch.cat(tokens),
        audio_feat=torch.cat(feats, dim=0),
        text_mask=torch.cat(text_masks),
        audio_mask=torch.cat(audio_masks),
    )


@dataclass(kw_only=True)
class VoxCPM2SGLangRequestData(SGLangARRequestData):
    """Per-request engine data the scheduler hangs off the sglang request."""

    state: VoxCPM2State
    prefill: VoxCPM2PrefillInputs
    cond: Any = None
    latent_patches: list[Any] = field(default_factory=list)
    engine_start_s: float = field(default_factory=time.perf_counter)


def audio_prefix_fingerprint(prefill: VoxCPM2PrefillInputs) -> str | None:
    """Stable hash of the latent positions, for use as the radix ``extra_key``.

    Audio positions carry token id 0 and the real content rides in the latent
    patches, so two requests with different reference audio produce identical
    input ids. Without this key they would share a KV prefix and one caller's
    voice would surface in another's audio. None when there is no audio, so
    zero-shot requests still share one subtree.
    """
    if not bool(prefill.audio_mask.any()):
        return None
    latents = prefill.audio_feat[prefill.audio_mask.bool()]
    return hash_bytes(latents.detach().to(torch.float32).cpu().numpy().tobytes())


def build_sglang_voxcpm2_request(
    payload: StagePayload,
    *,
    tokenizer: Any,
    patch_size: int,
    feat_dim: int,
    vocab_size: int,
) -> VoxCPM2SGLangRequestData:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    state = load_state(payload, VoxCPM2State)
    prefill = build_prefill_inputs(
        state, tokenizer=tokenizer, patch_size=patch_size, feat_dim=feat_dim
    )

    sampling_params = SamplingParams(
        max_new_tokens=int(state.max_len), temperature=0.0, stop_token_ids=[]
    )
    sampling_params.normalize(None)
    sampling_params.verify(int(vocab_size))

    req = Req(
        rid=payload.request_id,
        origin_input_text="",
        origin_input_ids=prefill.text_token.tolist(),
        sampling_params=sampling_params,
        eos_token_ids=set(),
        vocab_size=int(vocab_size),
        extra_key=audio_prefix_fingerprint(prefill),
    )
    req.tokenizer = None
    req._input_embeds_are_projected = True
    req._codec_suppress_tokens = None

    return VoxCPM2SGLangRequestData(
        stage_payload=payload,
        state=state,
        prefill=prefill,
        req=req,
        input_embeds_are_projected=True,
    )


def build_stream_output(
    request_id: str, data: VoxCPM2SGLangRequestData, req_output: Any
) -> list[OutgoingMessage]:
    """One streamed chunk: the patch sampled by the step that just finished."""
    if not data.state.stream or not data.latent_patches:
        return []
    return [
        OutgoingMessage(
            request_id=request_id,
            type="stream",
            data={"patch": data.latent_patches[-1]},
        )
    ]


def apply_voxcpm2_result(data: VoxCPM2SGLangRequestData) -> StagePayload:
    """Fold the sampled patches into the payload the vocoder stage reads."""
    if not data.latent_patches:
        raise RuntimeError("VoxCPM2 generated no latent patches")
    state = data.state
    patches = torch.stack(data.latent_patches, dim=0)
    state.generated_latents = patches.permute(2, 0, 1).reshape(patches.shape[2], -1)
    state.completion_tokens = len(data.latent_patches)
    state.prompt_tokens = int(data.prefill.text_token.numel())
    state.engine_time_s = time.perf_counter() - data.engine_start_s
    state.finish_reason = data.finish_reason
    return store_state(data.stage_payload, state)


def preprocess_voxcpm2_payload(payload: StagePayload) -> StagePayload:
    """Preprocessing-stage entry point: validate the request and tokenize text."""
    state = build_voxcpm2_state(payload, _get_context())
    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=state.to_dict(),
    )


__all__ = [
    "VoxCPM2PreprocessingContext",
    "build_voxcpm2_state",
    "preprocess_voxcpm2_payload",
    "set_voxcpm2_preprocessing_context",
]
