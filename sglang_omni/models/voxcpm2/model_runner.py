# SPDX-License-Identifier: Apache-2.0
"""VoxCPM2 AR runner: drives the diffusion half of every decode step."""

from __future__ import annotations

from typing import Any

import torch
from sglang.srt.managers.schedule_batch import FINISH_MATCHED_TOKEN

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.voxcpm2.request_builders import VoxCPM2SGLangRequestData
from sglang_omni.models.voxcpm2.sampling import VoxCPM2Sampling


class VoxCPM2ModelRunner(ModelRunner):
    """Runs the local DiT after each AR forward and feeds the result back."""

    def before_prefill(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        embeddings, audio_masks = [], []
        parameter = next(self.model.parameters())
        for request in requests:
            data = request.data
            prefill = data.prefill
            token_ids = prefill.text_token.to(parameter.device, dtype=torch.long)
            features = prefill.audio_feat.to(parameter.device, dtype=parameter.dtype)
            text_mask = prefill.text_mask.to(parameter.device).bool()
            audio_mask = prefill.audio_mask.to(parameter.device).bool()
            text_embed = self.model.embed_tokens(token_ids)
            config = self.model.config.lm_config
            if getattr(config, "use_mup", False):
                text_embed = text_embed * config.scale_emb
            feature_embed = self.model.projections.enc_to_lm_proj(
                self.model.feat_encoder(features.unsqueeze(0)).squeeze(0)
            )
            combined = text_embed * text_mask.unsqueeze(
                -1
            ) + feature_embed * audio_mask.unsqueeze(-1)
            span = data.req.extend_range
            embeddings.append(combined[span.start : span.end])
            audio_masks.append(audio_mask[span.start : span.end])
            data.cond = features[-1:].contiguous()
            if data.state.seed is not None:
                torch.manual_seed(int(data.state.seed))
        forward_batch.input_embeds = torch.cat(embeddings)
        # EagerRunner copies ForwardBatch through dataclasses.replace, which
        # drops dynamic attributes. Keep this synchronous forward's mask on
        # the model alongside its existing hidden-state/feedback buffers.
        self.model.prefill_audio_mask = torch.cat(audio_masks)

    def cleanup_prefill(self, forward_batch, schedule_batch, requests) -> None:
        self.model.prefill_audio_mask = None

    def requested_capture_hidden_mode_prefill(
        self, schedule_batch: Any, requests: list
    ) -> Any:
        del schedule_batch, requests
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        return CaptureHiddenMode.FULL

    def requested_capture_hidden_mode_decode(
        self, schedule_batch: Any, requests: list
    ) -> Any:
        del schedule_batch, requests
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        if self.model.graph_feedback_buffer is not None:
            return CaptureHiddenMode.FULL
        return CaptureHiddenMode.LAST

    def post_prefill(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        del result, forward_batch
        if bool(getattr(schedule_batch, "is_prefill_only", False)) or not requests:
            return
        self._advance(requests, rows=self._prefill_rows(requests), is_prefill=True)

    def post_decode(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        del result, forward_batch, schedule_batch
        if requests:
            self._advance(requests, rows=None, is_prefill=False)

    @staticmethod
    def _prefill_rows(requests: list) -> torch.Tensor:
        """Index of each request's final prompt position in the packed batch."""
        indices: list[int] = []
        offset = 0
        for request in requests:
            offset += int(request.data.req.extend_range.length)
            indices.append(offset - 1)
        return torch.tensor(indices, dtype=torch.long)

    def _advance(
        self, requests: list, *, rows: torch.Tensor | None, is_prefill: bool
    ) -> None:
        """Sample one latent patch per request and stage the next step's input."""
        rows_data = [request.data for request in requests]
        sampling = _shared_sampling(rows_data)

        patches, embeddings = self.model.decode_patch(
            self._batch_cond(rows_data),
            inference_timesteps=sampling.inference_timesteps,
            cfg_value=sampling.cfg_value,
            sway_sampling_coef=sampling.sway_sampling_coef,
            use_cfg_zero_star=sampling.use_cfg_zero_star,
            rows=rows,
        )
        stop_flags = self.model.stop_flags(rows)

        for index, data in enumerate(rows_data):
            patch = patches[index : index + 1]
            data.cond = patch
            data.latent_patches.append(patch.squeeze(0).detach().cpu())
            if is_prefill:
                continue
            state = data.state
            steps = len(data.latent_patches)
            if steps > state.min_len and bool(stop_flags[index]):
                data.finish_reason = "stop"
            elif steps >= state.max_len:
                data.finish_reason = "length"
            if data.finish_reason is not None:
                data.req.finished_reason = FINISH_MATCHED_TOKEN(0)

        if self.model.graph_feedback_buffer is not None:
            self.model.write_feedback(embeddings)

    def _batch_cond(self, rows_data: list[VoxCPM2SGLangRequestData]) -> torch.Tensor:
        """Stack each request's previous patch into the DiT's condition batch."""
        parameter = next(self.model.parameters())
        zeros = torch.zeros(
            (1, self.model.patch_size, self.model.feat_dim),
            device=parameter.device,
            dtype=parameter.dtype,
        )
        return torch.cat(
            [
                zeros if data.cond is None else data.cond.to(parameter.device)
                for data in rows_data
            ],
            dim=0,
        )


def _shared_sampling(rows_data: list[VoxCPM2SGLangRequestData]) -> VoxCPM2Sampling:
    return VoxCPM2Sampling.for_batch([data.state for data in rows_data])


__all__ = ["VoxCPM2ModelRunner"]
