# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runtime monkey-patches that teach vLLM-Omni's AR path to return the
thinker's last-layer prompt hidden states, for full-vocabulary OPD
(nitrobrew-style teacher signal).

vLLM-Omni already produces the pre-unembedding hidden states inside
``GPUARModelRunner.execute_model`` (``hidden_states`` is the exact tensor fed
to ``model.compute_logits``, minus the logits_indices gather for sampling).
This module reroutes them to the client through the existing
``multimodal_output`` channel, so no new wire fields are needed:

    runner   : prompt hidden per request -> OmniModelRunnerOutput.multimodal_outputs
    scheduler: -> OmniEngineCoreOutput.multimodal_output
    processor: -> OmniRequestState.mm_accumulated (key "prompt_hidden_states")
    client   : -> CompletionOutput.multimodal_output["prompt_hidden_states"]

Opt-in per request: put ``{"return_prompt_hidden_states": True}`` into the
prompt dict's ``model_intermediate_buffer`` (runner-owned request transport,
already deserialized by OmniGPUModelRunner). The verl-omni AR strategy sets
this flag when ``sampling_params["return_prompt_hidden_states"]`` is truthy.

Constraint: hidden states are only produced for tokens scheduled in the
current step and are emitted when the request finishes prefill. Chunked
prefill is not supported: a chunked request would emit partial hidden
tensors that the accumulator would silently concatenate. The patch asserts
the engine is launched with ``enable_chunked_prefill=False`` when the flag
first appears on a request.
"""

import logging

logger = logging.getLogger(__file__)

# Key under which the hidden states ride the multimodal_output channel.
PROMPT_HIDDEN_STATES_KEY = "prompt_hidden_states"
# Flag key in the request's model_intermediate_buffer.
RETURN_FLAG_KEY = "return_prompt_hidden_states"

_applied = False


def request_wants_prompt_hidden_states(runner, req_id: str) -> bool:
    """True when the request opted in via model_intermediate_buffer."""
    info = runner.model_intermediate_buffer.get(req_id)
    return bool(isinstance(info, dict) and info.get(RETURN_FLAG_KEY))


def apply_prompt_hidden_states_patches() -> None:
    """Install the hidden-state output channel patches (idempotent)."""
    global _applied
    if _applied:
        return

    from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

    _patch_runner(GPUARModelRunner)
    _patch_npu_runner()
    _applied = True
    logger.info("vllm_omni prompt-hidden-states patches applied")


def _patch_npu_runner() -> None:
    """Patch the Ascend AR runner, which builds OmniModelRunnerOutput inline.

    NPUARModelRunner.sample_tokens clears ``execute_model_state`` and returns
    either an OmniModelRunnerOutput or an AsyncGPUModelRunnerOutput wrapper.
    We peek the state first, call the original, then attach the per-request
    hidden payloads to the inner model_runner_output.
    """
    try:
        from vllm_omni.platforms.npu.worker.npu_ar_model_runner import NPUARModelRunner
    except ImportError:
        return  # non-Ascend deployment

    original_sample = NPUARModelRunner.sample_tokens

    def _sample_with_prompt_hidden_states(self, grammar_output):
        state = self.execute_model_state
        scheduler_output = state[0] if state is not None else None
        hidden_states = state[4] if state is not None else None

        output = original_sample(self, grammar_output)

        if scheduler_output is None or hidden_states is None:
            return output

        payloads = _collect_prompt_hidden_payloads(self, scheduler_output, hidden_states)
        if not payloads:
            return output

        # Sync path returns OmniModelRunnerOutput directly; async scheduling
        # wraps it in AsyncGPUModelRunnerOutput (private _model_runner_output).
        target = getattr(output, "_model_runner_output", None) or output
        _merge_into_multimodal_outputs(target, payloads)
        return output

    NPUARModelRunner.sample_tokens = _sample_with_prompt_hidden_states


def _patch_runner(runner_cls: type) -> None:
    """Attach hidden-state payloads to OmniModelRunnerOutput.multimodal_outputs."""
    original_build = runner_cls._build_omni_model_runner_output_from_snapshot

    def _build_with_prompt_hidden_states(self, **kwargs):
        output = original_build(self, **kwargs)
        if output is None:
            return output

        scheduler_output = kwargs["scheduler_output"]
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None:
            return output

        hidden_payloads = _collect_prompt_hidden_payloads(self, scheduler_output, hidden_states)
        if not hidden_payloads:
            return output

        _merge_into_multimodal_outputs(output, hidden_payloads)
        return output

    runner_cls._build_omni_model_runner_output_from_snapshot = _build_with_prompt_hidden_states


def _collect_prompt_hidden_payloads(runner, scheduler_output, hidden_states) -> dict[str, dict]:
    """Slice per-request prompt hidden states [S, D] (bf16, CPU) from the step's hidden tensor.

    Only requests that opted in AND are finishing their prefill in this step
    are captured; partial (chunked) prefills raise instead of emitting
    truncated tensors.
    """
    num_scheduled_tokens = scheduler_output.num_scheduled_tokens
    snapshot_qsl = getattr(runner, "_snapshot_query_start_loc_cpu", None)
    if callable(snapshot_qsl):
        query_start_loc_cpu = snapshot_qsl()
    else:  # NPU AR runner keeps a padded query_start_loc buffer
        query_start_loc_cpu = runner.query_start_loc.cpu

    payloads: dict[str, dict] = {}
    for req_id, num_tokens in num_scheduled_tokens.items():
        if not request_wants_prompt_hidden_states(runner, req_id):
            continue
        request = runner.requests.get(req_id)
        if request is None or request.prompt_token_ids is None:
            continue

        num_prompt_tokens = len(request.prompt_token_ids)
        start_idx = request.num_computed_tokens
        num_tokens = int(num_tokens)

        num_remaining = num_prompt_tokens - start_idx - num_tokens
        if num_remaining > 0:
            raise RuntimeError(
                "return_prompt_hidden_states does not support chunked prefill "
                f"(req={req_id}: {num_remaining} prompt tokens left unscheduled). "
                "Launch the teacher engine with enable_chunked_prefill=False."
            )

        req_idx = runner.input_batch.req_id_to_index.get(req_id)
        if req_idx is None:
            continue
        offset = int(query_start_loc_cpu[req_idx])
        h = hidden_states[offset : offset + num_tokens]
        payloads[req_id] = {PROMPT_HIDDEN_STATES_KEY: h.detach().cpu()}

    return payloads


def _merge_into_multimodal_outputs(output, hidden_payloads: dict[str, dict]) -> None:
    """Merge hidden payloads into the per-request multimodal_outputs list."""
    req_id_to_index = output.req_id_to_index
    num_reqs = len(output.req_ids)

    mm: list = list(output.multimodal_outputs) if output.multimodal_outputs else [None] * num_reqs
    while len(mm) < num_reqs:
        mm.append(None)

    for req_id, payload in hidden_payloads.items():
        idx = req_id_to_index.get(req_id)
        if idx is None or idx >= num_reqs:
            continue
        entry = mm[idx]
        if entry is None:
            mm[idx] = payload
        elif isinstance(entry, dict):
            entry = dict(entry)
            entry.update(payload)
            mm[idx] = entry

    output.multimodal_outputs = mm


def extract_prompt_hidden_states(req_output) -> object:
    """Read the hidden states off a client-facing OmniRequestOutput.

    Looks at the first completion's ``multimodal_output`` payload (a
    MultimodalPayload or dict). Returns the tensor or None.
    """
    outputs = getattr(req_output, "outputs", None)
    if not outputs:
        return None
    mm = getattr(outputs[0], "multimodal_output", None)
    if mm is None:
        return None
    try:
        return mm.get(PROMPT_HIDDEN_STATES_KEY)
    except AttributeError:
        if isinstance(mm, dict):
            return mm.get(PROMPT_HIDDEN_STATES_KEY)
        return None
