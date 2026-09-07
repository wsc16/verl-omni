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
"""NPU verification: teacher hidden states from vllm-omni match HF teacher logits.

End-to-end check of the nitrobrew premise for Qwen3-Omni thinker:

    1. Start a vllm-omni AsyncOmni teacher (AR / thinker-only pipeline).
    2. Send a request with ``return_prompt_hidden_states`` enabled.
    3. Load the thinker's ``lm_head.weight`` from safetensors.
    4. Compare ``softmax(hidden @ W.T)`` against a plain HF thinker forward
       pass over the same tokens (full-vocab KL per position).

Run inside the NPU training env (needs the model weights + vllm-omni):

    python examples/gspo_trainer/qwen3_omni/verify_teacher_hidden_alignment.py \
        --model /data2/model/Qwen3-Omni-30B-A3B-Instruct

Pass criteria: mean per-position KL < 1e-3 (bf16 matmul noise budget;
the reference uses fp32).
"""

import argparse
import asyncio
import json
import os

import torch

from verl_omni.workers.rollout.vllm_rollout.prompt_hidden_states import (
    PROMPT_HIDDEN_STATES_KEY,
    RETURN_FLAG_KEY,
    apply_prompt_hidden_states_patches,
    extract_prompt_hidden_states,
)


def load_lm_head_weight(model_path: str) -> torch.Tensor:
    """Read thinker.lm_head.weight [V, D] (fp32, CPU) from safetensors."""
    from safetensors.torch import load_file

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    for key in ("thinker.lm_head.weight", "lm_head.weight"):
        shard = weight_map.get(key)
        if shard is None:
            continue
        tensors = load_file(os.path.join(model_path, shard))
        if key in tensors:
            return tensors[key].float()
    raise ValueError(f"No lm_head weight found under {model_path}")


async def fetch_hidden_states(model_path: str, prompt_ids: list[int]) -> torch.Tensor:
    """One teacher request through the patched vllm-omni engine."""
    apply_prompt_hidden_states_patches()

    from vllm import SamplingParams
    from vllm_omni.entrypoints import AsyncOmni

    engine = AsyncOmni(
        model=model_path,
        deploy_config=None,  # thinker-only topology resolved from the HF config
        tensor_parallel_size=1,
        max_model_len=8192,
        gpu_memory_utilization=0.7,
        enable_chunked_prefill=False,
        disable_log_stats=True,
    )
    try:
        prompt = {
            "prompt_token_ids": prompt_ids,
            "model_intermediate_buffer": {RETURN_FLAG_KEY: True},
        }
        params = SamplingParams(max_tokens=1, temperature=1.0)
        final = None
        async for out in engine.generate(prompt=prompt, sampling_params_list=params, request_id="verify-0"):
            final = out
        assert final is not None
        hidden = extract_prompt_hidden_states(final)
        assert hidden is not None, (
            f"No {PROMPT_HIDDEN_STATES_KEY} in output; mm channel did not carry the payload"
        )
        return hidden.float()
    finally:
        engine.shutdown()


def hf_reference_logits(model_path: str, prompt_ids: list[int], lm_head: torch.Tensor) -> torch.Tensor:
    """HF thinker forward over the same tokens; logits from hidden @ lm_head.T."""
    from transformers import AutoModelForMultimodalLM

    model = AutoModelForMultimodalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device="npu" if torch.npu.is_available() else "cpu"
    )
    thinker = model.thinker
    thinker.eval()
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
    with torch.no_grad():
        out = thinker(input_ids=input_ids, use_cache=False, output_hidden_states=True)
    h = out.hidden_states[-1][0].float()  # [S, D]
    return h @ lm_head.T  # [S, V]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Qwen3-Omni checkpoint dir")
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kl-threshold", type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    prompt_ids = torch.randint(100, 50000, (args.prompt_len,)).tolist()

    hidden = asyncio.run(fetch_hidden_states(args.model, prompt_ids))
    print(f"hidden states: shape={tuple(hidden.shape)} dtype={hidden.dtype}")

    lm_head = load_lm_head_weight(args.model)
    print(f"lm_head: shape={tuple(lm_head.shape)}")

    assert hidden.shape[0] == len(prompt_ids), (
        f"hidden rows {hidden.shape[0]} != prompt len {len(prompt_ids)}"
    )
    assert hidden.shape[1] == lm_head.shape[1], (
        f"hidden dim {hidden.shape[1]} != lm_head dim {lm_head.shape[1]}"
    )

    teacher_logits = hidden @ lm_head.T  # [S, V]
    ref_logits = hf_reference_logits(args.model, prompt_ids, lm_head)

    p = torch.log_softmax(teacher_logits, dim=-1)
    q = torch.log_softmax(ref_logits, dim=-1)
    kl = (p.exp() * (p - q)).sum(dim=-1)  # [S]

    mean_kl = kl.mean().item()
    max_kl = kl.max().item()
    print(f"per-position KL(teacher || hf_ref): mean={mean_kl:.3e} max={max_kl:.3e}")

    if mean_kl < args.kl_threshold:
        print("PASS: hidden @ lm_head.T matches HF teacher logits")
    else:
        print(f"FAIL: mean KL {mean_kl:.3e} >= threshold {args.kl_threshold}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
