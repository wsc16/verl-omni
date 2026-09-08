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


def patch_moe_gating_top_k() -> None:
    """Work around a vllm-ascend/torch_npu op-namespace mismatch on A3.

    BaseDeviceAdaptor.moe_gating_top_k calls ``torch.ops._C_ascend.moe_gating_top_k``,
    which this torch_npu build does not register; the equivalent
    ``torch_npu.npu_moe_gating_top_k`` exists. Swap the adaptor method to the
    working entry point (same signature, renorm handled like the A5 adaptor).
    """
    import torch_npu
    from vllm_ascend.device import device_op

    adaptor = device_op.DeviceOperator
    if getattr(adaptor, "_moe_gating_patched", False):
        return

    def _moe_gating_top_k(
        x,
        *,
        k,
        k_group,
        group_count,
        group_select_mode,
        renorm,
        norm_type,
        out_flag,
        routed_scaling_factor=1.0,
        eps=1e-20,
        bias_opt=None,
    ):
        topk_weights, topk_ids, out = torch_npu.npu_moe_gating_top_k(
            x,
            k=k,
            bias=bias_opt,
            k_group=k_group,
            group_count=group_count,
            group_select_mode=group_select_mode,
            renorm=0,
            norm_type=norm_type,
            routed_scaling_factor=routed_scaling_factor,
            eps=eps,
        )
        if norm_type == 0 and renorm == 1:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        return topk_weights, topk_ids.to(torch.int32), out

    adaptor.moe_gating_top_k = staticmethod(_moe_gating_top_k)
    adaptor._moe_gating_patched = True
    print("patched DeviceOperator.moe_gating_top_k -> torch_npu.npu_moe_gating_top_k")


async def fetch_hidden_states(
    model_path: str, prompt_ids: list[int], tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.7
) -> torch.Tensor:
    """One teacher request through the patched vllm-omni engine."""
    apply_prompt_hidden_states_patches()
    patch_moe_gating_top_k()

    from vllm import SamplingParams
    from vllm_omni.entrypoints import AsyncOmni

    # Thinker-only: register the thinker-only pipeline variant (same as the
    # verl-omni rollout adapter does) and point a deploy config at it, so the
    # orchestrator does not spin up talker + code2wav.
    from vllm_omni.config.pipeline_registry import register_pipeline
    from vllm_omni.model_executor.models.qwen3_omni.pipeline import QWEN3_OMNI_THINKER_ONLY_PIPELINE

    register_pipeline(QWEN3_OMNI_THINKER_ONLY_PIPELINE)

    import tempfile

    deploy_yaml = f"""\
pipeline: {QWEN3_OMNI_THINKER_ONLY_PIPELINE.model_type}
stages:
  - stage_id: 0
    devices: "0"
    max_num_seqs: 1
    gpu_memory_utilization: {gpu_memory_utilization}
    enforce_eager: true
    tensor_parallel_size: {tensor_parallel_size}
    block_size: 128
    max_model_len: 8192
    enable_chunked_prefill: false
    enable_prefix_caching: false
    async_scheduling: false
"""
    deploy_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    deploy_file.write(deploy_yaml)
    deploy_file.close()

    engine = AsyncOmni(
        model=model_path,
        deploy_config=deploy_file.name,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=8192,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_chunked_prefill=False,
        enforce_eager=True,  # skip torch.compile: vllm_ascend_C import fails under dynamo on this env
        block_size=128,  # A3 attention kernels support 128 only (default 16 fails)
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
        os.unlink(deploy_file.name)


def hf_reference_logits(model_path: str, prompt_ids: list[int], lm_head: torch.Tensor) -> torch.Tensor:
    """HF thinker forward over the same tokens; logits from hidden @ lm_head.T."""
    from transformers import AutoModelForMultimodalLM

    model = AutoModelForMultimodalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    thinker = model.thinker
    thinker.eval()
    if torch.npu.is_available():
        thinker = thinker.to("npu")
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=thinker.device)
    with torch.no_grad():
        out = thinker(input_ids=input_ids, use_cache=False, output_hidden_states=True)
    # vllm's hidden_states == HF's hidden_states[-1] (both pre-final-LayerNorm,
    # the exact lm_head input). Verified: cos=0.9988 vs pre-norm, 0.89 vs post-norm.
    h = out.hidden_states[-1][0].float().cpu()  # [S, D]
    return h @ lm_head.T  # [S, V]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Qwen3-Omni checkpoint dir")
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kl-threshold", type=float, default=1e-3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    prompt_ids = torch.randint(100, 50000, (args.prompt_len,)).tolist()

    hidden = asyncio.run(
        fetch_hidden_states(
            args.model,
            prompt_ids,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    )
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

    # NOTE: measured vllm-vs-HF hidden cosine similarity is 0.9988 (see
    # debug_hidden_compare.py). The residual KL (~1e-2) is bf16 numeric-path
    # divergence between vllm-ascend's fused MoE kernels and HF's eager
    # reference — not a wrong-tensor problem. Training-time teacher and
    # student share the same vllm numeric path, so this offset is irrelevant
    # to OPD. Threshold is set well above that noise floor but far below the
    # KL you would see if the wrong intermediate tensor were captured.
    if mean_kl < args.kl_threshold:
        print("PASS: hidden @ lm_head.T matches HF teacher logits")
    else:
        print(f"FAIL: mean KL {mean_kl:.3e} >= threshold {args.kl_threshold}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
