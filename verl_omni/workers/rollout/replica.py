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
import logging as _lg

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict
from verl.workers.rollout.replica import RolloutReplicaRegistry


def _pHs_patch_verb_vllm_replica_init():
    """Probe: which replica class does each LLM server actually instantiate.

    Both verl_omni's ``vLLMOmniReplica`` (inherits) and verb's plain
    ``vLLMReplica`` go through ``vLLMReplica.__init__``, so logging
    ``type(self).__name__`` here distinguishes omni vs verb server.
    """
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica

    orig = vLLMReplica.__init__

    def patched_init(replica_self, replica_rank, config, model_config, *a, **k):
        import os

        _lg.warning(
            "[pHs-debug] Replica.__init__ pid=%d cls=%s name=%s teacher=%s",
            os.getpid(),
            type(replica_self).__name__,
            getattr(config, "name", None),
            k.get("is_teacher_model", False),
        )
        return orig(replica_self, replica_rank, config, model_config, *a, **k)

    vLLMReplica.__init__ = patched_init


_pHs_patch_verb_vllm_replica_init()


class DiffusionOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    diffusion_output: Any
    """Generated uint8 pixel tensor (CHW/TCHW) in [0, 255], or floating-point latents."""
    log_probs: Optional[Any] = None
    """logprobs of generated image/video"""
    stop_reason: Optional[str] = None
    """stop reason: 'completed', 'aborted', or None for unknown"""
    num_preempted: Optional[int] = None
    """number of preempted times for metric calculation"""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


def _load_vllm_omni():
    from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniReplica

    return vLLMOmniReplica


RolloutReplicaRegistry.register("vllm_omni", _load_vllm_omni)
