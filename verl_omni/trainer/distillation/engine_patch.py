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

import logging
import os

import torch
from tensordict import NonTensorData

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

__all__ = ["apply_loss_route_patch", "apply_actor_worker_patch"]

_LOSS_ROUTE_APPLIED = False
_ACTOR_WORKER_APPLIED = False


def apply_loss_route_patch():
    global _LOSS_ROUTE_APPLIED
    if _LOSS_ROUTE_APPLIED:
        logger.warning("Loss route patch has already been applied.")
        return

    import verl.trainer.distillation as _vd

    from verl_omni.trainer.distillation.losses import omni_distillation_ppo_loss

    if getattr(_vd, "distillation_ppo_loss", None) is not omni_distillation_ppo_loss:
        _vd.distillation_ppo_loss = omni_distillation_ppo_loss
        logger.info("Patched verl.trainer.distillation.distillation_ppo_loss with omni_distillation_ppo_loss.")

    try:
        import verl.workers.engine_workers as _ve

        if getattr(_ve, "distillation_ppo_loss", None) is not omni_distillation_ppo_loss:
            _ve.distillation_ppo_loss = omni_distillation_ppo_loss
            logger.info("Patched verl.workers.engine_workers.distillation_ppo_loss with omni_distillation_ppo_loss.")
    except ImportError:
        logger.warning("verl.workers.engine_workers module not found. Skipping patch for engine_workers.")

    _LOSS_ROUTE_APPLIED = True


def apply_actor_worker_patch():
    global _ACTOR_WORKER_APPLIED
    if _ACTOR_WORKER_APPLIED:
        logger.warning("Actor worker patch has already been applied.")
        return

    import verl.workers.engine_workers as _ve
    from verl.single_controller.base.decorator import MAGIC_ATTR, Dispatch, register

    _orig_update_actor = _ve.ActorRolloutRefWorker.update_actor

    def _update_actor_with_teacher_unembeds(self, data, *args, **kwargs):
        hidden = getattr(self, "_teacher_unembeds", None)
        if hidden:
            payload = {
                "teacher_unembeds": hidden,
                "teacher_key_to_id": {key: idx for idx, key in enumerate(sorted(hidden))},
            }

            if hasattr(data, "update_extra_info"):
                data.update_extra_info(payload)
            else:
                data["teacher_unembeds"] = NonTensorData(hidden)
                data["teacher_key_to_id"] = NonTensorData(payload["teacher_key_to_id"])

        return _orig_update_actor(self, data, *args, **kwargs)

    _attrs = getattr(_orig_update_actor, MAGIC_ATTR, None)
    if _attrs is not None:
        setattr(_update_actor_with_teacher_unembeds, MAGIC_ATTR, _attrs)
    _ve.ActorRolloutRefWorker.update_actor = _update_actor_with_teacher_unembeds

    def _set_teacher_unembeds(self, teacher_unembeds):
        self._teacher_unembeds = {
            key: W.detach().to(dtype=torch.bfloat16).cpu().contiguous() for key, W in teacher_unembeds.items()
        }
        self._teacher_key_vocab = sorted(teacher_unembeds.keys())

    _ve.ActorRolloutRefWorker.set_teacher_unembeds = register(dispatch_mode=Dispatch.ONE_TO_ALL)(_set_teacher_unembeds)

    _ACTOR_WORKER_APPLIED = True
