"""act_bbox 처리기: ACT와 같다. bbox 정규화 방식은 ACTBBoxConfig.normalization_mapping이 정한다."""

from typing import Any

import torch
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline

from .configuration_act_bbox import ACTBBoxConfig


def make_act_bbox_pre_post_processors(
	config: ACTBBoxConfig,
	dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
	PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
	PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
	return make_act_pre_post_processors(config, dataset_stats=dataset_stats)
