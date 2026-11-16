"""act_bbox 정책: ACT의 환경 상태 입력층을 bbox 전용 MLP로 바꾼다."""

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import OBS_ENV_STATE
from torch import Tensor, nn

from .configuration_act_bbox import ACTBBoxConfig


class BBoxEncoder(nn.Module):
	"""bbox(0~1) 벡터를 ACT 인코더 토큰 1개로 바꾼다. 2x-1로 [-1, 1]에 맞춘 뒤 2층 MLP를 거친다."""

	def __init__(self, in_dim: int, dim_model: int) -> None:
		super().__init__()
		self.net = nn.Sequential(nn.Linear(in_dim, dim_model), nn.GELU(), nn.Linear(dim_model, dim_model))

	def forward(self, x: Tensor) -> Tensor:
		return self.net(x * 2.0 - 1.0)


class ACTBBoxPolicy(ACTPolicy):
	"""영상·관절·YOLO bbox를 받는 ACT. 영상 없이 관절·bbox만으로도 학습할 수 있다."""

	config_class = ACTBBoxConfig
	name = "act_bbox"

	def __init__(self, config: ACTBBoxConfig, **kwargs) -> None:
		super().__init__(config, **kwargs)
		self.model.encoder_env_state_input_proj = BBoxEncoder(config.env_state_feature.shape[0], config.dim_model)

	def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
		if OBS_ENV_STATE not in batch:
			raise KeyError(
				f"act_bbox 입력에 '{OBS_ENV_STATE}'(bbox)가 없습니다. 이 정책은 lerobot-rollout-bbox로 실행하세요."
			)
		return super().predict_action_chunk(batch)
