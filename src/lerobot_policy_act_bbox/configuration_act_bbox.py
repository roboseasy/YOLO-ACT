"""act_bbox 설정: ACT에 YOLO bbox 입력(observation.environment_state)을 필수로 더한다."""

from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.utils.constants import OBS_ENV_STATE


@PreTrainedConfig.register_subclass("act_bbox")
@dataclass
class ACTBBoxConfig(ACTConfig):
	"""ACT 설정에 bbox 입력 조건과 녹화 때의 bbox·YOLO 설정을 더한다.

	bbox_* 값은 학습 데이터셋에서 채워져 체크포인트에 저장되고, lerobot-rollout-bbox가 기본값으로 쓴다.
	"""

	# bbox는 이미 0~1이라 데이터 통계로 정규화하지 않는다.
	# 검출률이 높으면 is_detected 표준편차가 0에 가까워 미검출 0이 큰 음수로 폭발한다.
	normalization_mapping: dict[str, NormalizationMode] = field(
		default_factory=lambda: {
			"VISUAL": NormalizationMode.MEAN_STD,
			"STATE": NormalizationMode.MEAN_STD,
			"ENV": NormalizationMode.IDENTITY,
			"ACTION": NormalizationMode.MEAN_STD,
		}
	)

	# 학습 데이터셋의 bbox 특징 이름과 녹화 때 YOLO 설정 (make_policy가 채운다)
	bbox_names: list[str] = field(default_factory=list)
	bbox_cameras: list[str] = field(default_factory=list)
	bbox_model: str = ""
	bbox_model_sha256: str = ""
	bbox_classes: list[str] = field(default_factory=list)
	bbox_conf: float = 0.0
	bbox_is_bgr_input: bool = False

	def set_dataset_feature_metadata(self, features: dict[str, dict]) -> None:
		"""make_policy가 학습 데이터셋 특징을 넘길 때 bbox 이름과 YOLO 설정을 복사한다."""
		feature = features.get(OBS_ENV_STATE)
		if feature is None:
			raise ValueError(
				f"act_bbox는 '{OBS_ENV_STATE}'(bbox) 특징이 있는 데이터셋이 필요합니다. "
				"lerobot-record-bbox로 녹화한 데이터셋을 쓰세요."
			)
		self.bbox_names = list(feature.get("names") or [])
		detector = (feature.get("info") or {}).get("bbox_detector") or {}
		self.bbox_cameras = list(detector.get("cameras", []))
		self.bbox_model = detector.get("model", "")
		self.bbox_model_sha256 = detector.get("model_sha256", "")
		self.bbox_classes = list(detector.get("classes", []))
		self.bbox_conf = float(detector.get("conf", 0.0))
		self.bbox_is_bgr_input = bool(detector.get("is_bgr_input", False))

	def validate_features(self) -> None:
		super().validate_features()
		if self.env_state_feature is None:
			raise ValueError(
				f"act_bbox는 입력에 '{OBS_ENV_STATE}'(bbox)가 있어야 합니다. "
				"lerobot-record-bbox로 녹화한 데이터셋을 쓰세요."
			)
		env_dim = self.env_state_feature.shape[0]
		if self.bbox_names and len(self.bbox_names) != env_dim:
			raise ValueError(f"bbox 이름 {len(self.bbox_names)}개와 '{OBS_ENV_STATE}' 차원 {env_dim}이 다릅니다.")
