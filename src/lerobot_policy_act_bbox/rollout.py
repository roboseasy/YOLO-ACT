"""lerobot-rollout-bbox: lerobot-rollout과 옵션이 같고, act_bbox 정책에 YOLO bbox를 넣어 실행한다.

YOLO 설정(가중치, 카메라 순서, 클래스, conf, 색 순서)은 정책에 저장된 녹화 때 값을 기본으로 쓴다.
--bbox.* 로 바꿀 수 있지만 학습 때와 다르면 경고한다. 카메라 순서가 다르면 정책 입력이 어긋나므로 멈춘다.

예:
	lerobot-rollout-bbox --strategy.type=base \\
		--policy.path=outputs/train/act_bbox/checkpoints/last/pretrained_model \\
		--robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=follower \\
		--robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \\
		--task="Pick up the box" --duration=60
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lerobot.configs import parser
from lerobot.rollout import context as rollout_context
from lerobot.rollout.configs import RolloutConfig
from lerobot.rollout.inference import RTCInferenceConfig
from lerobot.scripts import lerobot_rollout
from lerobot.utils.constants import HF_LEROBOT_HOME, OBS_ENV_STATE
from lerobot.utils.import_utils import register_third_party_plugins

from .bbox import (
	BBoxConfig,
	YoloBBoxStep,
	bbox_dataset_feature,
	bbox_feature_names,
	file_sha256,
	load_detector,
	make_observation_pipeline,
	measure_latency_ms,
	realtime_problem,
	require_robot_cameras,
)
from .configuration_act_bbox import ACTBBoxConfig
from .lerobot_hooks import with_dataset_feature, with_rollout_context_hook


@dataclass
class RolloutBBoxConfig(RolloutConfig):
	bbox: BBoxConfig = field(default_factory=BBoxConfig)


def read_training_detector(pretrained_path: Any) -> dict | None:
	"""체크포인트를 학습한 데이터셋의 YOLO 기록(info.bbox_detector). 로컬에 없으면 None (내려받지 않는다).

	클래스별 conf(class_conf)는 데이터셋에만 있다. 체크포인트 설정은 바꾸지 않는다: LeRobot은 config.json의
	모르는 키를 거절하므로, 키를 더하면 새 체크포인트를 이전 버전 플러그인이 읽지 못한다.
	"""
	if not pretrained_path or not Path(pretrained_path).is_dir():
		return None
	try:
		train_cfg = json.loads((Path(pretrained_path) / "train_config.json").read_text())
		dataset = train_cfg.get("dataset") or {}
		root = Path(dataset["root"]) if dataset.get("root") else HF_LEROBOT_HOME / str(dataset.get("repo_id") or "")
		info = json.loads((root / "meta" / "info.json").read_text())
	except (OSError, ValueError, KeyError, TypeError, AttributeError):
		return None
	feature = (info.get("features") or {}).get(OBS_ENV_STATE) if isinstance(info, dict) else None
	detector = ((feature or {}).get("info") or {}).get("bbox_detector") if isinstance(feature, dict) else None
	return detector if isinstance(detector, dict) and detector else None


def trusted_class_conf(policy_cfg: ACTBBoxConfig, recorded: dict | None) -> list[float] | None:
	"""데이터셋의 class_conf를 쓸 수 있으면 돌려준다. 같은 이름으로 다시 녹화한 데이터셋일 수 있어,
	클래스 · 카메라 · 가중치 sha256 · conf가 체크포인트와 모두 같을 때만 믿는다 (Physical Labs와 같은 규칙)."""
	if not recorded:
		return None
	class_conf = recorded.get("class_conf")
	same = (
		list(recorded.get("classes") or []) == list(policy_cfg.bbox_classes)
		and list(recorded.get("cameras") or []) == list(policy_cfg.bbox_cameras)
		and str(recorded.get("model_sha256") or "") == str(policy_cfg.bbox_model_sha256 or "")
		and abs(float(recorded.get("conf") or 0.0) - float(policy_cfg.bbox_conf or 0.0)) < 1e-9
	)
	if not same or not isinstance(class_conf, list) or len(class_conf) != len(policy_cfg.bbox_classes):
		return None
	return [float(value) for value in class_conf]


def resolve_rollout_bbox(
	policy_cfg: ACTBBoxConfig, override: BBoxConfig, recorded: dict | None = None
) -> tuple[BBoxConfig, list[str]]:
	"""정책에 저장된 녹화 때 YOLO 설정을 기본으로 명령줄 값을 덮어 쓴다. 학습 때와 다른 값은 경고로 돌려준다.

	recorded는 학습 데이터셋의 YOLO 기록이다 (read_training_detector). 클래스별 conf는 여기서만 얻는다.
	"""
	stored = {
		"cameras": list(policy_cfg.bbox_cameras) or None,
		"classes": list(policy_cfg.bbox_classes) or None,
		"conf": policy_cfg.bbox_conf or None,
		"is_bgr_input": policy_cfg.bbox_is_bgr_input,
	}
	warnings = []
	resolved = {}
	for name, stored_value in stored.items():
		override_value = getattr(override, name)
		if override_value is not None and stored_value is not None and override_value != stored_value:
			warnings.append(f"--bbox.{name}={override_value} 이(가) 학습 데이터를 녹화할 때 값 {stored_value} 와 다릅니다.")
		resolved[name] = stored_value if override_value is None else override_value
	model = override.model or policy_cfg.bbox_model or None
	stored_class_conf = trusted_class_conf(policy_cfg, recorded)
	if override.class_conf is not None:
		class_conf = list(override.class_conf)
		if stored_class_conf is not None and class_conf != stored_class_conf:
			warnings.append(
				f"--bbox.class_conf={class_conf} 이(가) 학습 데이터를 녹화할 때 값 {stored_class_conf} 와 다릅니다."
			)
	elif override.conf is not None or resolved["classes"] != list(policy_cfg.bbox_classes):
		class_conf = None  # 명령줄로 conf나 클래스를 바꿨다 — 그 값을 모든 클래스에 쓴다
	else:
		class_conf = stored_class_conf
	bbox_cfg = BBoxConfig(model=model, device=override.device, class_conf=class_conf, **resolved)

	if not bbox_cfg.model:
		raise ValueError("이 정책에는 YOLO 가중치 경로가 저장돼 있지 않습니다. --bbox.model 로 지정하세요.")
	if not bbox_cfg.cameras:
		raise ValueError("이 정책에는 bbox 카메라가 저장돼 있지 않습니다. --bbox.cameras 로 지정하세요.")
	if not bbox_cfg.classes:
		raise ValueError("이 정책에는 bbox 클래스가 저장돼 있지 않습니다. --bbox.classes 로 지정하세요.")
	names = bbox_feature_names(bbox_cfg.cameras, bbox_cfg.classes)
	if policy_cfg.bbox_names and names != list(policy_cfg.bbox_names):
		raise ValueError(
			f"카메라 {bbox_cfg.cameras}, 클래스 {bbox_cfg.classes} 로 만든 bbox 이름이 정책이 학습한 이름과 다릅니다."
		)
	model_path = os.path.abspath(os.path.expanduser(bbox_cfg.model))
	if (
		policy_cfg.bbox_model_sha256
		and os.path.isfile(model_path)
		and file_sha256(model_path) != policy_cfg.bbox_model_sha256
	):
		warnings.append(f"YOLO 가중치 {model_path} 가 학습 데이터를 녹화할 때와 다른 파일입니다(sha256 불일치).")
	return bbox_cfg, warnings


@parser.wrap()
def rollout_bbox(cfg: RolloutBBoxConfig) -> None:
	if not isinstance(cfg.policy, ACTBBoxConfig):
		raise ValueError("lerobot-rollout-bbox는 act_bbox 정책용입니다. --policy.path에 act_bbox 체크포인트를 주세요.")
	if isinstance(cfg.inference, RTCInferenceConfig):
		raise ValueError("bbox는 sync 추론에서만 정책에 들어갑니다. --inference.type=rtc는 쓰지 마세요.")

	recorded = read_training_detector(getattr(cfg.policy, "pretrained_path", None))
	bbox_cfg, warnings = resolve_rollout_bbox(cfg.policy, cfg.bbox, recorded)
	require_robot_cameras(cfg.robot, list(bbox_cfg.cameras))
	detector, settings = load_detector(bbox_cfg)
	logging.info(f"bbox class_conf {settings['class_conf']}")
	problem = realtime_problem(measure_latency_ms(detector, len(bbox_cfg.cameras)), cfg.fps) if cfg.fps else None
	if problem:
		warnings.append(f"{problem} 제어가 느려집니다. GPU(--bbox.device=cuda)를 쓰거나 --fps 를 낮추세요.")
	step = YoloBBoxStep(detector=detector, cameras=list(bbox_cfg.cameras), classes=settings["classes"])
	feature = bbox_dataset_feature(list(bbox_cfg.cameras), settings)
	with (
		with_dataset_feature(rollout_context, feature),
		with_rollout_context_hook(lerobot_rollout, make_observation_pipeline(step), step, warnings),
	):
		# rollout은 parser.wrap으로 감싸여 있어 하위 설정 클래스를 넘기면 명령줄을 다시 읽는다. 원래 함수를 부른다.
		lerobot_rollout.rollout.__wrapped__(cfg)
	summary = step.summary()
	if summary:
		logging.info(summary)


def main() -> None:
	register_third_party_plugins()
	rollout_bbox()


if __name__ == "__main__":
	main()
