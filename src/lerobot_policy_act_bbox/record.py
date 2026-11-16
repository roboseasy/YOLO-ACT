"""lerobot-record-bbox: lerobot-record와 옵션이 같고, --bbox.* 로 YOLO bbox를 함께 기록한다.

bbox는 observation.environment_state(카메라당 x1,y1,x2,y2,cx,cy,is_detected)로 저장되고,
YOLO 설정은 그 특징의 info.bbox_detector에 남아 학습(act_bbox)과 추론(lerobot-rollout-bbox)으로 이어진다.

예:
	lerobot-record-bbox \\
		--robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=follower \\
		--robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \\
		--teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=leader \\
		--dataset.repo_id=${HF_USER}/pick_box --dataset.single_task="Pick up the box" \\
		--bbox.model=~/workspace/YOLO/outputs/runs/box_pallet/weights/best.pt
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lerobot.configs import parser
from lerobot.datasets import LeRobotDataset
from lerobot.scripts import lerobot_record
from lerobot.scripts.lerobot_record import RecordConfig
from lerobot.utils.constants import HF_LEROBOT_HOME, OBS_ENV_STATE
from lerobot.utils.import_utils import register_third_party_plugins

from .bbox import (
	BBoxConfig,
	YoloBBoxStep,
	bbox_dataset_feature,
	load_detector,
	make_observation_pipeline,
	measure_latency_ms,
	realtime_problem,
	require_robot_cameras,
)
from .lerobot_hooks import with_dataset_feature, with_record_loop_hook

# 이어 녹화할 때 같아야 하는 YOLO 설정. 가중치는 옮길 수 있으므로 경로가 아니라 sha256으로 비교한다.
RESUME_KEYS = ("model_sha256", "classes", "conf", "class_conf", "is_bgr_input", "cameras")


def with_class_conf(settings: dict[str, Any]) -> dict[str, Any]:
	"""class_conf가 없는 설정(이전 버전으로 녹화한 데이터셋)은 모든 클래스가 conf 하나였다."""
	if settings.get("class_conf"):
		return settings
	return {**settings, "class_conf": [settings.get("conf")] * len(settings.get("classes") or [])}


@dataclass
class RecordBBoxConfig(RecordConfig):
	bbox: BBoxConfig = field(default_factory=BBoxConfig)


def resolve_record_cameras(cfg: RecordBBoxConfig) -> list[str]:
	"""bbox를 계산할 카메라. --bbox.cameras가 없으면 로봇 설정의 카메라 전부를 그 순서대로 쓴다."""
	cameras = list(cfg.bbox.cameras or getattr(cfg.robot, "cameras", None) or {})
	if not cameras:
		raise ValueError("bbox를 계산할 카메라가 없습니다. --bbox.cameras='[front,wrist]' 처럼 지정하세요.")
	require_robot_cameras(cfg.robot, cameras)
	return cameras


def check_resume_settings(existing_feature: dict | None, settings: dict[str, Any], cameras: list[str]) -> None:
	"""이어 녹화할 데이터셋의 YOLO 설정이 지금과 같은지 확인한다. 다르면 한 데이터셋에 두 bbox 분포가 섞인다."""
	if existing_feature is None:
		raise ValueError(f"이어 녹화할 데이터셋에 '{OBS_ENV_STATE}'(bbox)가 없습니다. 새 데이터셋으로 녹화하세요.")
	recorded = with_class_conf((existing_feature.get("info") or {}).get("bbox_detector") or {})
	current = with_class_conf({**settings, "cameras": list(cameras)})
	changed = [
		f"{key}: 기존 {recorded.get(key)!r} → 지금 {current.get(key)!r}"
		for key in RESUME_KEYS
		if recorded.get(key) != current.get(key)
	]
	if changed:
		raise ValueError(
			"이어 녹화할 데이터셋과 YOLO 설정이 다릅니다. 같은 설정으로 녹화하거나 새 데이터셋으로 녹화하세요.\n  "
			+ "\n  ".join(changed)
		)


def read_recorded_bbox_feature(cfg: RecordBBoxConfig) -> dict | None:
	"""이어 녹화할 로컬 데이터셋의 bbox 특징 (없으면 None)."""
	root = Path(cfg.dataset.root) if cfg.dataset.root else HF_LEROBOT_HOME / cfg.dataset.repo_id
	info = json.loads((root / "meta" / "info.json").read_text())
	return info["features"].get(OBS_ENV_STATE)


@parser.wrap()
def record_bbox(cfg: RecordBBoxConfig) -> LeRobotDataset:
	cameras = resolve_record_cameras(cfg)
	detector, settings = load_detector(cfg.bbox)
	latency_ms = measure_latency_ms(detector, len(cameras))
	problem = realtime_problem(latency_ms, cfg.dataset.fps)
	if problem:
		# 로봇에 연결하기 전에 멈춘다. 느린 채로 녹화하면 데이터셋의 시간이 조용히 압축된다.
		raise ValueError(
			f"{problem}\n이대로 녹화하면 프레임이 모자라 데이터셋의 시간이 압축됩니다. "
			"--dataset.fps 를 낮추거나 GPU(--bbox.device=cuda)를 쓰세요."
		)
	logging.info(f"YOLO {latency_ms:.1f} ms/주기 (카메라 {len(cameras)}대, class_conf {settings['class_conf']})")
	if cfg.resume:
		# LeRobot의 이어 녹화 검사는 특징의 info를 비교하지 않아 YOLO 설정이 바뀌어도 그대로 섞인다.
		check_resume_settings(read_recorded_bbox_feature(cfg), settings, cameras)
	step = YoloBBoxStep(detector=detector, cameras=cameras, classes=settings["classes"])
	feature = bbox_dataset_feature(cameras, settings)
	with with_dataset_feature(lerobot_record, feature), with_record_loop_hook(lerobot_record, step):
		# record는 parser.wrap으로 감싸여 있어 하위 설정 클래스를 넘기면 명령줄을 다시 읽는다. 원래 함수를 부른다.
		return lerobot_record.record.__wrapped__(cfg, robot_observation_processor=make_observation_pipeline(step))


def main() -> None:
	register_third_party_plugins()
	record_bbox()


if __name__ == "__main__":
	main()
