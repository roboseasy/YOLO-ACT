"""lerobot-rollout-bbox가 정책에 저장된 녹화 때 YOLO 설정을 어떻게 쓰는지 시험한다."""

import hashlib

import pytest

from lerobot_policy_act_bbox import ACTBBoxConfig
from lerobot_policy_act_bbox.bbox import BBoxConfig, bbox_feature_names
from lerobot_policy_act_bbox.rollout import read_training_detector, resolve_rollout_bbox

NAMES = bbox_feature_names(["front", "wrist"], ["box", "pallet"])


def make_policy_config(model: str = "/weights/best.pt", sha: str = "ab12") -> ACTBBoxConfig:
	config = ACTBBoxConfig(device="cpu")
	config.bbox_names = NAMES
	config.bbox_cameras = ["front", "wrist"]
	config.bbox_model = model
	config.bbox_model_sha256 = sha
	config.bbox_classes = ["box", "pallet"]
	config.bbox_conf = 0.3
	config.bbox_is_bgr_input = True
	return config


def test_resolve_uses_policy_settings():
	resolved, warnings = resolve_rollout_bbox(make_policy_config(), BBoxConfig())
	assert resolved.model == "/weights/best.pt"
	assert resolved.cameras == ["front", "wrist"]
	assert resolved.classes == ["box", "pallet"]
	assert resolved.conf == 0.3
	assert resolved.is_bgr_input is True
	assert warnings == []


def test_resolve_override_warns():
	resolved, warnings = resolve_rollout_bbox(make_policy_config(), BBoxConfig(conf=0.5, device="cpu"))
	assert resolved.conf == 0.5
	assert resolved.device == "cpu"
	assert len(warnings) == 1
	assert "conf" in warnings[0]


def test_resolve_camera_mismatch_raises():
	with pytest.raises(ValueError, match="wrist"):
		resolve_rollout_bbox(make_policy_config(), BBoxConfig(cameras=["wrist", "front"]))


def test_resolve_sha_mismatch_warns(tmp_path):
	weights = tmp_path / "best.pt"
	weights.write_bytes(b"retrained weights")
	_, warnings = resolve_rollout_bbox(make_policy_config(model=str(weights), sha="ab12"), BBoxConfig())
	assert len(warnings) == 1
	assert "sha256" in warnings[0]


def test_resolve_same_weights_no_warning(tmp_path):
	weights = tmp_path / "best.pt"
	weights.write_bytes(b"same weights")
	sha = hashlib.sha256(b"same weights").hexdigest()
	_, warnings = resolve_rollout_bbox(make_policy_config(model=str(weights), sha=sha), BBoxConfig(model=str(weights)))
	assert warnings == []


def test_resolve_requires_model_when_unknown():
	with pytest.raises(ValueError, match="--bbox.model"):
		resolve_rollout_bbox(make_policy_config(model="", sha=""), BBoxConfig())


def test_rollout_checks_robot_cameras_before_connecting():
	from lerobot.cameras.opencv import OpenCVCameraConfig
	from lerobot.robots.so_follower import SO101FollowerConfig

	from lerobot_policy_act_bbox.rollout import RolloutBBoxConfig, rollout_bbox

	camera = OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30)
	robot = SO101FollowerConfig(port="/dev/null", cameras={"cam0": camera})
	cfg = RolloutBBoxConfig(robot=robot, policy=make_policy_config())
	with pytest.raises(ValueError, match="front"):
		rollout_bbox.__wrapped__(cfg)


# 학습 데이터셋의 YOLO 기록 — 클래스별 conf는 여기에만 있다 (체크포인트 설정은 bbox_conf 하나)
RECORDED = {
	"model": "/old/best.pt",
	"model_sha256": "ab12",
	"classes": ["box", "pallet"],
	"cameras": ["front", "wrist"],
	"conf": 0.3,
	"class_conf": [0.9, 0.3],
	"is_bgr_input": True,
}


def test_resolve_uses_class_conf_from_training_dataset():
	resolved, warnings = resolve_rollout_bbox(make_policy_config(), BBoxConfig(), RECORDED)
	assert resolved.class_conf == [0.9, 0.3]
	assert warnings == []


@pytest.mark.parametrize(
	"changed", [{"model_sha256": "zz99"}, {"classes": ["pallet", "box"]}, {"conf": 0.5}, {"class_conf": [0.9]}]
)
def test_resolve_ignores_dataset_that_does_not_match_the_checkpoint(changed):
	resolved, _ = resolve_rollout_bbox(make_policy_config(), BBoxConfig(), {**RECORDED, **changed})
	assert resolved.class_conf is None  # 모든 클래스에 bbox_conf


def test_resolve_class_conf_override_warns():
	resolved, warnings = resolve_rollout_bbox(make_policy_config(), BBoxConfig(class_conf=[0.5, 0.5]), RECORDED)
	assert resolved.class_conf == [0.5, 0.5]
	assert any("class_conf" in message for message in warnings)


def test_single_conf_override_applies_to_every_class():
	resolved, _ = resolve_rollout_bbox(make_policy_config(), BBoxConfig(conf=0.5), RECORDED)
	assert resolved.class_conf is None
	assert resolved.conf == 0.5


def test_read_training_detector(tmp_path):
	import json

	root = tmp_path / "dataset"
	(root / "meta").mkdir(parents=True)
	info = {"features": {"observation.environment_state": {"info": {"bbox_detector": RECORDED}}}}
	(root / "meta" / "info.json").write_text(json.dumps(info))
	checkpoint = tmp_path / "pretrained_model"
	checkpoint.mkdir()
	(checkpoint / "train_config.json").write_text(json.dumps({"dataset": {"repo_id": "user/data", "root": str(root)}}))
	assert read_training_detector(checkpoint) == RECORDED
	assert read_training_detector(tmp_path / "missing") is None
	assert read_training_detector("user/hub_repo") is None
