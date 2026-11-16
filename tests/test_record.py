"""lerobot-record-bbox의 카메라 결정 규칙 시험."""

from types import SimpleNamespace

import pytest

from lerobot_policy_act_bbox.bbox import BBoxConfig
from lerobot_policy_act_bbox.record import check_resume_settings, resolve_record_cameras


def make_config(cameras: list[str] | None = None, robot_cameras: dict | None = None) -> SimpleNamespace:
	robot = SimpleNamespace() if robot_cameras is None else SimpleNamespace(cameras=robot_cameras)
	return SimpleNamespace(bbox=BBoxConfig(cameras=cameras), robot=robot)


def test_cameras_default_to_robot_cameras():
	assert resolve_record_cameras(make_config(robot_cameras={"front": object(), "wrist": object()})) == ["front", "wrist"]


def test_explicit_cameras_keep_order():
	assert resolve_record_cameras(make_config(cameras=["wrist", "front"], robot_cameras={"front": 1, "wrist": 2})) == [
		"wrist",
		"front",
	]


def test_unknown_camera_raises():
	with pytest.raises(ValueError, match="side"):
		resolve_record_cameras(make_config(cameras=["side"], robot_cameras={"front": 1}))


def test_no_cameras_raises():
	with pytest.raises(ValueError, match="--bbox.cameras"):
		resolve_record_cameras(make_config())


EXISTING_FEATURE = {
	"names": [],
	"info": {
		"bbox_detector": {
			"model": "/old/place/best.pt",
			"model_sha256": "ab12",
			"classes": ["box", "pallet"],
			"conf": 0.3,
			"is_bgr_input": False,
			"cameras": ["front", "wrist"],
		}
	},
}
SAME_SETTINGS = {"model": "/new/place/best.pt", "model_sha256": "ab12", "classes": ["box", "pallet"], "conf": 0.3, "is_bgr_input": False}


def test_resume_accepts_same_settings_from_moved_weights():
	check_resume_settings(EXISTING_FEATURE, SAME_SETTINGS, ["front", "wrist"])


def test_resume_rejects_changed_settings():
	with pytest.raises(ValueError, match="conf"):
		check_resume_settings(EXISTING_FEATURE, {**SAME_SETTINGS, "conf": 0.6}, ["front", "wrist"])
	with pytest.raises(ValueError, match="model_sha256"):
		check_resume_settings(EXISTING_FEATURE, {**SAME_SETTINGS, "model_sha256": "cd34"}, ["front", "wrist"])
	with pytest.raises(ValueError, match="cameras"):
		check_resume_settings(EXISTING_FEATURE, SAME_SETTINGS, ["wrist", "front"])


def test_resume_rejects_dataset_without_bbox():
	with pytest.raises(ValueError, match="observation.environment_state"):
		check_resume_settings(None, SAME_SETTINGS, ["front", "wrist"])


def test_resume_accepts_dataset_recorded_before_class_conf():
	# 이전 버전 데이터셋에는 class_conf가 없다: 모든 클래스가 conf 하나였다
	check_resume_settings(EXISTING_FEATURE, {**SAME_SETTINGS, "class_conf": [0.3, 0.3]}, ["front", "wrist"])


def test_resume_rejects_changed_class_conf():
	detector = {**EXISTING_FEATURE["info"]["bbox_detector"], "class_conf": [0.9, 0.3]}
	feature = {"names": [], "info": {"bbox_detector": detector}}
	check_resume_settings(feature, {**SAME_SETTINGS, "class_conf": [0.9, 0.3]}, ["front", "wrist"])
	with pytest.raises(ValueError, match="class_conf"):
		check_resume_settings(feature, {**SAME_SETTINGS, "class_conf": [0.8, 0.3]}, ["front", "wrist"])
