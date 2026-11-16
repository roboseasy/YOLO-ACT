"""bbox 처리 단계 시험: 클래스별 5칸, 칸 순서, 빈 칸, YOLO 입력 색, 요약 로그."""

import numpy as np
import pytest
import torch
from ultralytics.engine.results import Results

from lerobot_policy_act_bbox.bbox import (
	BBoxConfig,
	YoloBBoxStep,
	YoloDetector,
	bbox_dataset_feature,
	bbox_feature_names,
	load_detector,
	make_observation_pipeline,
	measure_latency_ms,
	pick_boxes,
	realtime_problem,
	resolve_class_ids,
)

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
EMPTY = np.zeros((0, 4), dtype=np.float32)
CAMERAS = ["front", "wrist"]
CLASSES = ["box", "pallet"]


class ScriptedDetector:
	"""정해진 순서대로 [카메라][클래스] 박스 배열을 돌려주는 탐지기 대역."""

	def __init__(self, outputs: list[list[list[np.ndarray]]]) -> None:
		self.outputs = list(outputs)

	def __call__(self, frames: list[np.ndarray]) -> list[list[np.ndarray]]:
		assert len(frames) == 2
		return self.outputs.pop(0)


class RecordingModel:
	"""Ultralytics YOLO 대역: predict 입력을 기록하고 실제 Results를 돌려준다."""

	names = {0: "box", 1: "pallet"}

	def __init__(self) -> None:
		self.sources: list[np.ndarray] | None = None
		self.kwargs: dict | None = None

	def predict(self, source: list[np.ndarray], **kwargs) -> list[Results]:
		self.sources = source
		self.kwargs = kwargs
		boxes = torch.tensor([[64.0, 48.0, 320.0, 240.0, 0.9, 0.0], [0.0, 0.0, 64.0, 48.0, 0.8, 1.0]])
		return [Results(image, path="", names=self.names, boxes=boxes) for image in source]


def make_observation() -> dict:
	return {"front": FRAME, "wrist": FRAME, "shoulder_pan.pos": 1.0}


def make_step(outputs: list[list[list[np.ndarray]]]) -> YoloBBoxStep:
	return YoloBBoxStep(detector=ScriptedDetector(outputs), cameras=CAMERAS, classes=CLASSES)


def test_feature_names_order():
	names = bbox_feature_names(CAMERAS, CLASSES)
	assert len(names) == 140
	assert names[:8] == [
		"front_box0_x1",
		"front_box0_y1",
		"front_box0_x2",
		"front_box0_y2",
		"front_box0_cx",
		"front_box0_cy",
		"front_box0_is_detected",
		"front_box1_x1",
	]
	assert names[35] == "front_pallet0_x1"
	assert names[70] == "wrist_box0_x1"
	assert names[-1] == "wrist_pallet4_is_detected"


def test_step_fills_slots_and_zeros_the_rest():
	boxes = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.5, 0.6, 0.6]], dtype=np.float32)
	observation = make_step([[[boxes, EMPTY], [EMPTY, EMPTY]]]).observation(make_observation())
	assert observation["front_box0_cx"] == pytest.approx(0.2)
	assert observation["front_box0_cy"] == pytest.approx(0.3)
	assert observation["front_box0_is_detected"] == 1.0
	assert observation["front_box1_x1"] == pytest.approx(0.5)
	assert observation["front_box1_is_detected"] == 1.0
	empty_slot = ["front_box2_x1", "front_box2_y1", "front_box2_x2", "front_box2_y2", "front_box2_cx", "front_box2_cy", "front_box2_is_detected"]
	assert [observation[key] for key in empty_slot] == [0.0] * 7
	assert observation["front_pallet0_is_detected"] == 0.0
	assert observation["wrist_box0_x1"] == 0.0
	assert observation["shoulder_pan.pos"] == 1.0


def test_step_does_not_keep_missed_boxes():
	box = np.array([[0.5, 0.5, 0.7, 0.9]], dtype=np.float32)
	step = make_step([[[box, EMPTY], [EMPTY, EMPTY]], [[EMPTY, EMPTY], [EMPTY, EMPTY]]])
	step.observation(make_observation())
	observation = step.observation(make_observation())
	assert observation["front_box0_x1"] == 0.0
	assert observation["front_box0_is_detected"] == 0.0


def test_missing_camera_raises():
	with pytest.raises(KeyError, match="wrist.*front"):
		make_step([]).observation({"front": FRAME})


def test_pipeline_runs_step():
	box = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
	observation = make_observation_pipeline(make_step([[[box, EMPTY], [box, box]]]))(make_observation())
	assert observation["wrist_pallet0_cx"] == pytest.approx(0.2)
	assert observation["front"] is FRAME


def test_pick_boxes_keeps_top_confidence_then_orders_by_size():
	boxes = np.array(
		[
			[0.0, 0.0, 0.125, 0.125],  # 0: 넓이 0.015625, conf 0.9
			[0.0, 0.0, 0.25, 0.25],  # 1: 0.0625, conf 0.8
			[0.0, 0.0, 0.375, 0.375],  # 2: 0.140625, conf 0.7
			[0.0, 0.0, 0.5, 0.25],  # 3: 0.125, conf 0.6
			[0.0, 0.0, 0.5, 0.5],  # 4: 0.25, conf 0.5
			[0.0, 0.0, 1.0, 1.0],  # 5: 가장 크지만 confidence 6등이라 빠진다
			[0.0, 0.0, 0.75, 0.75],  # 6: confidence 7등
			[0.0, 0.0, 0.9, 0.9],  # 7: 다른 클래스
			[0.0, 0.0, 0.8, 0.8],  # 8: conf 하한 미만
		]
	)
	classes = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0])
	confs = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.31, 0.99, 0.2])
	picked = pick_boxes(classes, confs, boxes, class_id=0, min_conf=0.3)
	np.testing.assert_allclose(picked, boxes[[4, 2, 3, 1, 0]])


def test_pick_boxes_breaks_size_ties_from_bottom_left():
	boxes = np.array(
		[
			[0.5, 0.25, 0.75, 0.5],  # 오른쪽
			[0.0, 0.0, 0.25, 0.25],  # 왼쪽 위
			[0.0, 0.5, 0.25, 0.75],  # 왼쪽 아래
		]
	)
	picked = pick_boxes(np.zeros(3), np.full(3, 0.9), boxes, class_id=0, min_conf=0.3)
	np.testing.assert_allclose(picked, boxes[[2, 1, 0]])


def test_pick_boxes_without_candidates_is_empty():
	assert pick_boxes(np.array([]), np.array([]), np.zeros((0, 4)), class_id=0, min_conf=0.3).shape == (0, 4)


def test_resolve_class_ids():
	assert resolve_class_ids({0: "box", 1: "pallet"}, None) == [0, 1]
	assert resolve_class_ids({0: "box", 1: "pallet"}, ["pallet"]) == [1]
	with pytest.raises(ValueError, match="dice"):
		resolve_class_ids({0: "box"}, ["dice"])


def test_detector_flips_rgb_to_bgr():
	rgb = np.zeros((480, 640, 3), dtype=np.uint8)
	rgb[..., 0] = 255
	model = RecordingModel()
	boxes = YoloDetector(model, class_ids=[0, 1], conf=0.3, is_bgr_input=False)([rgb])
	assert model.sources[0][0, 0].tolist() == [0, 0, 255]
	assert model.kwargs["conf"] == 0.3
	assert model.kwargs["classes"] == [0, 1]
	np.testing.assert_allclose(boxes[0][0], [[0.1, 0.1, 0.5, 0.5]], atol=1e-6)
	np.testing.assert_allclose(boxes[0][1], [[0.0, 0.0, 0.1, 0.1]], atol=1e-6)

	bgr_model = RecordingModel()
	YoloDetector(bgr_model, class_ids=[0, 1], conf=0.3, is_bgr_input=True)([rgb])
	assert bgr_model.sources[0][0, 0].tolist() == [255, 0, 0]


def test_summary_reports_rate_and_latency():
	box = np.array([[0.1, 0.1, 0.2, 0.2]], dtype=np.float32)
	seen = [[box, EMPTY], [EMPTY, EMPTY]]
	missed = [[EMPTY, EMPTY], [EMPTY, EMPTY]]
	step = make_step([seen, seen, missed, seen])
	assert step.summary() is None
	for _ in range(4):
		step.observation(make_observation())
	text = step.summary()
	assert "4 frames" in text
	assert "front box 75%" in text
	assert "front pallet 0%" in text
	assert "ms" in text
	step.reset()
	assert step.summary() is None


def test_dataset_feature():
	settings = {
		"model": "/weights/best.pt",
		"model_sha256": "ab12",
		"classes": CLASSES,
		"conf": 0.3,
		"is_bgr_input": False,
	}
	feature = bbox_dataset_feature(CAMERAS, settings)["observation.environment_state"]
	assert feature["dtype"] == "float32"
	assert tuple(feature["shape"]) == (140,)
	assert feature["names"][-1] == "wrist_pallet4_is_detected"
	assert feature["info"]["bbox_detector"] == {**settings, "cameras": CAMERAS}


def test_load_detector_validates_input():
	with pytest.raises(ValueError, match="--bbox.model"):
		load_detector(BBoxConfig())
	with pytest.raises(ValueError, match="conf"):
		load_detector(BBoxConfig(model="/nonexistent/best.pt", conf=1.5))
	with pytest.raises(FileNotFoundError, match="--bbox.model"):
		load_detector(BBoxConfig(model="/nonexistent/best.pt"))


def test_detector_applies_class_thresholds():
	# RecordingModel: box conf 0.9, pallet conf 0.8
	model = RecordingModel()
	boxes = YoloDetector(model, class_ids=[0, 1], conf=[0.95, 0.5], is_bgr_input=False)([FRAME])
	assert model.kwargs["conf"] == 0.5  # YOLO에는 가장 낮은 하한
	assert boxes[0][0].shape == (0, 4)  # box 0.9 < 0.95 라 칸에 넣지 않는다
	np.testing.assert_allclose(boxes[0][1], [[0.0, 0.0, 0.1, 0.1]], atol=1e-6)


def test_load_detector_records_class_thresholds(tmp_path, monkeypatch):
	import ultralytics

	weights = tmp_path / "best.pt"
	weights.write_bytes(b"weights")
	monkeypatch.setattr(ultralytics, "YOLO", lambda path: RecordingModel())
	detector, settings = load_detector(BBoxConfig(model=str(weights), class_conf=[0.9, 0.6]))
	assert settings["class_conf"] == [0.9, 0.6]
	assert settings["conf"] == 0.6  # act_bbox 설정(bbox_conf)은 가장 낮은 값을 읽는다
	assert detector.class_conf == [0.9, 0.6]
	_, single = load_detector(BBoxConfig(model=str(weights), conf=0.4))
	assert single["class_conf"] == [0.4, 0.4]
	with pytest.raises(ValueError, match="class_conf"):
		load_detector(BBoxConfig(model=str(weights), class_conf=[0.9]))
	with pytest.raises(ValueError, match="class_conf"):
		load_detector(BBoxConfig(model=str(weights), class_conf=[0.9, 1.2]))


def test_realtime_problem():
	assert realtime_problem(10.0, 30) is None
	message = realtime_problem(80.0, 30)
	assert "80 ms" in message and "30 fps" in message and "8 fps 이하" in message
	assert "이하" not in realtime_problem(800.0, 30)  # 1 fps도 못 따라가면 fps 안내를 하지 않는다


def test_measure_latency_uses_one_batch_per_camera():
	sizes = []
	assert measure_latency_ms(lambda frames: sizes.append(len(frames)), 2) >= 0.0
	assert set(sizes) == {2}
