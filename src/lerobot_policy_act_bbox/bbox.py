"""YOLO bbox를 LeRobot 관측에 넣는 처리 단계와 데이터셋 특징 정의.

카메라마다, 추적할 클래스마다 박스 칸 5개를 둔다. 칸 하나는
`{camera}_{class}{slot}_{x1,y1,x2,y2,cx,cy,is_detected}` 7개 값이고 좌표는 YOLO의 0~1 정규화 값 그대로다.
- 칸보다 많이 보이면 confidence 상위 5개만 남긴다.
- 남긴 박스는 넓이가 큰 것부터 칸 0, 1, ... 에 넣고, 넓이가 같으면 왼쪽, 그다음 아래에 있는 것부터 넣는다.
- 남는 칸은 0과 is_detected=0이다. 이전 프레임 값을 유지하지 않는다.
- 클래스마다 conf 하한(class_conf)을 따로 둘 수 있다. YOLO에는 그중 가장 낮은 값을 넘기고, 칸에는 클래스별 하한 이상만 넣는다.
녹화(lerobot-record-bbox)와 추론(lerobot-rollout-bbox)이 이 단계를 함께 써서 학습과 실행의 bbox가 같다.
"""

import hashlib
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
	ObservationProcessorStep,
	RobotObservation,
	RobotProcessorPipeline,
	observation_to_transition,
	transition_to_observation,
)
from lerobot.utils.constants import OBS_ENV_STATE

BBOX_FIELDS = ("x1", "y1", "x2", "y2", "cx", "cy", "is_detected")
SLOTS_PER_CLASS = 5
DEFAULT_CONF = 0.3
# 첫 추론은 CUDA 초기화로 약 3초 걸린다. 녹화·롤아웃 전에 이 크기로 한 번 돌려 둔다(이후 첫 호출은 30 ms대).
WARMUP_SHAPE = (480, 640, 3)
# 한 제어 주기에서 YOLO가 쓸 수 있는 몫. 나머지는 카메라·모터 읽기, 송신, 프레임 저장이 쓴다.
YOLO_BUDGET_SHARE = 0.7


@dataclass
class BBoxConfig:
	"""`--bbox.*` 명령줄 옵션. None은 녹화에서는 기본값, 롤아웃에서는 정책에 저장된 값을 뜻한다."""

	# YOLO 가중치(.pt) 경로
	model: str | None = None
	# bbox를 계산할 카메라 이름과 순서. 녹화에서 비우면 로봇 설정의 카메라 전부
	cameras: list[str] | None = None
	# 추적할 클래스 이름과 순서. 비우면 모델 클래스 전부(클래스 번호 순)
	classes: list[str] | None = None
	# 검출 confidence 하한 (녹화 기본 0.3). 모든 클래스에 같은 값을 쓴다
	conf: float | None = None
	# 클래스별 confidence 하한 (classes 순서). 주면 conf 대신 쓴다. 예: '[0.9,0.8]'
	class_conf: list[float] | None = None
	# YOLO 추론 장치 (예: cuda, cpu). 비우면 Ultralytics가 고른다
	device: str | None = None
	# 카메라 프레임이 이미 BGR인가 (ROS 호스트에 lekiwi_client로 붙은 LeKiwi)
	is_bgr_input: bool | None = None


def bbox_feature_names(cameras: list[str], classes: list[str]) -> list[str]:
	"""카메라 → 클래스 → 칸 → 필드 순서로 bbox 특징 이름을 만든다."""
	return [
		f"{camera}_{class_name}{slot}_{field_name}"
		for camera in cameras
		for class_name in classes
		for slot in range(SLOTS_PER_CLASS)
		for field_name in BBOX_FIELDS
	]


def file_sha256(path: str) -> str:
	"""파일 내용의 sha256 (YOLO 가중치가 바뀌었는지 확인할 때 쓴다)."""
	digest = hashlib.sha256()
	with open(path, "rb") as file:
		for chunk in iter(lambda: file.read(1 << 20), b""):
			digest.update(chunk)
	return digest.hexdigest()


def pick_boxes(
	classes: np.ndarray,
	confs: np.ndarray,
	boxes_xyxyn: np.ndarray,
	class_id: int,
	min_conf: float,
) -> np.ndarray:
	"""목표 클래스에서 칸에 넣을 박스(정규화 xyxy, 최대 SLOTS_PER_CLASS개)를 칸 순서대로 돌려준다."""
	candidates = np.flatnonzero((classes == class_id) & (confs >= min_conf))
	kept = candidates[np.argsort(-confs[candidates], kind="stable")[:SLOTS_PER_CLASS]]
	boxes = np.asarray(boxes_xyxyn[kept], dtype=np.float32).reshape(-1, 4)
	areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
	# lexsort는 마지막 키가 1순위다: 넓이 큰 순 → 왼쪽(x1 작은) 순 → 아래(y2 큰) 순.
	return boxes[np.lexsort((-boxes[:, 3], boxes[:, 0], -areas))]


def require_robot_cameras(robot_config: Any, cameras: list[str]) -> None:
	"""로봇 설정에 카메라 목록이 있으면, 로봇에 연결하기 전에 bbox 카메라가 모두 그 안에 있는지 확인한다."""
	robot_cameras = getattr(robot_config, "cameras", None) or {}
	missing = [camera for camera in cameras if robot_cameras and camera not in robot_cameras]
	if missing:
		raise ValueError(f"로봇 설정에 없는 bbox 카메라입니다: {missing}. 로봇 카메라: {list(robot_cameras)}")


def resolve_class_ids(names: dict[int, str], classes: list[str] | None) -> list[int]:
	"""추적할 클래스 이름을 YOLO 클래스 번호로 바꾼다. 비우면 모델 클래스 전부를 번호 순으로 쓴다."""
	if not classes:
		return sorted(names)
	ids = {name: class_id for class_id, name in names.items()}
	unknown = [name for name in classes if name not in ids]
	if unknown:
		raise ValueError(f"모델에 없는 클래스입니다: {unknown}. 있는 클래스: {[names[i] for i in sorted(names)]}")
	return [ids[name] for name in classes]


class YoloDetector:
	"""카메라 프레임 묶음을 YOLO 한 번으로 추론해 [카메라][클래스]마다 칸 순서로 정렬한 박스를 돌려준다."""

	def __init__(
		self,
		model: Any,
		class_ids: list[int],
		conf: float | list[float],
		is_bgr_input: bool,
		device: str | None = None,
	) -> None:
		self.model = model
		self.class_ids = class_ids
		# conf는 모든 클래스에 같은 값 하나이거나 클래스별 목록(class_ids 순서)이다.
		self.class_conf = [float(c) for c in conf] if isinstance(conf, (list, tuple)) else [float(conf)] * len(class_ids)
		self.conf = min(self.class_conf)
		self.is_bgr_input = is_bgr_input
		self.device = device

	def __call__(self, frames: list[np.ndarray]) -> list[list[np.ndarray]]:
		# LeRobot 카메라는 RGB를 주고, Ultralytics의 numpy 입력은 BGR로 읽힌다.
		images = [frame if self.is_bgr_input else np.ascontiguousarray(frame[..., ::-1]) for frame in frames]
		options = {"conf": self.conf, "classes": list(self.class_ids), "verbose": False}
		if self.device:
			options["device"] = self.device
		results = self.model.predict(source=images, **options)
		detections = []
		for result in results:
			classes = result.boxes.cls.cpu().numpy()
			confs = result.boxes.conf.cpu().numpy()
			boxes = result.boxes.xyxyn.cpu().numpy()
			detections.append(
				[
					pick_boxes(classes, confs, boxes, class_id, class_conf)
					for class_id, class_conf in zip(self.class_ids, self.class_conf, strict=True)
				]
			)
		return detections


def measure_latency_ms(detector: Callable[[list[np.ndarray]], Any], n_frames: int, repeats: int = 5) -> float:
	"""카메라 수만큼 묶은 프레임으로 탐지기를 몇 번 돌려 한 번 걸리는 시간(중앙값, ms)을 잰다."""
	frames = [np.zeros(WARMUP_SHAPE, dtype=np.uint8)] * n_frames
	detector(frames)  # 묶음 크기가 바뀐 첫 호출은 느릴 수 있어 재지 않는다
	times = []
	for _ in range(repeats):
		start = time.perf_counter()
		detector(frames)
		times.append((time.perf_counter() - start) * 1000.0)
	return float(np.median(times))


def realtime_problem(latency_ms: float, fps: float) -> str | None:
	"""YOLO가 fps를 따라가지 못하면 그 이유를 돌려준다 (따라가면 None).

	느린 채로 녹화하면 프레임이 모자라는데 LeRobot은 timestamp를 frame_index / fps로 적는다.
	그러면 데이터셋의 시간이 빨리 감긴 것처럼 압축되고, 그 데이터로 학습한 정책은 시연보다 빨리 움직인다.
	"""
	if latency_ms <= 1000.0 / fps * YOLO_BUDGET_SHARE:
		return None
	max_fps = int(1000.0 * YOLO_BUDGET_SHARE / latency_ms)
	limit = f" {max_fps} fps 이하에서는 따라갈 수 있습니다." if max_fps >= 1 else ""
	return f"YOLO가 한 주기에 약 {latency_ms:.0f} ms 걸려 {fps:g} fps를 따라가지 못합니다 (한 주기 {1000.0 / fps:.0f} ms).{limit}"


def load_detector(cfg: BBoxConfig) -> tuple[YoloDetector, dict[str, Any]]:
	"""`--bbox.*` 설정으로 YOLO를 불러오고, 데이터셋과 정책에 기록할 설정값을 함께 돌려준다."""
	if not cfg.model:
		raise ValueError("--bbox.model 에 YOLO 가중치(.pt) 경로를 주세요.")
	conf = DEFAULT_CONF if cfg.conf is None else float(cfg.conf)
	class_conf = None if cfg.class_conf is None else [float(value) for value in cfg.class_conf]
	for value in class_conf or [conf]:
		if not 0.0 < value < 1.0:
			raise ValueError(f"--bbox.conf · --bbox.class_conf 는 0과 1 사이여야 합니다: {value}")
	model_path = os.path.abspath(os.path.expanduser(cfg.model))
	if not os.path.isfile(model_path):
		raise FileNotFoundError(f"YOLO 가중치가 없습니다: {model_path}. --bbox.model 로 경로를 지정하세요.")

	# 플러그인은 모든 LeRobot 명령이 import하므로 무거운 ultralytics는 여기서만 불러온다.
	from ultralytics import YOLO

	model = YOLO(model_path)
	class_ids = resolve_class_ids(model.names, cfg.classes)
	classes = [model.names[class_id] for class_id in class_ids]
	if class_conf is None:
		class_conf = [conf] * len(class_ids)
	elif len(class_conf) != len(class_ids):
		raise ValueError(
			f"--bbox.class_conf 는 클래스마다 하나씩 {len(class_ids)}개여야 합니다 (클래스 {classes}): {class_conf}"
		)
	is_bgr_input = bool(cfg.is_bgr_input)
	settings = {
		"model": model_path,
		"model_sha256": file_sha256(model_path),
		"classes": classes,
		# conf는 가장 낮은 하한이다. act_bbox 설정(bbox_conf)과 이전 버전은 이 값 하나만 읽는다.
		"conf": min(class_conf),
		"class_conf": class_conf,
		"is_bgr_input": is_bgr_input,
	}
	detector = YoloDetector(model, class_ids, class_conf, is_bgr_input, cfg.device)
	detector([np.zeros(WARMUP_SHAPE, dtype=np.uint8)])
	return detector, settings


@dataclass
class YoloBBoxStep(ObservationProcessorStep):
	"""매 제어 주기마다 카메라 프레임에서 bbox를 계산해 관측에 칸별 값을 더한다.

	YOLO는 대기열 없이 이 주기 안에서 동기로 부른다. 늘 최신 프레임을 쓰고 지연이 쌓이지 않는다.
	"""

	detector: Callable[[list[np.ndarray]], list[list[np.ndarray]]]
	cameras: list[str]
	classes: list[str]
	_hit_counts: dict[tuple[str, str], int] = field(default_factory=dict, init=False, repr=False)
	_latencies_ms: list[float] = field(default_factory=list, init=False, repr=False)

	def observation(self, observation: RobotObservation) -> RobotObservation:
		missing = [camera for camera in self.cameras if camera not in observation]
		if missing:
			raise KeyError(f"bbox 카메라 {missing} 가 관측에 없습니다. 관측 키: {sorted(observation)}")

		start = time.perf_counter()
		detections = self.detector([observation[camera] for camera in self.cameras])
		self._latencies_ms.append((time.perf_counter() - start) * 1000.0)

		values: list[float] = []
		for camera, per_class in zip(self.cameras, detections, strict=True):
			for class_name, boxes in zip(self.classes, per_class, strict=True):
				if len(boxes):
					self._hit_counts[(camera, class_name)] = self._hit_counts.get((camera, class_name), 0) + 1
				for slot in range(SLOTS_PER_CLASS):
					if slot < len(boxes):
						x1, y1, x2, y2 = (float(value) for value in boxes[slot])
						values += [x1, y1, x2, y2, (x1 + x2) / 2.0, (y1 + y2) / 2.0, 1.0]
					else:
						values += [0.0] * len(BBOX_FIELDS)
		observation.update(zip(bbox_feature_names(self.cameras, self.classes), values, strict=True))
		return observation

	def reset(self) -> None:
		"""에피소드가 바뀔 때 검출 통계를 비운다."""
		self._hit_counts.clear()
		self._latencies_ms.clear()

	def summary(self) -> str | None:
		"""카메라·클래스별로 1개 이상 검출된 프레임 비율과 YOLO 지연(p50/p95). 처리한 프레임이 없으면 None."""
		frame_count = len(self._latencies_ms)
		if frame_count == 0:
			return None
		rates = " ".join(
			f"{camera} {class_name} {100.0 * self._hit_counts.get((camera, class_name), 0) / frame_count:.0f}%"
			for camera in self.cameras
			for class_name in self.classes
		)
		p50, p95 = np.percentile(self._latencies_ms, [50, 95])
		return f"bbox: {frame_count} frames | 검출 {rates} | yolo p50 {p50:.1f} ms p95 {p95:.1f} ms"

	def transform_features(
		self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
	) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
		# bbox는 observation.state가 아니라 observation.environment_state로 따로 기록한다(bbox_dataset_feature).
		return features


def make_observation_pipeline(step: YoloBBoxStep) -> RobotProcessorPipeline[RobotObservation, RobotObservation]:
	"""record()와 build_rollout_context()에 넘길 로봇 관측 파이프라인."""
	return RobotProcessorPipeline[RobotObservation, RobotObservation](
		steps=[step],
		to_transition=observation_to_transition,
		to_output=transition_to_observation,
	)


def bbox_dataset_feature(cameras: list[str], settings: dict[str, Any]) -> dict[str, dict]:
	"""`observation.environment_state` 데이터셋 특징. YOLO 설정을 info에 함께 남긴다."""
	names = bbox_feature_names(cameras, settings["classes"])
	return {
		OBS_ENV_STATE: {
			"dtype": "float32",
			"shape": (len(names),),
			"names": names,
			"info": {"bbox_detector": {**settings, "cameras": list(cameras)}},
		}
	}
