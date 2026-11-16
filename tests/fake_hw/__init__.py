"""시험 전용 가짜 로봇·텔레오퍼레이터. 실제 장치 없이 lerobot-record-bbox와 lerobot-rollout-bbox를 돌린다.

명령줄에서는 PYTHONPATH에 tests/를 넣고 --robot.discover_packages_path=fake_hw 로 불러온다.
"""

import glob
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from lerobot.robots import Robot, RobotConfig
from lerobot.teleoperators import Teleoperator, TeleoperatorConfig

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
FRAME_SHAPE = (480, 640, 3)
# 카메라마다 다른 이미지를 보여 줘서 카메라 순서가 뒤바뀌면 시험이 잡게 한다.
CAMERA_OFFSET = 7


def load_frames(frames_dir: str) -> list[np.ndarray]:
	"""frames_dir의 이미지를 LeRobot 카메라와 같은 RGB 480x640으로 읽는다."""
	paths = sorted(glob.glob(os.path.join(frames_dir, "*")))
	if not paths:
		raise FileNotFoundError(f"이미지가 없습니다: {frames_dir}")
	size = (FRAME_SHAPE[1], FRAME_SHAPE[0])
	return [cv2.cvtColor(cv2.resize(cv2.imread(path), size), cv2.COLOR_BGR2RGB) for path in paths]


def camera_frame_index(index: int, camera_position: int, frame_count: int) -> int:
	"""index번째 관측에서 camera_position번째 카메라가 보여 주는 이미지 번호."""
	return (index + CAMERA_OFFSET * camera_position) % frame_count


@RobotConfig.register_subclass("bbox_replay_robot")
@dataclass
class BBoxReplayRobotConfig(RobotConfig):
	frames_dir: str = ""
	camera_names: list[str] = field(default_factory=lambda: ["front", "wrist"])
	# 받은 행동을 한 줄씩 적을 파일 (비우면 적지 않는다)
	action_log: str = ""


class BBoxReplayRobot(Robot):
	"""이미지를 순서대로 카메라에 내보내고 받은 행동을 기록한다. shoulder_pan.pos에 이미지 번호를 넣는다."""

	config_class = BBoxReplayRobotConfig
	name = "bbox_replay_robot"

	def __init__(self, config: BBoxReplayRobotConfig) -> None:
		super().__init__(config)
		self.config = config
		self.frames = load_frames(config.frames_dir)
		self.cameras = {name: None for name in config.camera_names}
		self._index = 0
		self._is_connected = False

	@property
	def observation_features(self) -> dict[str, Any]:
		joints = {f"{joint}.pos": float for joint in JOINTS}
		return {**joints, **{name: FRAME_SHAPE for name in self.config.camera_names}}

	@property
	def action_features(self) -> dict[str, type]:
		return {f"{joint}.pos": float for joint in JOINTS}

	@property
	def is_connected(self) -> bool:
		return self._is_connected

	@property
	def is_calibrated(self) -> bool:
		return True

	def connect(self, calibrate: bool = True) -> None:
		self._is_connected = True

	def calibrate(self) -> None:
		pass

	def configure(self) -> None:
		pass

	def get_observation(self) -> dict[str, Any]:
		index = self._index % len(self.frames)
		self._index += 1
		observation: dict[str, Any] = {f"{joint}.pos": 0.0 for joint in JOINTS}
		observation["shoulder_pan.pos"] = float(index)
		for position, name in enumerate(self.config.camera_names):
			observation[name] = self.frames[camera_frame_index(index, position, len(self.frames))]
		return observation

	def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
		if self.config.action_log:
			with open(self.config.action_log, "a", encoding="utf-8") as file:
				file.write(json.dumps({key: float(value) for key, value in action.items()}) + "\n")
		return action

	def disconnect(self) -> None:
		self._is_connected = False


@TeleoperatorConfig.register_subclass("bbox_replay_teleop")
@dataclass
class BBoxReplayTeleopConfig(TeleoperatorConfig):
	pass


class BBoxReplayTeleop(Teleoperator):
	"""사인파 관절 목표를 내는 가짜 리더 팔."""

	config_class = BBoxReplayTeleopConfig
	name = "bbox_replay_teleop"

	def __init__(self, config: BBoxReplayTeleopConfig) -> None:
		super().__init__(config)
		self._step = 0
		self._is_connected = False

	@property
	def action_features(self) -> dict[str, type]:
		return {f"{joint}.pos": float for joint in JOINTS}

	@property
	def feedback_features(self) -> dict[str, type]:
		return {}

	@property
	def is_connected(self) -> bool:
		return self._is_connected

	@property
	def is_calibrated(self) -> bool:
		return True

	def connect(self, calibrate: bool = True) -> None:
		self._is_connected = True

	def calibrate(self) -> None:
		pass

	def configure(self) -> None:
		pass

	def get_action(self) -> dict[str, float]:
		self._step += 1
		return {f"{joint}.pos": 10.0 * math.sin(0.1 * self._step + position) for position, joint in enumerate(JOINTS)}

	def send_feedback(self, feedback: dict[str, Any]) -> None:
		pass

	def disconnect(self) -> None:
		self._is_connected = False
