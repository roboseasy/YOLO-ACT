"""LeRobot 0.6.0에 훅이 없는 지점을 이 프로세스 안에서만 감싼다.

shortcut: LeRobot 모듈의 함수를 with 블록 동안 바꿔 끼운다. LeRobot이 ENV 특징과 관측 처리 단계 훅을 주면 그쪽으로 옮긴다.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType
from typing import Any

from lerobot.processor import RobotObservation, RobotProcessorPipeline

from .bbox import YoloBBoxStep


@contextmanager
def with_dataset_feature(module: ModuleType, feature: dict[str, dict]) -> Iterator[None]:
	"""module.combine_feature_dicts가 만든 데이터셋 특징 뒤에 feature를 붙인다.

	원래 함수에 함께 넘기면 1차원 특징을 dtype·names·shape만으로 다시 만들어 info(YOLO 설정)가 지워진다.
	"""
	original = module.combine_feature_dicts

	def combine_feature_dicts(*dicts: dict) -> dict:
		return {**original(*dicts), **feature}

	module.combine_feature_dicts = combine_feature_dicts
	try:
		yield
	finally:
		module.combine_feature_dicts = original


@contextmanager
def with_record_loop_hook(module: ModuleType, step: YoloBBoxStep) -> Iterator[None]:
	"""record_loop 호출(에피소드 또는 리셋 구간)마다 bbox 유지값을 비우고, 끝나면 검출 요약을 남긴다."""
	original = module.record_loop

	def record_loop(*args: Any, **kwargs: Any) -> Any:
		step.reset()
		try:
			return original(*args, **kwargs)
		finally:
			summary = step.summary()
			if summary:
				phase = "episode" if kwargs.get("dataset") is not None else "reset"
				logging.info(f"[{phase}] {summary}")

	module.record_loop = record_loop
	try:
		yield
	finally:
		module.record_loop = original


@contextmanager
def with_rollout_context_hook(
	module: ModuleType,
	pipeline: RobotProcessorPipeline[RobotObservation, RobotObservation],
	step: YoloBBoxStep,
	warnings: list[str],
) -> Iterator[None]:
	"""module.build_rollout_context에 bbox 관측 파이프라인을 넘기고, 추론 엔진 reset(에피소드 시작) 때 bbox도 비운다.

	warnings는 컨텍스트가 다 만들어진 뒤(제어 루프 직전)에 남긴다. 그 전에 찍으면 시작 로그에 밀려 묻힌다.
	"""
	original = module.build_rollout_context

	def build_rollout_context(cfg: Any, shutdown_event: Any, *args: Any, **kwargs: Any) -> Any:
		kwargs["robot_observation_processor"] = pipeline
		ctx = original(cfg, shutdown_event, *args, **kwargs)
		for message in warnings:
			logging.warning(message)
		engine_reset = ctx.policy.inference.reset

		def reset() -> None:
			summary = step.summary()
			if summary:
				logging.info(summary)
			step.reset()
			engine_reset()

		ctx.policy.inference.reset = reset
		return ctx

	module.build_rollout_context = build_rollout_context
	try:
		yield
	finally:
		module.build_rollout_context = original
