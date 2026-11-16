"""명령줄 통합 시험: 가짜 로봇으로 녹화하고 데이터셋을 파생한다. CUDA, YOLO 가중치, 프레임 이미지가 필요하다."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from lerobot_policy_act_bbox.bbox import BBoxConfig, file_sha256, load_detector

TESTS_DIR = Path(__file__).resolve().parent
BIN_DIR = Path(sys.executable).parent
# box · pallet 두 클래스를 검출하는 YOLO 가중치와 그 학습 이미지 폴더. 환경 변수로 바꿀 수 있다.
WEIGHTS = Path(os.environ.get("YOLO_ACT_TEST_WEIGHTS", Path.home() / "workspace/YOLO/outputs/runs/box_pallet/weights/best.pt"))
IMAGES_DIR = Path(os.environ.get("YOLO_ACT_TEST_IMAGES", Path.home() / "workspace/YOLO/dataset/train/images"))
ENV_KEY = "observation.environment_state"
IMAGE_KEYS = ["observation.images.front", "observation.images.wrist"]

pytestmark = pytest.mark.skipif(
	not (torch.cuda.is_available() and WEIGHTS.is_file() and IMAGES_DIR.is_dir()),
	reason="CUDA, YOLO 가중치(YOLO_ACT_TEST_WEIGHTS), 프레임 이미지(YOLO_ACT_TEST_IMAGES)가 필요합니다",
)


def run(command: list[str], timeout: int = 900) -> subprocess.CompletedProcess:
	"""명령을 실행하고 실패하면 출력 끝부분과 함께 시험을 실패시킨다."""
	env = {**os.environ, "PYTHONPATH": str(TESTS_DIR)}
	result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=timeout)
	if result.returncode != 0:
		raise AssertionError(
			f"{Path(command[0]).name} 실패 (exit {result.returncode})\n{result.stdout[-4000:]}\n{result.stderr[-4000:]}"
		)
	return result


def hardware_args(workdir: Path) -> list[str]:
	return [
		"--robot.type=bbox_replay_robot",
		"--robot.discover_packages_path=fake_hw",
		f"--robot.frames_dir={workdir / 'frames'}",
		f"--robot.calibration_dir={workdir / 'calibration'}",
		"--robot.id=test",
	]


def read_frames_table(root: Path) -> pd.DataFrame:
	return pd.concat([pd.read_parquet(path) for path in sorted(root.glob("data/**/*.parquet"))], ignore_index=True)


@pytest.fixture(scope="module")
def workdir(tmp_path_factory: pytest.TempPathFactory) -> Path:
	root = tmp_path_factory.mktemp("cli")
	frames = root / "frames"
	frames.mkdir()
	for path in sorted(IMAGES_DIR.iterdir())[::43][:30]:
		shutil.copy(path, frames / path.name)
	return root


@pytest.fixture(scope="module")
def recorded_dataset(workdir: Path) -> Path:
	root = workdir / "bbox_test"
	run(
		[
			str(BIN_DIR / "lerobot-record-bbox"),
			*hardware_args(workdir),
			"--teleop.type=bbox_replay_teleop",
			f"--teleop.calibration_dir={workdir / 'calibration'}",
			"--teleop.id=test",
			"--dataset.repo_id=local/bbox_test",
			f"--dataset.root={root}",
			"--dataset.single_task=pick the box",
			"--dataset.num_episodes=2",
			"--dataset.episode_time_s=2",
			"--dataset.reset_time_s=0.5",
			"--dataset.push_to_hub=false",
			"--play_sounds=false",
			f"--bbox.model={WEIGHTS}",
			"--bbox.cameras=[front,wrist]",
		]
	)
	return root


def derive_dataset(source: Path, target: Path, feature_names: list[str]) -> Path:
	run(
		[
			str(BIN_DIR / "lerobot-edit-dataset"),
			f"--repo_id=local/{source.name}",
			f"--root={source}",
			f"--new_repo_id=local/{target.name}",
			f"--new_root={target}",
			"--operation.type=remove_feature",
			f"--operation.feature_names={json.dumps(feature_names)}",
		]
	)
	return target


@pytest.fixture(scope="module")
def no_image_dataset(recorded_dataset: Path, workdir: Path) -> Path:
	return derive_dataset(recorded_dataset, workdir / "bbox_test_noimg", IMAGE_KEYS)


def train(dataset: Path, output: Path) -> Path:
	"""수정하지 않은 lerobot-train으로 act_bbox를 20스텝 학습하고 마지막 체크포인트 경로를 돌려준다."""
	run(
		[
			str(BIN_DIR / "lerobot-train"),
			f"--dataset.repo_id=local/{dataset.name}",
			f"--dataset.root={dataset}",
			"--policy.type=act_bbox",
			"--policy.device=cuda",
			"--policy.push_to_hub=false",
			"--policy.chunk_size=10",
			"--policy.n_action_steps=10",
			"--steps=20",
			"--batch_size=2",
			"--num_workers=0",
			"--save_freq=20",
			"--log_freq=10",
			"--wandb.enable=false",
			"--job_name=test",
			f"--output_dir={output}",
		]
	)
	return output / "checkpoints/last/pretrained_model"


@pytest.fixture(scope="module")
def trained_policy(recorded_dataset: Path, workdir: Path) -> Path:
	return train(recorded_dataset, workdir / "train_img_bbox")


def rollout_args(policy: Path, workdir: Path) -> list[str]:
	return [
		f"--policy.path={policy}",
		*hardware_args(workdir),
		"--play_sounds=false",
		"--return_to_initial_position=false",
		"--device=cuda",
	]


def test_record_writes_bbox(recorded_dataset: Path, workdir: Path):
	info = json.loads((recorded_dataset / "meta/info.json").read_text())
	assert info["total_episodes"] == 2
	feature = info["features"][ENV_KEY]
	assert feature["shape"] == [140]
	assert feature["names"][:2] == ["front_box0_x1", "front_box0_y1"]
	assert feature["names"][70] == "wrist_box0_x1"
	detector_info = feature["info"]["bbox_detector"]
	assert detector_info["model_sha256"] == file_sha256(str(WEIGHTS))
	assert detector_info["cameras"] == ["front", "wrist"]

	table = read_frames_table(recorded_dataset)
	env = np.stack(table[ENV_KEY].to_numpy())
	state = np.stack(table["observation.state"].to_numpy())
	assert env.shape[1] == 140
	assert np.all((env >= 0.0) & (env <= 1.0))
	names = feature["names"]
	front_detected = env[:, names.index("front_box0_is_detected")] == 1.0
	assert front_detected.any()
	assert (env[:, names.index("front_pallet0_is_detected")] == 1.0).any()

	from fake_hw import camera_frame_index, load_frames

	frames = load_frames(str(workdir / "frames"))
	detector, _ = load_detector(BBoxConfig(model=str(WEIGHTS)))
	for row in np.flatnonzero(front_detected)[:10]:
		index = int(state[row, 0])
		detections = detector(
			[frames[camera_frame_index(index, 0, len(frames))], frames[camera_frame_index(index, 1, len(frames))]]
		)
		# 녹화된 각 칸이 같은 이미지를 오프라인으로 검출해 정렬한 결과와 같아야 한다 (카메라·클래스·칸 순서).
		for camera_position, camera in enumerate(["front", "wrist"]):
			for class_position, class_name in enumerate(["box", "pallet"]):
				boxes = detections[camera_position][class_position]
				for slot in range(5):
					start = names.index(f"{camera}_{class_name}{slot}_x1")
					assert env[row, start + 6] == float(slot < len(boxes))
					if slot < len(boxes):
						np.testing.assert_allclose(env[row, start : start + 4], boxes[slot], atol=2e-3)


def test_episodes_keep_full_length(recorded_dataset: Path):
	# 첫 YOLO 추론(CUDA 초기화, 약 3초)이 녹화 중에 일어나면 첫 에피소드가 1프레임으로 잘린다. 2초 × 30fps = 60프레임.
	counts = read_frames_table(recorded_dataset).groupby("episode_index").size()
	assert counts.min() >= 48, counts.to_dict()


def test_derived_datasets_keep_or_drop_bbox(recorded_dataset: Path, no_image_dataset: Path, workdir: Path):
	features = json.loads((no_image_dataset / "meta/info.json").read_text())["features"]
	assert not any(key.startswith("observation.images.") for key in features)
	assert features[ENV_KEY]["info"]["bbox_detector"]["cameras"] == ["front", "wrist"]

	no_bbox = derive_dataset(recorded_dataset, workdir / "bbox_test_nobbox", [ENV_KEY])
	features = json.loads((no_bbox / "meta/info.json").read_text())["features"]
	assert ENV_KEY not in features
	assert "observation.images.front" in features


def test_train_saves_bbox_settings(trained_policy: Path):
	config = json.loads((trained_policy / "config.json").read_text())
	assert config["type"] == "act_bbox"
	assert len(config["bbox_names"]) == 140
	assert config["bbox_classes"] == ["box", "pallet"]
	assert config["bbox_cameras"] == ["front", "wrist"]
	assert config["bbox_model_sha256"] == file_sha256(str(WEIGHTS))


def test_train_without_images(no_image_dataset: Path, workdir: Path):
	from safetensors import safe_open

	checkpoint = train(no_image_dataset, workdir / "train_bbox_only")
	with safe_open(str(checkpoint / "model.safetensors"), "pt") as weights:
		keys = list(weights.keys())
	assert keys
	assert not any("backbone" in key for key in keys)


def test_rollout_runs_with_bbox(trained_policy: Path, workdir: Path):
	action_log = workdir / "actions.jsonl"
	result = run(
		[
			str(BIN_DIR / "lerobot-rollout-bbox"),
			"--strategy.type=base",
			*rollout_args(trained_policy, workdir),
			f"--robot.action_log={action_log}",
			"--duration=3",
		]
	)
	assert len(action_log.read_text().splitlines()) >= 30
	assert "bbox:" in result.stdout + result.stderr


def test_plain_rollout_explains_bbox(trained_policy: Path, workdir: Path):
	result = subprocess.run(
		[str(BIN_DIR / "lerobot-rollout"), "--strategy.type=base", *rollout_args(trained_policy, workdir), "--duration=2"],
		env={**os.environ, "PYTHONPATH": str(TESTS_DIR)},
		capture_output=True,
		text=True,
		timeout=600,
	)
	assert result.returncode != 0
	assert "lerobot-rollout-bbox" in result.stdout + result.stderr


def test_episodic_rollout_records_bbox(trained_policy: Path, workdir: Path):
	root = workdir / "rollout_bbox_test"
	run(
		[
			str(BIN_DIR / "lerobot-rollout-bbox"),
			"--strategy.type=episodic",
			*rollout_args(trained_policy, workdir),
			"--dataset.repo_id=local/rollout_bbox_test",
			f"--dataset.root={root}",
			"--dataset.single_task=pick the box",
			"--dataset.num_episodes=1",
			"--dataset.episode_time_s=2",
			"--dataset.reset_time_s=0.5",
			"--dataset.push_to_hub=false",
		]
	)
	feature = json.loads((root / "meta/info.json").read_text())["features"][ENV_KEY]
	assert feature["names"][0] == "front_box0_x1"
	assert feature["info"]["bbox_detector"]["cameras"] == ["front", "wrist"]
	env = np.stack(read_frames_table(root)[ENV_KEY].to_numpy())
	assert env.shape[1] == 140
	assert env[:, 6].max() == 1.0


def test_rollout_override_warning_shown_before_control_loop(trained_policy: Path, workdir: Path):
	# 학습 때와 다른 설정 경고가 시작 로그에 밀려 묻히지 않게, 컨텍스트 조립 뒤(제어 루프 직전)에 나와야 한다.
	result = run(
		[
			str(BIN_DIR / "lerobot-rollout-bbox"),
			"--strategy.type=base",
			*rollout_args(trained_policy, workdir),
			"--duration=1",
			"--bbox.conf=0.5",
		]
	)
	lines = result.stderr.splitlines()
	assembled = next(index for index, line in enumerate(lines) if "Rollout context assembled" in line)
	warned = [index for index, line in enumerate(lines) if line.startswith("WARNING") and "--bbox.conf=0.5" in line]
	assert warned
	assert warned[0] > assembled


def test_resume_rejects_changed_bbox_settings(recorded_dataset: Path, workdir: Path):
	root = workdir / "bbox_resume"
	shutil.copytree(recorded_dataset, root)
	result = subprocess.run(
		[
			str(BIN_DIR / "lerobot-record-bbox"),
			*hardware_args(workdir),
			"--teleop.type=bbox_replay_teleop",
			f"--teleop.calibration_dir={workdir / 'calibration'}",
			"--teleop.id=test",
			"--dataset.repo_id=local/bbox_resume",
			f"--dataset.root={root}",
			"--dataset.single_task=pick the box",
			"--dataset.num_episodes=1",
			"--dataset.episode_time_s=1",
			"--dataset.push_to_hub=false",
			"--resume=true",
			"--play_sounds=false",
			f"--bbox.model={WEIGHTS}",
			"--bbox.cameras=[front,wrist]",
			"--bbox.conf=0.6",
		],
		env={**os.environ, "PYTHONPATH": str(TESTS_DIR)},
		capture_output=True,
		text=True,
		timeout=600,
	)
	assert result.returncode != 0
	assert "conf" in result.stdout + result.stderr
	assert json.loads((root / "meta/info.json").read_text())["total_episodes"] == 2
