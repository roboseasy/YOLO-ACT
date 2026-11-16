"""act_bbox 정책 시험: 등록, bbox 입력 필수, 데이터셋 메타데이터, 학습·추론, 저장·불러오기, 정규화."""

import pytest
import torch
from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE

from lerobot_policy_act_bbox import ACTBBoxConfig, ACTBBoxPolicy
from lerobot_policy_act_bbox.modeling_act_bbox import BBoxEncoder

IMAGE_KEY = "observation.images.front"
BBOX_NAMES = [f"{camera}_{name}" for camera in ("front", "wrist") for name in ("x1", "y1", "x2", "y2", "cx", "cy", "is_detected")]
DETECTOR = {
	"model": "/weights/best.pt",
	"model_sha256": "ab12",
	"classes": ["box", "pallet"],
	"conf": 0.3,
	"is_bgr_input": False,
	"cameras": ["front", "wrist"],
}


def make_config(has_images: bool = True, has_bbox: bool = True) -> ACTBBoxConfig:
	input_features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,))}
	if has_images:
		input_features[IMAGE_KEY] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64))
	if has_bbox:
		input_features[OBS_ENV_STATE] = PolicyFeature(type=FeatureType.ENV, shape=(14,))
	return ACTBBoxConfig(
		input_features=input_features,
		output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
		device="cpu",
		chunk_size=10,
		n_action_steps=10,
		dim_model=64,
		n_heads=2,
		dim_feedforward=128,
		n_encoder_layers=1,
		n_vae_encoder_layers=1,
		pretrained_backbone_weights=None,
	)


def make_batch(config: ACTBBoxConfig, batch_size: int = 2) -> dict[str, torch.Tensor]:
	generator = torch.Generator().manual_seed(0)
	batch = {
		OBS_STATE: torch.randn(batch_size, 6, generator=generator),
		OBS_ENV_STATE: torch.rand(batch_size, 14, generator=generator),
		ACTION: torch.randn(batch_size, 10, 6, generator=generator),
		"action_is_pad": torch.zeros(batch_size, 10, dtype=torch.bool),
	}
	if config.image_features:
		batch[IMAGE_KEY] = torch.rand(batch_size, 3, 64, 64, generator=generator)
	return batch


def test_registered():
	assert PreTrainedConfig.get_choice_class("act_bbox") is ACTBBoxConfig
	assert get_policy_class("act_bbox") is ACTBBoxPolicy
	preprocessor, postprocessor = make_pre_post_processors(make_config())
	assert preprocessor is not None and postprocessor is not None


def test_requires_env_state():
	with pytest.raises(ValueError, match="observation.environment_state"):
		make_config(has_bbox=False).validate_features()


def test_bbox_names_must_match_env_dim():
	config = make_config()
	config.bbox_names = BBOX_NAMES[:7]
	with pytest.raises(ValueError, match="7"):
		config.validate_features()


def test_reads_dataset_metadata():
	config = make_config()
	feature = {"dtype": "float32", "shape": [14], "names": BBOX_NAMES, "info": {"bbox_detector": DETECTOR}}
	config.set_dataset_feature_metadata({OBS_ENV_STATE: feature})
	assert config.bbox_names == BBOX_NAMES
	assert config.bbox_cameras == ["front", "wrist"]
	assert config.bbox_model == "/weights/best.pt"
	assert config.bbox_model_sha256 == "ab12"
	assert config.bbox_classes == ["box", "pallet"]
	assert config.bbox_conf == 0.3
	assert config.bbox_is_bgr_input is False


def test_dataset_without_bbox_is_rejected():
	with pytest.raises(ValueError, match="lerobot-record-bbox"):
		make_config().set_dataset_feature_metadata({OBS_STATE: {"dtype": "float32", "shape": [6], "names": None}})


@pytest.mark.parametrize("has_images", [True, False])
def test_forward_backward(has_images: bool):
	policy = ACTBBoxPolicy(make_config(has_images=has_images))
	policy.train()
	loss, _ = policy.forward(make_batch(policy.config))
	loss.backward()
	assert torch.isfinite(loss)
	grads = [parameter.grad for parameter in policy.model.encoder_env_state_input_proj.parameters()]
	assert grads and all(grad is not None and grad.abs().sum() > 0 for grad in grads)


def test_bbox_encoder_replaces_linear():
	policy = ACTBBoxPolicy(make_config())
	assert isinstance(policy.model.encoder_env_state_input_proj, BBoxEncoder)


def test_bbox_changes_actions():
	policy = ACTBBoxPolicy(make_config())
	policy.eval()
	batch = make_batch(policy.config)
	moved = dict(batch)
	moved[OBS_ENV_STATE] = 1.0 - batch[OBS_ENV_STATE]
	assert not torch.allclose(policy.predict_action_chunk(batch), policy.predict_action_chunk(moved))


def test_save_load_roundtrip(tmp_path):
	config = make_config()
	config.bbox_names = BBOX_NAMES
	config.bbox_model_sha256 = "ab12"
	policy = ACTBBoxPolicy(config)
	policy.eval()
	policy.save_pretrained(tmp_path)

	loaded = ACTBBoxPolicy.from_pretrained(tmp_path)
	loaded.eval()
	assert loaded.config.bbox_names == BBOX_NAMES
	assert loaded.config.bbox_model_sha256 == "ab12"
	batch = make_batch(config)
	torch.testing.assert_close(loaded.predict_action_chunk(batch), policy.predict_action_chunk(batch))


def test_missing_env_state_message():
	policy = ACTBBoxPolicy(make_config())
	policy.eval()
	batch = make_batch(policy.config)
	batch.pop(OBS_ENV_STATE)
	with pytest.raises(KeyError, match="lerobot-rollout-bbox"):
		policy.predict_action_chunk(batch)


def test_preprocessor_keeps_bbox_values():
	# 검출률이 높은 데이터셋의 is_detected 통계(평균 0.995, 표준편차 0.07)로 정규화하면 0이 -14가 된다.
	stats = {
		OBS_STATE: {"mean": torch.ones(6), "std": torch.full((6,), 2.0)},
		OBS_ENV_STATE: {"mean": torch.full((14,), 0.995), "std": torch.full((14,), 0.07)},
		ACTION: {"mean": torch.zeros(6), "std": torch.ones(6)},
		IMAGE_KEY: {"mean": torch.zeros(3, 1, 1), "std": torch.ones(3, 1, 1)},
	}
	preprocessor, _ = make_pre_post_processors(make_config(), dataset_stats=stats)
	bbox = torch.zeros(14)
	bbox[:4] = torch.tensor([0.1, 0.2, 0.3, 0.4])
	processed = preprocessor({OBS_STATE: torch.ones(6), OBS_ENV_STATE: bbox.clone(), IMAGE_KEY: torch.rand(3, 64, 64)})
	torch.testing.assert_close(processed[OBS_ENV_STATE], bbox.unsqueeze(0))
	torch.testing.assert_close(processed[OBS_STATE], torch.zeros(1, 6))
