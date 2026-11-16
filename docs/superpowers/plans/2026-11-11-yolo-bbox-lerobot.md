# YOLO bbox LeRobot 통합 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** LeRobot 0.6.0에서 YOLO bbox를 녹화(`lerobot-record-bbox`), 학습(`lerobot-train --policy.type=act_bbox`), 추론(`lerobot-rollout-bbox`)까지 끊김 없이 쓴다.

**Architecture:** 이 저장소를 LeRobot 정책 플러그인 패키지 `lerobot_policy_act_bbox`로 만든다. 관측 처리 단계 `YoloBBoxStep`이 매 제어 주기 카메라 프레임에서 bbox를 계산해 `observation.environment_state`로 넘기고, 녹화와 추론이 같은 단계를 쓴다. LeRobot 본체는 고치지 않는다. CLI 래퍼가 `record()`와 `build_rollout_context()`에 처리 단계를 넘기고, LeRobot 0.6.0에 훅이 없는 두 지점(특징 합치기, 에피소드 경계)만 감싼다.

**Tech Stack:** Python 3.12, LeRobot 0.6.0(`~/lerobot`, editable), Ultralytics 8.4.138, PyTorch 2.11, conda env `lerobot`, pytest

**Spec:** 사용자 요청(2026-11-11)과 [기획서](../../yolo_act_proposal.md) §5 bbox 정의
> `lerobot-train --dataset.repo_id=${HF_USER}/your_dataset --policy.type=act_bbox ...` 이런 느낌으로 학습. 입력은 영상과 bbox 좌표. `lerobot-record --robot.type=so101_follower`에 bbox 옵션을 추가하면 YOLO 모델을 불러와 bbox 좌표를 함께 기록. 금요일까지 데이터셋 수집부터 모델 학습, 추론까지.

## 설계 결정 (2026-11-11, 작성자 판단 — 사용자 부재로 확인 못 함)

- **별도 명령:** `lerobot-record-bbox`와 `lerobot-rollout-bbox`는 원래 명령과 옵션이 같고 `--bbox.*`만 더합니다. 이유는 두 가지입니다.
  - LeRobot 0.6.0 CLI에는 관측 처리 단계를 넣을 훅이 없습니다.
  - 본체(`~/lerobot`)는 다른 conda 환경과 프로젝트도 함께 씁니다.
- **추론:** `lerobot-rollout`으로 합니다. 0.6.0의 `lerobot-record`는 수집 전용이라 정책 인자가 없습니다.
- **학습:** 수정하지 않은 `lerobot-train`을 씁니다. `lerobot_policy_*` 패키지는 자동으로 탐색됩니다.
- **bbox 정의:** 키는 `observation.environment_state`(ENV 타입)입니다. 나머지는 기획서 §5를 그대로 따릅니다.
  - 카메라마다 `x1,y1,x2,y2,cx,cy,is_detected` 7개 값을 0~1로 정규화해 넣습니다.
  - 목표 클래스 중 confidence가 가장 높은 박스 1개를 고르고, conf 기본값은 0.3입니다.
  - 검출이 안 되면 마지막 값을 유지하고 `is_detected=0`으로 둡니다. 첫 검출 전에는 0이고, 에피소드마다 초기화합니다.
- **act_bbox:** ACT와 다른 점은 세 가지입니다.
  1. bbox 입력이 반드시 있어야 합니다.
  2. bbox 입력층을 `Linear` 대신 `2x-1 → Linear → GELU → Linear`로 바꿉니다. ENV는 IDENTITY로 정규화합니다. 검출률이 높으면 `is_detected`의 표준편차가 0에 가까워져서 MEAN_STD 정규화가 폭발하기 때문입니다.
  3. 학습 데이터셋의 bbox 이름과 YOLO 설정을 config에 저장합니다. 롤아웃 때 이를 기본값으로 쓰고, 값이 다르면 알립니다.
- **YOLO 설정 저장 위치:** 녹화할 때 `features["observation.environment_state"]["info"]["bbox_detector"]`에 기록해 데이터셋과 함께 다니게 합니다.
- **실험별 데이터셋:** 세 실험 모두 한 번의 녹화에서 만듭니다.
  - 실험 1: `observation.environment_state`를 제거한 파생 데이터셋으로 `--policy.type=act`를 학습합니다.
  - 실험 2: 영상을 제거한 파생 데이터셋으로 act_bbox를 학습합니다.
  - 실험 3: 원본 데이터셋으로 act_bbox를 학습합니다.
- **trackio:** LeRobot 0.6.0이 지원하지 않습니다. README에서 `--wandb.enable=true`를 안내합니다.
- **LeKiwi:**
  - ROS 호스트에 `lekiwi_client`로 붙으면 프레임이 BGR이므로 `--bbox.is_bgr_input=true`가 필요합니다.
  - `lerobot-rollout`은 `.pos` 키만 정책에 연결하므로 바퀴를 구동하지 못합니다. LeRobot의 한계라 README에 적습니다.

## Global Constraints

- **LeRobot 본체:** `~/lerobot`는 수정하지 않습니다.
- **설치:** 새 외부 의존성은 설치하지 않습니다. 우리 패키지만 conda env `lerobot`에 `pip install -e . --no-deps --no-build-isolation`으로 설치합니다.
- **이름 고정:** 아래 이름은 LeRobot이 이름 규칙으로 찾으므로 바꾸면 안 됩니다.
  - 배포·임포트 이름 `lerobot_policy_act_bbox`, 정책 타입 `act_bbox`
  - 클래스 `ACTBBoxConfig`, `ACTBBoxPolicy`
  - 함수 `make_act_bbox_pre_post_processors`
  - 파일 `configuration_act_bbox.py`, `modeling_act_bbox.py`, `processor_act_bbox.py`
- **콘솔 명령:** `lerobot-record-bbox`, `lerobot-rollout-bbox`
- **bbox 특징 이름:** `{camera}_{x1|y1|x2|y2|cx|cy|is_detected}` 형식이고, 순서는 카메라 순서 다음에 필드 순서입니다.
- **YOLO 입력 색:** YOLO에는 BGR numpy를 넘깁니다. LeRobot 카메라는 RGB를 주므로 뒤집습니다. `is_bgr_input=True`이면 그대로 넘깁니다.
- **규칙서:** 탭 들여쓰기, Boolean은 `is_`/`has_`로 시작, 함수 타입 힌트, Google docstring, 절대 경로 하드코딩 금지를 지킵니다.
- **시험:** 실제 로봇을 움직이지 않습니다. 가짜 장치(`tests/fake_hw.py`)만 씁니다.
- **커밋:** 하지 않습니다. 사용자가 요청하지 않았고 git 저장소도 아닙니다. 그래서 각 Task는 "전체 시험 통과 확인"으로 끝납니다.
- **이름 금지:** 코드와 문서에 이전 직장의 내부 이름을 쓰지 않습니다.

## Review Focus

1. **색 옵션 누락:** LeKiwi `lekiwi_client`와 ROS 호스트 조합에서 `--bbox.is_bgr_input`을 빼먹으면 검출이 조용히 나빠집니다. 에피소드마다 카메라별 검출률이 로그로 보여야 합니다. 시험: Task 1의 `test_summary_reports_rate_and_latency`
2. **실험 1 오염:** 같은 녹화로 `--policy.type=act`를 돌리면 ACT가 bbox를 조용히 씁니다. README에 파생 데이터셋 명령을 넣습니다. 시험: Task 3 통합 시험의 env_state 제거 확인
3. **카메라 불일치:** 롤아웃 때 카메라 이름이나 순서가 학습 때와 다르면 정책이 엉뚱한 입력을 받습니다. 시험: Task 4의 `test_resolve_camera_mismatch_raises`
4. **YOLO 가중치 변경:** 학습 뒤 가중치가 바뀌면 bbox 분포가 달라집니다. 시험: Task 4의 `test_resolve_sha_mismatch_warns`
5. **일반 롤아웃으로 실행:** act_bbox를 일반 `lerobot-rollout`으로 돌리면 bbox가 없습니다. 원인을 알려 주는 오류가 나야 합니다. 시험: Task 2의 `test_missing_env_state_message`, Task 4 통합 시험

---

### Task 1: 패키지 골격과 bbox 처리 단계

**Files:**
- Create: `pyproject.toml`, `src/lerobot_policy_act_bbox/__init__.py`, `src/lerobot_policy_act_bbox/bbox.py`, `tests/test_bbox.py`

**Interfaces (Produces):**
- `BBOX_FIELDS = ("x1", "y1", "x2", "y2", "cx", "cy", "is_detected")`, `DEFAULT_CONF = 0.3`
- `bbox_feature_names(cameras: list[str]) -> list[str]`
- `@dataclass BBoxConfig`: 모든 필드의 기본값이 None입니다. 녹화에서는 기본값을 채우고, 롤아웃에서는 정책에 저장된 값을 씁니다.
  - `model: str | None`
  - `cameras: list[str] | None`
  - `class_name: str | None`
  - `conf: float | None`
  - `device: str | None`
  - `is_bgr_input: bool | None`
- `file_sha256(path: str) -> str`
- `pick_box(classes: np.ndarray, confs: np.ndarray, boxes_xyxyn: np.ndarray, class_id: int, min_conf: float) -> np.ndarray | None`
- `class YoloDetector`
  - `__init__(self, model: Any, class_id: int, conf: float, is_bgr_input: bool, device: str | None = None)`
  - `__call__(self, frames: list[np.ndarray]) -> list[np.ndarray | None]`: 카메라마다 정규화된 xyxy `float32 (4,)` 또는 None을 돌려줍니다. 한 번의 `model.predict(list)`로 처리합니다.
- `load_detector(cfg: BBoxConfig) -> tuple[YoloDetector, dict]`
  - Ultralytics를 이 함수 안에서만 import합니다. 플러그인은 모든 LeRobot 명령이 import하기 때문입니다.
  - `class_name`이 None이면 모델 클래스가 1개일 때만 허용하고, 아니면 `ValueError`를 냅니다.
  - settings는 `{"model": 절대경로, "model_sha256", "class_name", "conf", "is_bgr_input"}`입니다.
- `@dataclass class YoloBBoxStep(ObservationProcessorStep)`
  - 필드 `detector: Callable[[list[np.ndarray]], list[np.ndarray | None]]`, `cameras: list[str]`
  - `observation()`은 bbox 이름 키를 더합니다.
  - `reset()`은 유지값과 통계를 비웁니다.
  - `summary() -> str | None`은 카메라별 검출률과 YOLO 지연 p50/p95를 돌려줍니다.
  - `transform_features()`는 입력을 그대로 돌려줍니다. bbox는 `observation.state`가 아니라 별도 특징으로 넣기 때문입니다.
- `make_observation_pipeline(step: YoloBBoxStep) -> RobotProcessorPipeline`: `observation_to_transition`과 `transition_to_observation`을 씁니다.
- `bbox_dataset_feature(cameras: list[str], settings: dict) -> dict[str, dict]`: 반환값은 `{OBS_ENV_STATE: {"dtype": "float32", "shape": (7*n,), "names": ..., "info": {"bbox_detector": {**settings, "cameras": cameras}}}}`입니다.

- [ ] **Step 1: 실패하는 시험 작성** (`tests/test_bbox.py`)
  - `test_feature_names_order`: `bbox_feature_names(["front","wrist"])`의 처음 7개가 `["front_x1","front_y1","front_x2","front_y2","front_cx","front_cy","front_is_detected"]`이고 길이는 14
  - `test_step_writes_box_and_center`: 탐지기가 `[array([.1,.2,.3,.4]), None]`을 내면 `front_cx==0.2`, `front_cy==0.3`, `front_is_detected==1.0`, wrist 7개는 모두 0.0
  - `test_step_holds_last_box_on_miss`: 박스 A 다음에 None이 오면 front 좌표는 A이고 `front_is_detected==0.0`
  - `test_reset_clears_hold`: reset 뒤 None이 오면 좌표가 0
  - `test_missing_camera_raises`: 관측에 카메라 키가 없으면 `KeyError`가 나고 메시지에 있는 키 목록이 들어감
  - `test_pick_box_highest_conf_of_class`: cls `[0,1,0]`, conf `[.5,.9,.8]`, class 0, min 0.3이면 2번째 행을 돌려주고, 모두 min 미만이면 None
  - `test_detector_flips_rgb_to_bgr`: 가짜 모델이 받은 입력의 채널이 뒤집힘. `is_bgr_input=True`이면 그대로
  - `test_summary_reports_rate_and_latency`: 4번 호출해 front에서 3번 검출되면 `"front 75%"`와 `"ms"`가 들어감
  - `test_dataset_feature`: shape `(14,)`, names 일치, `info.bbox_detector.cameras == ["front","wrist"]`
- [ ] **Step 2:** `~/miniconda3/envs/lerobot/bin/python -m pytest tests/test_bbox.py -v` → FAIL (ModuleNotFoundError)
- [ ] **Step 3:** `pyproject.toml`(setuptools, src 레이아웃, 의존성 없음)과 `bbox.py`를 구현하고 `pip install -e . --no-deps --no-build-isolation`으로 설치
- [ ] **Step 4:** 같은 명령 → PASS. `python -c "import lerobot_policy_act_bbox"` 성공

### Task 2: act_bbox 정책 플러그인

**Files:**
- Create: `src/lerobot_policy_act_bbox/configuration_act_bbox.py`, `modeling_act_bbox.py`, `processor_act_bbox.py`, `tests/test_policy.py`
- Modify: `src/lerobot_policy_act_bbox/__init__.py`(설정, 정책, 처리기를 export해 등록)

**Interfaces:**
- Consumes: Task 1 `BBOX_FIELDS`
- Produces:
  - `ACTBBoxConfig(ACTConfig)`, `@PreTrainedConfig.register_subclass("act_bbox")`
    - 추가 필드: `bbox_names: list[str]`, `bbox_cameras: list[str]`, `bbox_model: str = ""`, `bbox_model_sha256: str = ""`, `bbox_class_name: str = ""`, `bbox_conf: float = 0.0`, `bbox_is_bgr_input: bool = False`
    - `normalization_mapping` 기본값에 `"ENV": IDENTITY`를 더합니다.
    - `set_dataset_feature_metadata(self, features: dict[str, dict]) -> None`: `OBS_ENV_STATE`가 없으면 `ValueError`를 냅니다. names와 `info.bbox_detector`를 복사합니다.
    - `validate_features()`: 부모 검사에 더해 env_state를 필수로 하고, names가 있으면 개수가 env 차원과 같은지 봅니다.
  - `BBoxEncoder(nn.Module)`: `__init__(self, in_dim: int, dim_model: int)`, `forward(x: Tensor) -> Tensor`
  - `ACTBBoxPolicy(ACTPolicy)`: `config_class = ACTBBoxConfig`, `name = "act_bbox"`
    - `__init__`에서 `model.encoder_env_state_input_proj`를 `BBoxEncoder`로 바꿉니다.
    - `predict_action_chunk`에 env_state가 없으면 `"lerobot-rollout-bbox"`를 안내하는 `KeyError`를 냅니다.
  - `make_act_bbox_pre_post_processors(config, dataset_stats=None)`: `make_act_pre_post_processors`에 넘깁니다.

시험용 작은 설정: `dim_model=64, n_heads=2, dim_feedforward=128, n_encoder_layers=1, n_vae_encoder_layers=1, chunk_size=10, n_action_steps=10, pretrained_backbone_weights=None`, 영상 `(3,64,64)`, state 6, env 14, action 6, CPU

- [ ] **Step 1: 실패하는 시험 작성** (`tests/test_policy.py`)
  - `test_registered`: `get_choice_class("act_bbox") is ACTBBoxConfig`, `get_policy_class("act_bbox") is ACTBBoxPolicy`, `make_pre_post_processors(cfg, dataset_stats=...)`가 파이프라인 2개를 돌려줌
  - `test_requires_env_state`: env 특징이 없으면 `ValueError`이고 메시지에 `"observation.environment_state"`가 들어감
  - `test_reads_dataset_metadata`: names와 info를 담은 features를 넣으면 `bbox_names`, `bbox_cameras`, `bbox_model`, `bbox_model_sha256`이 채워짐
  - `test_forward_backward_with_images`, `test_forward_without_images`: loss가 유한하고 BBoxEncoder 파라미터에 grad가 생김
  - `test_bbox_encoder_replaces_linear`: `isinstance(policy.model.encoder_env_state_input_proj, BBoxEncoder)`
  - `test_save_load_roundtrip`: `save_pretrained`와 `from_pretrained`를 거쳐도 `bbox_names`가 같고, eval에서 `predict_action_chunk` 출력이 같음
  - `test_missing_env_state_message`: env_state 없이 `predict_action_chunk`를 부르면 `KeyError`에 `"lerobot-rollout-bbox"`가 들어감
- [ ] **Step 2:** `pytest tests/test_policy.py -v` → FAIL
- [ ] **Step 3:** 구현
- [ ] **Step 4:** `pytest tests -v` → 모두 PASS

### Task 3: lerobot-record-bbox, 가짜 장치, 녹화 통합 시험

**Files:**
- Create: `src/lerobot_policy_act_bbox/lerobot_hooks.py`, `record.py`, `tests/fake_hw.py`, `tests/test_cli.py`
- Modify: `pyproject.toml`(스크립트 `lerobot-record-bbox = "lerobot_policy_act_bbox.record:main"`)

**Interfaces:**
- Consumes: Task 1 `BBoxConfig`, `load_detector`, `YoloBBoxStep`, `make_observation_pipeline`, `bbox_dataset_feature`
- Produces:
  - `with_dataset_feature(module: ModuleType, feature: dict[str, dict]) -> ContextManager[None]`: `module.combine_feature_dicts`를 감싸 **결과 뒤에** feature를 붙입니다. 원래 함수에 넣으면 `info`가 지워지기 때문입니다.
  - `with_record_loop_hook(module: ModuleType, step: YoloBBoxStep) -> ContextManager[None]`: `module.record_loop` 호출마다 시작 때 `step.reset()`을 부르고, 끝나면 `step.summary()`를 로그에 남깁니다. `dataset` 인자가 있으면 "episode", 없으면 "reset"으로 표시합니다.
  - `RecordBBoxConfig(RecordConfig)`: `bbox: BBoxConfig`
  - `resolve_record_cameras(cfg: RecordBBoxConfig) -> list[str]`: `cfg.bbox.cameras`가 있으면 쓰고, 없으면 `cfg.robot.cameras`의 키를 씁니다. 둘 다 비면 `ValueError`
  - `record_bbox(cfg)`(`@parser.wrap()`): `lerobot_record.record.__wrapped__(cfg, robot_observation_processor=...)`를 호출합니다.
  - `main()`: `register_third_party_plugins()` 다음에 `record_bbox()`
  - `tests/fake_hw.py`: `--robot.discover_packages_path=fake_hw`로 불러옵니다.
    - `BBoxReplayRobotConfig` / `BBoxReplayRobot`("bbox_replay_robot")
      - 설정: `frames_dir: str`, `camera_names: list[str]`, `action_log: str`
      - 관측: 이미지를 RGB로 순환하며 내고, 관절 6개를 냅니다. `shoulder_pan.pos`에는 이미지 번호를 넣습니다.
      - `send_action`: action_log에 한 줄씩 씁니다.
    - `BBoxReplayTeleopConfig` / `BBoxReplayTeleop`("bbox_replay_teleop"): 사인파 관절 목표를 냅니다.

- [ ] **Step 1: 실패하는 시험 작성** (`tests/test_cli.py::test_record_writes_bbox`)
  - CUDA나 가중치 `~/workspace/YOLO/outputs/runs/box-2/weights/best.pt`가 없으면 skip
  - 프레임: `~/workspace/YOLO/dataset/train/images` 앞쪽 30장을 tmp로 복사
  - 녹화 명령: `lerobot-record-bbox`, 2 에피소드 × 2초, reset 0.5초, `--play_sounds=false --dataset.push_to_hub=false`
  - 확인할 것:
    - info.json의 env 특징: shape `[14]`, names, `info.bbox_detector.model_sha256 == file_sha256(weights)`
    - env 값이 모두 [0,1]이고, `is_detected`가 1인 행이 적어도 하나 있음
    - `front_is_detected==1`인 행의 front 박스가 `shoulder_pan.pos`번 이미지를 오프라인 `YoloDetector`로 돌린 결과와 1e-4 안에서 같음
    - `lerobot-edit-dataset remove_feature`로 영상을 지워도 env 특징과 info가 남음
    - env를 지우면 env 특징이 없음
- [ ] **Step 2:** `pytest tests/test_cli.py -v -k record` → FAIL
- [ ] **Step 3:** 구현하고 다시 설치
- [ ] **Step 4:** `pytest tests -v` → 모두 PASS

### Task 4: lerobot-rollout-bbox와 학습·추론 통합 시험

**Files:**
- Create: `src/lerobot_policy_act_bbox/rollout.py`, `tests/test_rollout_settings.py`
- Modify: `lerobot_hooks.py`, `tests/test_cli.py`, `pyproject.toml`(스크립트 `lerobot-rollout-bbox`)

**Interfaces:**
- Consumes: Task 1~3 전부, Task 2 `ACTBBoxConfig` 필드
- Produces:
  - `RolloutBBoxConfig(RolloutConfig)`: `bbox: BBoxConfig`
  - `resolve_rollout_bbox(policy_cfg: ACTBBoxConfig, override: BBoxConfig) -> tuple[BBoxConfig, list[str]]`
    - 정책에 저장된 값이 기본이고, CLI 값이 그 위를 덮습니다.
    - 두 값이 다르면 경고 문자열을 만듭니다.
    - 모델 파일의 sha256이 저장값과 다르면 경고합니다.
    - `bbox_feature_names(cameras) != policy_cfg.bbox_names`이면 `ValueError`, 모델이 없어도 `ValueError`
  - `with_rollout_hooks(rollout_module, context_module, pipeline, step, feature) -> ContextManager[None]`
    - `context_module.combine_feature_dicts`에 feature를 붙입니다.
    - `rollout_module.build_rollout_context`에 `robot_observation_processor=pipeline`을 넘기고, `ctx.policy.inference.reset`에 `step.reset`을 연결합니다.
  - `rollout_bbox(cfg)`(`@parser.wrap()`)
    - 정책 타입이 `act_bbox`가 아니면 `ValueError`, RTC 추론이면 `ValueError`
    - 끝나면 `step.summary()`를 로그에 남깁니다.
  - `main()`

- [ ] **Step 1: 실패하는 시험 작성**
  - **단위 시험** (`tests/test_rollout_settings.py`)
    - `test_resolve_uses_policy_settings`
    - `test_resolve_override_warns`: conf를 다르게 주면 경고 1개
    - `test_resolve_camera_mismatch_raises`
    - `test_resolve_sha_mismatch_warns`: tmp 파일 사용
  - **통합 시험** (`tests/test_cli.py::test_train_and_rollout`): Task 3의 녹화 데이터셋을 모듈 fixture로 씁니다.
    1. `lerobot-train --policy.type=act_bbox --steps=20 --batch_size=2 --num_workers=0 --save_freq=20 --policy.push_to_hub=false --wandb.enable=false`
       - `checkpoints/last/pretrained_model/config.json`의 `type == "act_bbox"`, `bbox_names` 길이 14, sha가 일치
    2. 영상을 지운 파생 데이터셋으로 같은 명령
       - safetensors 키에 `backbone`이 없음
    3. `lerobot-rollout-bbox --policy.path=... --robot.type=bbox_replay_robot --duration=3 --return_to_initial_position=false`
       - exit 0, action_log 30줄 이상, 출력에 bbox 요약
    4. 같은 정책을 일반 `lerobot-rollout`으로 실행
       - exit가 0이 아니고 출력에 `"lerobot-rollout-bbox"`가 들어감
    5. `--strategy.type=episodic --dataset.repo_id=local/rollout_bbox_test` 1 에피소드
       - 롤아웃 데이터셋에 env 특징과 info가 있음
- [ ] **Step 2:** `pytest tests -v -k "resolve or rollout"` → FAIL
- [ ] **Step 3:** 구현하고 다시 설치
- [ ] **Step 4:** `pytest tests -v` → 모두 PASS

### Task 5: 문서와 실기 점검

**Files:**
- Modify: `README.md`(지금 비어 있음), `docs/yolo_act_proposal.md`(맨 위 변경 안내와 §11 결정 기록), `docs/yolo_act_plan.md`(맨 위 변경 안내)

- [ ] **Step 1: README 작성.** 담을 내용은 다음과 같습니다.
  - 설치
  - 녹화: SO-101 예시, LeKiwi 예시(`--bbox.is_bgr_input=true`)
  - 학습: 실험 1·2·3 명령과 파생 데이터셋 만드는 명령
  - 추론
  - bbox 정의
  - 주의: trackio 미지원, LeKiwi 바퀴 미지원, 실험 1 오염
  - 시험 실행법
- [ ] **Step 2: 실기 점검.** 로봇은 정지 상태로 두고 관측만 받습니다.
  - 로봇의 ZMQ 접속 상태를 확인합니다.
  - LeKiwi 관측 스트림 10초 동안 `YoloDetector`로 카메라별 검출률과 지연을 잽니다. 스크립트는 스크래치에 둡니다.
- [ ] **Step 3: 전체 시험.** `pytest tests -v` → 모두 PASS
- [ ] **Step 4: 독립 리뷰.** 새 리뷰어(code-reviewer)에게 맡기고, 지적을 반영한 뒤 다시 시험합니다.
