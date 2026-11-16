"""LeRobot 플러그인: YOLO bbox를 입력으로 받는 ACT(act_bbox)와 bbox 녹화·롤아웃 명령."""

try:
	import lerobot  # noqa: F401
except ImportError as error:
	raise ImportError("lerobot 이 설치된 환경에서 쓰세요 (conda env lerobot).") from error

from .configuration_act_bbox import ACTBBoxConfig
from .modeling_act_bbox import ACTBBoxPolicy
from .processor_act_bbox import make_act_bbox_pre_post_processors

__all__ = ["ACTBBoxConfig", "ACTBBoxPolicy", "make_act_bbox_pre_post_processors"]
