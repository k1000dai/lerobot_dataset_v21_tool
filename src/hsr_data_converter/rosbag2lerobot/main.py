"""
Main entry point for HSR rosbag to lerobot conversion
"""

from pathlib import Path
from typing import List

from lerobot.configs import parser

from hsr_data_converter.convert_config import ConvertConfig

from .aws_handler import handle_aws_upload, setup_aws_environment
from .core_converter import perform_conversion

_extract_topics: List[str] = [
    "/hsrb/arm_trajectory_controller/command",
    "/hsrb/head_trajectory_controller/command",
    "/hsrb/gripper_controller/command",
    "/hsrb/gripper_controller/grasp/goal",
    "/hsrb/command_velocity",
    "/control_mode",
    "/hsrb/joint_states",
    "/hsrb/hand_camera/image_raw/compressed",
    "/hsrb/head_rgbd_sensor/rgb/image_rect_color/compressed",
    "/tf",
    "/tf_static",
    "/hsrb/wrist_wrench/raw",
]


@parser.wrap()
def convert_hsr_rosbag_to_lerobot_format(cfg: ConvertConfig):
    """
    Convert HSR rosbag datasets to LeRobotDatasetV2 format (action as delta joint except gripper)
    specified by the configuration cfg

    Directory structure specified by cfg.raw_dir:

    ├ raw_dir
    │ ├ 20250308_rosbag
    │ │ ├ ...bag
    │ │ └ meta.json
    │ ├ 20250309_rosbag
    │ │ ├ ...bag
    │ │ └ meta.json
    │ └ 20250310_rosbag
    │   ├ ...bag
    │   └ meta.json
    ...

    Parameters
    ----------
    cfg : ConvertConfig
        configuration for the conversion

    Returns
    -------
    LeRobotDataset
        the converted dataset

    """
    # Handle AWS preprocessing if needed
    if cfg.use_aws:
        raw_dir, out_dir, cleanup_func = setup_aws_environment(cfg)
    else:
        raw_dir = Path(cfg.raw_dir)
        out_dir = Path(cfg.out_dir)
        cleanup_func = None

    try:
        # Perform the actual conversion
        result = perform_conversion(cfg, raw_dir, out_dir)

        # Handle AWS postprocessing if needed
        if cfg.use_aws:
            handle_aws_upload(cfg, out_dir)

        return result

    finally:
        # Cleanup if needed
        if cleanup_func:
            cleanup_func()


if __name__ == "__main__":
    convert_hsr_rosbag_to_lerobot_format()
