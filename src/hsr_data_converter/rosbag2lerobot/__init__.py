"""
HSR rosbag to lerobot dataset conversion module

This module provides functionality to convert HSR rosbag datasets
to LeRobotDatasetV2 format with action as delta joint (except gripper).
"""

from .main import convert_hsr_rosbag_to_lerobot_format

__all__ = ["convert_hsr_rosbag_to_lerobot_format"]
