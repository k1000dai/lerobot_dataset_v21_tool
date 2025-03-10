"""
Action calculation functions for rosbag to lerobot conversion
"""

from typing import Any, Dict, Tuple

import torch

from hsr_data_converter.hsr_utils import (
    GRIPPER_CLOSE_ACTION,
    GRIPPER_OPEN_ACTION,
)


def calc_initial_subaction(frame: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate initial action which is not received and contained in frame yet

    Parameters
    ----------
    frame : Dict[str, Any]
        The current frame

    Returns
    -------
    Dict[str, Any]
        Initial action for the frame
    """
    initial_subaction = {}
    # Only calculate initial action if the action is not received yet
    if frame["action.arm"] is None:
        # use the current state as the initial action
        initial_subaction["action.arm"] = frame["observation.state"][:5]
        initial_subaction["action.arm.is_fresh"] = torch.full([5], False)
    if frame["action.gripper"] is None:
        # determine gripper close or open from the current state:  state[5] > 0: open, state[5] <= 0: close
        if frame["observation.state"][5] > 0:
            initial_subaction["action.gripper"] = torch.tensor(
                [GRIPPER_OPEN_ACTION], dtype=torch.float32
            )
        else:
            initial_subaction["action.gripper"] = torch.tensor(
                [GRIPPER_CLOSE_ACTION], dtype=torch.float32
            )
        initial_subaction["action.gripper.is_fresh"] = torch.full([1], False)
    if frame["action.head"] is None:
        # use the current state as the initial action
        initial_subaction["action.head"] = frame["observation.state"][6:8]
        initial_subaction["action.head.is_fresh"] = torch.full([2], False)
    if frame["action.base"] is None:
        # set zeros as the initial action
        initial_subaction["action.base"] = torch.tensor(
            [0.0, 0.0, 0.0], dtype=torch.float32
        )
        initial_subaction["action.base.is_fresh"] = torch.full([3], False)

    return initial_subaction


def calc_delta_action(frame: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate delta action except gripper from the current action and the current state

    Parameters
    ----------
    frame : Dict[str, Any]
        The current frame

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The tuple of delta action (gripper action is absolute action) and is_fresh.
    """
    delta_arm_action = torch.tensor(
        frame["action.arm"] - frame["observation.state"][:5], dtype=torch.float32
    )
    gripper_action = frame["action.gripper"]
    delta_head_action = torch.tensor(
        frame["action.head"] - frame["observation.state"][6:8], dtype=torch.float32
    )
    delta_base_action = frame["action.base"]

    delta_action = torch.cat(
        [delta_arm_action, gripper_action, delta_head_action, delta_base_action]
    )
    delta_is_fresh = torch.cat(
        [
            frame["action.arm.is_fresh"] & frame["observation.state.is_fresh"][:5],
            frame["action.gripper.is_fresh"],
            frame["action.head.is_fresh"] & frame["observation.state.is_fresh"][6:8],
            frame["action.base.is_fresh"],
        ]
    )
    return delta_action, delta_is_fresh


def calc_absolute_action(frame: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate absolute action except gripper from the current action

    Parameters
    ----------
    frame : Dict[str, Any]
        The current frame

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The tuple of absolute action and is_fresh.
    """
    abs_arm_action = torch.tensor(frame["action.arm"], dtype=torch.float32)
    abs_gripper_action = torch.tensor(frame["action.gripper"], dtype=torch.float32)
    abs_head_action = torch.tensor(frame["action.head"], dtype=torch.float32)

    abs_action = torch.cat([abs_arm_action, abs_gripper_action, abs_head_action])
    abs_is_fresh = torch.cat(
        [
            frame["action.arm.is_fresh"],
            frame["action.gripper.is_fresh"],
            frame["action.head.is_fresh"],
        ]
    )
    return abs_action, abs_is_fresh
