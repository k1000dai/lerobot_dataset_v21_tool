from typing import Any

# Define constants for HSR
GRIPPER_OPEN_ACTION = 1
GRIPPER_CLOSE_ACTION = 0

# Define the features of the transformed dataset
hsr_features: dict[str, dict[str, Any]] = {
    "observation.image.head": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channel"],
        "info": {
            "video.fps": 10,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": "false",
            "has_audio": "false",
        },
    },
    "observation.image.hand": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channel"],
        "info": {
            "video.fps": 10,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": "false",
            "has_audio": "false",
        },
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (8,),
        "names": [
            "arm_lift_joint",
            "arm_flex_joint",
            "arm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
            "hand_motor_joint",
            "head_pan_joint",
            "head_tilt_joint",
        ],
    },
    "observation.wrist.wrench": {
        "dtype": "float32",
        "shape": (6,),
        "names": [
            "force_x",
            "force_y",
            "force_z",
            "torque_x",
            "torque_y",
            "torque_z",
        ],
        "description": "Wrist wrench data (force and torque) flattened",
    },
    "observation.end_effector_pose.absolute": {
        "shape": (6,),
        "dtype": "float32",
        "names": ["x", "y", "z", "roll", "pitch", "yaw"],
    },
    "observation.end_effector_pose.relative": {
        "shape": (6,),
        "dtype": "float32",
        "names": ["x", "y", "z", "roll", "pitch", "yaw"],
    },
    "action.absolute": {
        "dtype": "float32",
        "shape": (8,),
        "names": [
            "arm_lift_joint",
            "arm_flex_joint",
            "arm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
            "hand_motor_joint",
            "head_pan_joint",
            "head_tilt_joint",
        ],
        "description": "absolute action for all joints without hand_motor_joint(gripper)",
    },
    "action.relative": {
        "dtype": "float32",
        "shape": (11,),
        "names": [
            "arm_lift_joint",
            "arm_flex_joint",
            "arm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
            "hand_motor_joint",
            "head_pan_joint",
            "head_tilt_joint",
            "base_x",
            "base_y",
            "base_t",
        ],
        "description": "delta action for all joints and base without hand_motor_joint(gripper)",
    },
    "action.arm": {
        "dtype": "float32",
        "shape": (5,),
        "names": [
            "arm_lift_joint",
            "arm_flex_joint",
            "arm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
        ],
        "description": "absolute action for arm joints",
    },
    "action.gripper": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["hand_motor_joint"],
        "description": "absolute action for gripper",
    },
    "action.head": {
        "dtype": "float32",
        "shape": (2,),
        "names": ["head_pan_joint", "head_tilt_joint"],
        "description": "absolute action for head joints",
    },
    "action.base": {
        "dtype": "float32",
        "shape": (3,),
        "names": ["base_x", "base_y", "base_t"],
        "description": "delta action for base",
    },
    # fresh mask to indicate whether the action is a fresh action or paded former action
    "observation.image.head.is_fresh": {
        "dtype": "bool",
        "shape": (
            3,
            1,
            1,
        ),  # NOTE: need to be [3,1,1] to be compatible with lerobot assertion check `_assert_type_and_shape` in `compute_stats.py`
        "names": None,
    },
    "observation.image.hand.is_fresh": {
        "dtype": "bool",
        "shape": (
            3,
            1,
            1,
        ),  # NOTE: need to be [3,1,1] to be compatible with lerobot assertion check `_assert_type_and_shape` in `compute_stats.py`
        "names": None,
    },
    "observation.state.is_fresh": {
        "dtype": "bool",
        "shape": (8,),
        "names": None,
    },
    "action.absolute.is_fresh": {
        "dtype": "bool",
        "shape": (8,),
        "names": None,
    },
    "action.relative.is_fresh": {
        "dtype": "bool",
        "shape": (11,),
        "names": None,
    },
    "action.arm.is_fresh": {
        "dtype": "bool",
        "shape": (5,),
        "names": None,
    },
    "action.gripper.is_fresh": {
        "dtype": "bool",
        "shape": (1,),
        "names": None,
    },
    "action.head.is_fresh": {
        "dtype": "bool",
        "shape": (2,),
        "names": None,
    },
    "action.base.is_fresh": {
        "dtype": "bool",
        "shape": (3,),
        "names": None,
    },
    "episode_index": {
        "dtype": "int64",
        "shape": (1,),
        "names": None,
    },
    "frame_index": {
        "dtype": "int64",
        "shape": (1,),
        "names": None,
    },
    "timestamp": {
        "dtype": "float32",
        "shape": (1,),
        "names": None,
    },
    "next.done": {
        "dtype": "bool",
        "shape": (1,),
        "names": None,
    },
    "index": {
        "dtype": "int64",
        "shape": (1,),
        "names": None,
    },
    "short_horizon_task_index": {
        "dtype": "int64",
        "shape": (1,),
        "names": None,
    },
    "primitive_action_index": {
        "dtype": "int64",
        "shape": (1,),
        "names": None,
    },
    "success_primitive_action": {
        "dtype": "bool",
        "shape": (1,),
        "names": None,
    },
}
