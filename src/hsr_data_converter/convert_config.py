from dataclasses import dataclass


@dataclass
class ConvertConfig:
    raw_dir: str = ""  # Path to the directory containing the rosbag files
    out_dir: str = ""  # Path to the directory where the lerobot dataset will be saved
    fps: int = 10  # fps of the extracted dataset
    conversion_type: str = "individual"  # "individual" or "aggregate"
    repo_id: str = "hsr"  # repository ID for the dataset
    robot_type: str = "hsr"  # robot type
    separate_per_primitive: bool = (
        False  # whether to separate an episode per primitive action
    )
    # AWS-specific parameters
    use_aws: bool = False  # Whether to use AWS S3 for input/output
    secret_name: str = ""  # AWS Secrets Manager secret name (required if use_aws=True)
    rosbags_bucket_name: str = (
        ""  # Input S3 bucket for rosbag data (required if use_aws=True)
    )
    lerobot_bucket_name: str = (
        ""  # Output S3 bucket for converted dataset (required if use_aws=True)
    )
    template_dir: str = ""  # Directory path containing meta.json and *.bag files (required if use_aws=True)
    debug: bool = False  # Debug mode (process only first rosbag)
