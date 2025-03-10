import argparse
import contextlib
import json
import os
import shutil
import traceback

import numpy as np
import pandas as pd


def load_jsonl(file_path):
    """
    Load data from a JSONL file

    Args:
        file_path (str): Path to the JSONL file

    Returns:
        list: List containing JSON objects from each line
    """
    data = []

    # Special handling for episodes_stats.jsonl
    if "episodes_stats.jsonl" in file_path:
        try:
            # Try to load the entire file as a JSON array
            with open(file_path) as f:
                content = f.read()
                # Check if the content starts with '[' and ends with ']'
                if content.strip().startswith("[") and content.strip().endswith("]"):
                    return json.loads(content)
                else:
                    # Try to add brackets and parse
                    try:
                        return json.loads("[" + content + "]")
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            print(f"Error loading {file_path} as JSON array: {e}")

        # Fall back to line-by-line parsing
        try:
            with open(file_path) as f:
                for line in f:
                    if line.strip():
                        with contextlib.suppress(json.JSONDecodeError):
                            data.append(json.loads(line))
        except Exception as e:
            print(f"Error loading {file_path} line by line: {e}")
    else:
        # Standard JSONL parsing for other files
        with open(file_path) as f:
            for line in f:
                if line.strip():
                    with contextlib.suppress(json.JSONDecodeError):
                        data.append(json.loads(line))

    return data


def save_jsonl(data, file_path):
    """
    Save data in JSONL format

    Args:
        data (list): List of JSON objects to save
        file_path (str): Path to the output file
    """
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def merge_stats(stats_list):
    """
    Merge statistics from multiple datasets, ensuring dimensional consistency

    Args:
        stats_list (list): List of dictionaries containing statistics for each dataset

    Returns:
        dict: Merged statistics
    """
    # Initialize merged stats with the structure of the first stats
    merged_stats = {}

    # Find common features across all stats
    common_features = set(stats_list[0].keys())
    for stats in stats_list[1:]:
        common_features = common_features.intersection(set(stats.keys()))

    # Process features in the order they appear in the first stats file
    for feature in stats_list[0]:
        if feature not in common_features:
            continue

        merged_stats[feature] = {}

        # Find common stat types for this feature
        common_stat_types = []
        for stat_type in ["mean", "std", "max", "min"]:
            if all(stat_type in stats[feature] for stats in stats_list):
                common_stat_types.append(stat_type)

        # Determine the original shape of each value
        original_shapes = []
        for stats in stats_list:
            if "mean" in stats[feature]:
                shape = np.array(stats[feature]["mean"]).shape
                original_shapes.append(shape)

        # Special handling for image features to preserve nested structure
        if feature.startswith("observation.images."):
            for stat_type in common_stat_types:
                try:
                    # Get all values
                    values = [stats[feature][stat_type] for stats in stats_list]

                    # For image features, we need to preserve the nested structure
                    # Initialize with the first value's structure
                    result = []

                    # For RGB channels
                    for channel_idx in range(len(values[0])):
                        channel_result = []

                        # For each pixel row
                        for pixel_idx in range(len(values[0][channel_idx])):
                            pixel_result = []

                            # For each pixel value
                            for value_idx in range(
                                len(values[0][channel_idx][pixel_idx])
                            ):
                                # Calculate statistic based on type
                                if stat_type == "mean":
                                    # Simple average
                                    avg = sum(
                                        values[i][channel_idx][pixel_idx][value_idx]
                                        for i in range(len(values))
                                    ) / len(values)
                                    pixel_result.append(avg)
                                elif stat_type == "std":
                                    # Simple average of std
                                    avg = sum(
                                        values[i][channel_idx][pixel_idx][value_idx]
                                        for i in range(len(values))
                                    ) / len(values)
                                    pixel_result.append(avg)
                                elif stat_type == "max":
                                    # Maximum
                                    max_val = max(
                                        values[i][channel_idx][pixel_idx][value_idx]
                                        for i in range(len(values))
                                    )
                                    pixel_result.append(max_val)
                                elif stat_type == "min":
                                    # Minimum
                                    min_val = min(
                                        values[i][channel_idx][pixel_idx][value_idx]
                                        for i in range(len(values))
                                    )
                                    pixel_result.append(min_val)

                            channel_result.append(pixel_result)

                        result.append(channel_result)

                    merged_stats[feature][stat_type] = result
                except Exception as e:
                    print(
                        f"Warning: Error processing image feature {feature}.{stat_type}: {e}"
                    )
                    # Fallback to first value
                    merged_stats[feature][stat_type] = values[0]
        # If all shapes are the same, no need for special handling
        elif len({str(shape) for shape in original_shapes}) == 1:
            # All shapes are the same, use standard merging
            for stat_type in common_stat_types:
                values = [stats[feature][stat_type] for stats in stats_list]

                try:
                    # Calculate the new statistic based on the type
                    if stat_type == "mean":
                        if all("count" in stats[feature] for stats in stats_list):
                            counts = [
                                stats[feature]["count"][0] for stats in stats_list
                            ]
                            total_count = sum(counts)
                            weighted_values = [
                                np.array(val) * count / total_count
                                for val, count in zip(values, counts, strict=False)
                            ]
                            merged_stats[feature][stat_type] = np.sum(
                                weighted_values, axis=0
                            ).tolist()
                        else:
                            merged_stats[feature][stat_type] = np.mean(
                                np.array(values), axis=0
                            ).tolist()

                    elif stat_type == "std":
                        if all("count" in stats[feature] for stats in stats_list):
                            counts = [
                                stats[feature]["count"][0] for stats in stats_list
                            ]
                            total_count = sum(counts)
                            variances = [np.array(std) ** 2 for std in values]
                            weighted_variances = [
                                var * count / total_count
                                for var, count in zip(variances, counts, strict=False)
                            ]
                            merged_stats[feature][stat_type] = np.sqrt(
                                np.sum(weighted_variances, axis=0)
                            ).tolist()
                        else:
                            merged_stats[feature][stat_type] = np.mean(
                                np.array(values), axis=0
                            ).tolist()

                    elif stat_type == "max":
                        merged_stats[feature][stat_type] = np.maximum.reduce(
                            np.array(values)
                        ).tolist()

                    elif stat_type == "min":
                        merged_stats[feature][stat_type] = np.minimum.reduce(
                            np.array(values)
                        ).tolist()
                except Exception as e:
                    print(f"Warning: Error processing {feature}.{stat_type}: {e}")
                    continue
        else:
            # Shapes are different, need special handling for state vectors
            if feature in ["observation.state", "action"]:
                # For state vectors, we need to handle different dimensions
                max_dim = max(
                    len(np.array(stats[feature]["mean"]).flatten())
                    for stats in stats_list
                )

                for stat_type in common_stat_types:
                    try:
                        # Get values and their original dimensions
                        values_with_dims = []
                        for stats in stats_list:
                            val = np.array(stats[feature][stat_type]).flatten()
                            dim = len(val)
                            values_with_dims.append((val, dim))

                        # Initialize result array with zeros
                        result = np.zeros(max_dim)

                        # Calculate statistics for each dimension separately
                        if stat_type == "mean":
                            if all("count" in stats[feature] for stats in stats_list):
                                counts = [
                                    stats[feature]["count"][0] for stats in stats_list
                                ]
                                total_count = sum(counts)

                                # For each dimension, calculate weighted mean of available values
                                for d in range(max_dim):
                                    dim_values = []
                                    dim_weights = []
                                    for (val, dim), count in zip(
                                        values_with_dims, counts, strict=False
                                    ):
                                        if (
                                            d < dim
                                        ):  # Only use values that have this dimension
                                            dim_values.append(val[d])
                                            dim_weights.append(count)

                                    if (
                                        dim_values
                                    ):  # If we have values for this dimension
                                        weighted_sum = sum(
                                            v * w
                                            for v, w in zip(
                                                dim_values, dim_weights, strict=False
                                            )
                                        )
                                        result[d] = weighted_sum / sum(dim_weights)
                            else:
                                # Simple average for each dimension
                                for d in range(max_dim):
                                    dim_values = [
                                        val[d]
                                        for val, dim in values_with_dims
                                        if d < dim
                                    ]
                                    if dim_values:
                                        result[d] = sum(dim_values) / len(dim_values)

                        elif stat_type == "std":
                            if all("count" in stats[feature] for stats in stats_list):
                                counts = [
                                    stats[feature]["count"][0] for stats in stats_list
                                ]
                                total_count = sum(counts)

                                # For each dimension, calculate weighted variance
                                for d in range(max_dim):
                                    dim_variances = []
                                    dim_weights = []
                                    for (val, dim), count in zip(
                                        values_with_dims, counts, strict=False
                                    ):
                                        if (
                                            d < dim
                                        ):  # Only use values that have this dimension
                                            dim_variances.append(
                                                val[d] ** 2
                                            )  # Square for variance
                                            dim_weights.append(count)

                                    if (
                                        dim_variances
                                    ):  # If we have values for this dimension
                                        weighted_var = sum(
                                            v * w
                                            for v, w in zip(
                                                dim_variances, dim_weights, strict=False
                                            )
                                        ) / sum(dim_weights)
                                        result[d] = np.sqrt(
                                            weighted_var
                                        )  # Take sqrt for std
                            else:
                                # Simple average of std for each dimension
                                for d in range(max_dim):
                                    dim_values = [
                                        val[d]
                                        for val, dim in values_with_dims
                                        if d < dim
                                    ]
                                    if dim_values:
                                        result[d] = sum(dim_values) / len(dim_values)

                        elif stat_type == "max":
                            # For each dimension, take the maximum of available values
                            for d in range(max_dim):
                                dim_values = [
                                    val[d] for val, dim in values_with_dims if d < dim
                                ]
                                if dim_values:
                                    result[d] = max(dim_values)

                        elif stat_type == "min":
                            # For each dimension, take the minimum of available values
                            for d in range(max_dim):
                                dim_values = [
                                    val[d] for val, dim in values_with_dims if d < dim
                                ]
                                if dim_values:
                                    result[d] = min(dim_values)

                        # Convert result to list and store
                        merged_stats[feature][stat_type] = result.tolist()

                    except Exception as e:
                        print(
                            f"Warning: Error processing {feature}.{stat_type} with different dimensions: {e}"
                        )
                        continue
            else:
                # For other features with different shapes, use the first shape as template
                template_shape = original_shapes[0]
                print(f"Using shape {template_shape} as template for {feature}")

                for stat_type in common_stat_types:
                    try:
                        # Use the first stats as template
                        merged_stats[feature][stat_type] = stats_list[0][feature][
                            stat_type
                        ]
                    except Exception as e:
                        print(
                            f"Warning: Error processing {feature}.{stat_type} with shape {template_shape}: {e}"
                        )
                        continue

        # Add count if available in all stats
        if all("count" in stats[feature] for stats in stats_list):
            try:
                merged_stats[feature]["count"] = [
                    sum(stats[feature]["count"][0] for stats in stats_list)
                ]
            except Exception as e:
                print(f"Warning: Error processing {feature}.count: {e}")

    return merged_stats


def copy_videos(source_folders, output_folder, episode_mapping):
    """
    Copy video files from source folders to output folder, maintaining correct indices and structure

    Args:
        source_folders (list): List of source dataset folder paths
        output_folder (str): Output folder path
        episode_mapping (list): List of tuples containing (old_folder, old_index, new_index)
    """
    # Get info.json to determine video structure
    info_path = os.path.join(source_folders[0], "meta", "info.json")
    with open(info_path) as f:
        info = json.load(f)

    video_path_template = info["video_path"]

    # Identify video keys from the template
    # Example: "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    video_keys = []
    for feature_name, feature_info in info["features"].items():
        if feature_info.get("dtype") == "video":
            # Use the full feature name as the video key
            video_keys.append(feature_name)

    print(f"Found video keys: {video_keys}")

    # Copy videos for each episode
    for old_folder, old_index, new_index in episode_mapping:
        # Determine episode chunk (usually 0 for small datasets)
        episode_chunk = old_index // info["chunks_size"]
        new_episode_chunk = new_index // info["chunks_size"]

        for video_key in video_keys:
            # Try different possible source paths
            source_patterns = [
                # Standard path with the episode index from metadata
                os.path.join(
                    old_folder,
                    video_path_template.format(
                        episode_chunk=episode_chunk,
                        video_key=video_key,
                        episode_index=old_index,
                    ),
                ),
                # Try with 0-based indexing
                os.path.join(
                    old_folder,
                    video_path_template.format(
                        episode_chunk=0, video_key=video_key, episode_index=0
                    ),
                ),
                # Try with different formatting
                os.path.join(
                    old_folder,
                    f"videos/chunk-{episode_chunk:03d}/{video_key}/episode_{old_index}.mp4",
                ),
                os.path.join(
                    old_folder, f"videos/chunk-000/{video_key}/episode_000000.mp4"
                ),
            ]

            # Find the first existing source path
            source_video_path = None
            for pattern in source_patterns:
                if os.path.exists(pattern):
                    source_video_path = pattern
                    break

            if source_video_path:
                # Construct destination path
                dest_video_path = os.path.join(
                    output_folder,
                    video_path_template.format(
                        episode_chunk=new_episode_chunk,
                        video_key=video_key,
                        episode_index=new_index,
                    ),
                )

                # Create destination directory if it doesn't exist
                os.makedirs(os.path.dirname(dest_video_path), exist_ok=True)

                print(f"Copying video: {source_video_path} -> {dest_video_path}")
                shutil.copy2(source_video_path, dest_video_path)
            else:
                # If no file is found, search the directory recursively
                found = False
                for root, _, files in os.walk(os.path.join(old_folder, "videos")):
                    for file in files:
                        if file.endswith(".mp4") and video_key in root:
                            source_video_path = os.path.join(root, file)

                            # Construct destination path
                            dest_video_path = os.path.join(
                                output_folder,
                                video_path_template.format(
                                    episode_chunk=new_episode_chunk,
                                    video_key=video_key,
                                    episode_index=new_index,
                                ),
                            )

                            # Create destination directory if it doesn't exist
                            os.makedirs(os.path.dirname(dest_video_path), exist_ok=True)

                            print(
                                f"Copying video (found by search): {source_video_path} -> {dest_video_path}"
                            )
                            shutil.copy2(source_video_path, dest_video_path)
                            found = True
                            break
                    if found:
                        break

                if not found:
                    print(
                        f"Warning: Video file not found for {video_key}, episode {old_index} in {old_folder}"
                    )


def validate_timestamps(source_folders, tolerance_s=1e-4):
    """
    Validate timestamp structure of source datasets, identify potential issues

    Args:
        source_folders (list): List of source dataset folder paths
        tolerance_s (float): Tolerance for timestamp discontinuities in seconds

    Returns:
        tuple: (issues, fps_values) - List of issues and list of detected FPS values
    """
    issues = []
    fps_values = []

    for folder in source_folders:
        try:
            # Try to get FPS from info.json
            info_path = os.path.join(folder, "meta", "info.json")
            if os.path.exists(info_path):
                with open(info_path) as f:
                    info = json.load(f)
                    if "fps" in info:
                        fps = info["fps"]
                        fps_values.append(fps)
                        print(f"Dataset {folder} FPS={fps}")

            # Check if any parquet files contain timestamps
            parquet_path = None
            for root, _, files in os.walk(os.path.join(folder, "parquet")):
                for file in files:
                    if file.endswith(".parquet"):
                        parquet_path = os.path.join(root, file)
                        break
                if parquet_path:
                    break

            if not parquet_path:
                for root, _, files in os.walk(os.path.join(folder, "data")):
                    for file in files:
                        if file.endswith(".parquet"):
                            parquet_path = os.path.join(root, file)
                            break
                    if parquet_path:
                        break

            if parquet_path:
                df = pd.read_parquet(parquet_path)
                timestamp_cols = [
                    col for col in df.columns if "timestamp" in col or "time" in col
                ]
                if timestamp_cols:
                    print(
                        f"Dataset {folder} contains timestamp columns: {timestamp_cols}"
                    )
                else:
                    issues.append(f"Warning: Dataset {folder} has no timestamp columns")
            else:
                issues.append(f"Warning: No parquet files found in dataset {folder}")

        except Exception as e:
            issues.append(f"Error: Failed to validate dataset {folder}: {e}")
            print(f"Validation error: {e}")
            traceback.print_exc()

    # Check if FPS values are consistent
    if len(set(fps_values)) > 1:
        issues.append(f"Warning: Inconsistent FPS across datasets: {fps_values}")

    return issues, fps_values


def copy_data_files(
    source_folders,
    output_folder,
    episode_mapping,
    max_dim=18,
    fps=None,
    episode_to_frame_index=None,
    folder_to_task_strings=None,
    task_string_to_new_index=None,
    chunks_size=1000,
    default_fps=20,
):
    """
    Copy and process parquet data files, including dimension padding and index updates

    Args:
        source_folders (list): List of source dataset folder paths
        output_folder (str): Output folder path
        episode_mapping (list): List of tuples containing (old_folder, old_index, new_index)
        max_dim (int): Maximum dimension for vectors
        fps (float, optional): Frame rate, if not provided will be obtained from the first dataset
        episode_to_frame_index (dict, optional): Mapping of each new episode index to its starting frame index
        folder_to_task_strings (dict, optional): Mapping from folder path to its local (task_index -> task_string) map.
        task_string_to_new_index (dict, optional): Global mapping from task_string to the new merged task_index.
        chunks_size (int): Number of episodes per chunk
        default_fps (float): Default frame rate when unable to obtain from dataset

    Returns:
        bool: Whether at least one file was successfully copied
    """
    # Get FPS from first dataset if not provided
    if fps is None:
        info_path = os.path.join(source_folders[0], "meta", "info.json")
        if os.path.exists(info_path):
            with open(info_path) as f:
                info = json.load(f)
                fps = info.get(
                    "fps", default_fps
                )  # Use variable instead of hardcoded 20
        else:
            fps = default_fps  # Use variable instead of hardcoded 20

    print(f"Using FPS={fps}")

    # Copy and process data files for each episode
    total_copied = 0
    total_failed = 0

    # Add a list to record failed files and reasons
    failed_files = []

    for i, (old_folder, old_index, new_index) in enumerate(episode_mapping):
        # Try to find source parquet file
        episode_str = f"episode_{old_index:06d}.parquet"
        source_paths = [
            os.path.join(old_folder, "parquet", episode_str),
            os.path.join(old_folder, "data", episode_str),
        ]

        source_path = None
        for path in source_paths:
            if os.path.exists(path):
                source_path = path
                break

        if source_path:
            try:
                # Read parquet file
                df = pd.read_parquet(source_path)

                # Check if dimensions need padding
                # for feature in ["observation.state", "action"]:
                #     if feature in df.columns:
                #         # Check first non-null value
                #         for _idx, value in enumerate(df[feature]):
                #             if value is not None and isinstance(value, (list, np.ndarray)):
                #                 current_dim = len(value)
                #                 if current_dim < max_dim:
                #                     print(
                #                         f"Padding {feature} from {current_dim} to {max_dim} dimensions"
                #                     )
                #                     # Pad with zeros to target dimension
                #                     df[feature] = df[feature].apply(
                #                         lambda x: np.pad(x, (0, max_dim - len(x)), "constant").tolist()
                #                         if x is not None
                #                         and isinstance(x, (list, np.ndarray))
                #                         and len(x) < max_dim
                #                         else x
                #                     )
                #                 break

                # Update episode_index column
                if "episode_index" in df.columns:
                    print(
                        f"Update episode_index from {df['episode_index'].iloc[0]} to {new_index}"
                    )
                    df["episode_index"] = new_index

                # Update index column
                if "index" in df.columns:
                    if episode_to_frame_index and new_index in episode_to_frame_index:
                        # Use pre-calculated frame index start value
                        first_index = episode_to_frame_index[new_index]
                        print(
                            f"Update index column, start value: {first_index} (using global cumulative frame count)"
                        )
                    else:
                        # If no mapping provided, use current calculation as fallback
                        first_index = new_index * len(df)
                        print(
                            f"Update index column, start value: {first_index} (using episode index multiplied by length)"
                        )

                    # Update indices for all frames
                    df["index"] = [first_index + i for i in range(len(df))]

                # Update task_index, primitive_action_index, and short_horizon_task_index columns
                if (
                    folder_to_task_strings
                    and task_string_to_new_index
                    and old_folder in folder_to_task_strings
                ):
                    local_task_map = folder_to_task_strings[old_folder]
                    for col_name in [
                        "task_index",
                        "primitive_action_index",
                        "short_horizon_task_index",
                    ]:
                        if col_name in df.columns:
                            original_col = df[col_name]
                            if original_col.isnull().all():
                                continue

                            # Pre-process column to handle unhashable types like numpy arrays
                            def to_hashable(val):
                                if isinstance(val, np.ndarray):
                                    return val.item(0) if val.size > 0 else None
                                return val

                            processed_col = original_col.apply(to_hashable)

                            # Step 1: Convert original index to task string
                            task_strings_col = processed_col.map(local_task_map)

                            # Step 2: Convert task string to new global index
                            new_col = task_strings_col.map(task_string_to_new_index)

                            # Report any indices that were not found in the mapping
                            unmapped_mask = new_col.isnull() & processed_col.notnull()
                            if unmapped_mask.any():
                                unique_unmapped_indices = processed_col[
                                    unmapped_mask
                                ].unique()
                                print(
                                    f"Warning: For column `{col_name}`, no mapping found for task indices: {list(unique_unmapped_indices)}. "
                                    "These values will be set to -1."
                                )
                                new_col = new_col.fillna(-1).astype("int64")

                            df[col_name] = new_col
                            print(
                                f"Column `{col_name}` was updated with new task indices following string-based mapping."
                            )

                # Calculate chunk number
                chunk_index = new_index // chunks_size

                # Create correct target directory
                chunk_dir = os.path.join(
                    output_folder, "data", f"chunk-{chunk_index:03d}"
                )
                os.makedirs(chunk_dir, exist_ok=True)

                # Build correct target path
                dest_path = os.path.join(chunk_dir, f"episode_{new_index:06d}.parquet")

                # Save to correct location
                df.to_parquet(dest_path, index=False)

                total_copied += 1
                print(f"Processed and saved: {dest_path}")

            except Exception as e:
                error_msg = f"Processing {source_path} failed: {e}"
                print(error_msg)
                traceback.print_exc()
                failed_files.append(
                    {"file": source_path, "reason": str(e), "episode": old_index}
                )
                total_failed += 1
        else:
            # File not in standard location, trying recursive search
            found = False
            for root, _, files in os.walk(old_folder):
                for file in files:
                    if file.endswith(".parquet") and f"episode_{old_index:06d}" in file:
                        try:
                            source_path = os.path.join(root, file)

                            # Read parquet file
                            df = pd.read_parquet(source_path)

                            # Check if dimensions need padding
                            # for feature in ["observation.state", "action"]:
                            #     if feature in df.columns:
                            #         # Check first non-null value
                            #         for _idx, value in enumerate(df[feature]):
                            #             if value is not None and isinstance(value, (list, np.ndarray)):
                            #                 current_dim = len(value)
                            #                 if current_dim < max_dim:
                            #                     print(
                            #                         f"Padding {feature} from {current_dim} to {max_dim} dimensions"
                            #                     )
                            #                     # Pad with zeros to target dimension
                            #                     df[feature] = df[feature].apply(
                            #                         lambda x: np.pad(
                            #                             x, (0, max_dim - len(x)), "constant"
                            #                         ).tolist()
                            #                         if x is not None
                            #                         and isinstance(x, (list, np.ndarray))
                            #                         and len(x) < max_dim
                            #                         else x
                            #                     )
                            #                 break

                            # Update episode_index column
                            if "episode_index" in df.columns:
                                print(
                                    f"Update episode_index from {df['episode_index'].iloc[0]} to {new_index}"
                                )
                                df["episode_index"] = new_index

                            # Update index column
                            if "index" in df.columns:
                                if (
                                    episode_to_frame_index
                                    and new_index in episode_to_frame_index
                                ):
                                    # Use pre-calculated frame index start value
                                    first_index = episode_to_frame_index[new_index]
                                    print(
                                        f"Update index column, start value: {first_index} (using global cumulative frame count)"
                                    )
                                else:
                                    # If no mapping provided, use current calculation as fallback
                                    first_index = new_index * len(df)
                                    print(
                                        f"Update index column, start value: {first_index} (using episode index multiplied by length)"
                                    )

                                # Update indices for all frames
                                df["index"] = [first_index + i for i in range(len(df))]

                            # Update task_index, primitive_action_index, and short_horizon_task_index columns
                            if (
                                folder_to_task_strings
                                and task_string_to_new_index
                                and old_folder in folder_to_task_strings
                            ):
                                local_task_map = folder_to_task_strings[old_folder]
                                for col_name in [
                                    "task_index",
                                    "primitive_action_index",
                                    "short_horizon_task_index",
                                ]:
                                    if col_name in df.columns:
                                        original_col = df[col_name]
                                        if original_col.isnull().all():
                                            continue

                                        # Pre-process column to handle unhashable types like numpy arrays
                                        def to_hashable(val):
                                            if isinstance(val, np.ndarray):
                                                return (
                                                    val.item(0)
                                                    if val.size > 0
                                                    else None
                                                )
                                            return val

                                        processed_col = original_col.apply(to_hashable)

                                        # Step 1: Convert original index to task string
                                        task_strings_col = processed_col.map(
                                            local_task_map
                                        )

                                        # Step 2: Convert task string to new global index
                                        new_col = task_strings_col.map(
                                            task_string_to_new_index
                                        )

                                        # Report any indices that were not found in the mapping
                                        unmapped_mask = (
                                            new_col.isnull() & processed_col.notnull()
                                        )
                                        if unmapped_mask.any():
                                            unique_unmapped_indices = processed_col[
                                                unmapped_mask
                                            ].unique()
                                            print(
                                                f"Warning: For column `{col_name}`, no mapping found for task indices: {list(unique_unmapped_indices)}. "
                                                "These values will be set to -1."
                                            )
                                            new_col = new_col.fillna(-1).astype("int64")

                                        df[col_name] = new_col
                                        print(
                                            f"Column `{col_name}` was updated with new task indices following string-based mapping."
                                        )

                            # Calculate chunk number
                            chunk_index = new_index // chunks_size

                            # Create correct target directory
                            chunk_dir = os.path.join(
                                output_folder, "data", f"chunk-{chunk_index:03d}"
                            )
                            os.makedirs(chunk_dir, exist_ok=True)

                            # Build correct target path
                            dest_path = os.path.join(
                                chunk_dir, f"episode_{new_index:06d}.parquet"
                            )

                            # Save to correct location
                            df.to_parquet(dest_path, index=False)

                            total_copied += 1
                            found = True
                            print(f"Processed and saved: {dest_path}")
                            break
                        except Exception as e:
                            error_msg = f"Processing {source_path} failed: {e}"
                            print(error_msg)
                            traceback.print_exc()
                            failed_files.append(
                                {
                                    "file": source_path,
                                    "reason": str(e),
                                    "episode": old_index,
                                }
                            )
                            total_failed += 1
                    if found:
                        break

            if not found:
                error_msg = f"Could not find parquet file for episode {old_index}, source folder: {old_folder}"
                print(error_msg)
                failed_files.append(
                    {
                        "file": f"episode_{old_index:06d}.parquet",
                        "reason": "File not found",
                        "folder": old_folder,
                    }
                )
                total_failed += 1

    print(f"Copied {total_copied} data files, {total_failed} failed")

    # Print details of all failed files
    if failed_files:
        print("\nDetails of failed files:")
        for i, failed in enumerate(failed_files):
            print(f"{i + 1}. File: {failed['file']}")
            if "folder" in failed:
                print(f"   Folder: {failed['folder']}")
            if "episode" in failed:
                print(f"   Episode index: {failed['episode']}")
            print(f"   Reason: {failed['reason']}")
            print("---")

    return total_copied > 0


def pad_parquet_data(source_path, target_path, original_dim=14, target_dim=18):
    """
    Extend parquet data from original dimension to target dimension by zero-padding

    Args:
        source_path (str): Source parquet file path
        target_path (str): Target parquet file path
        original_dim (int): Original vector dimension
        target_dim (int): Target vector dimension
    """
    # Read parquet file
    df = pd.read_parquet(source_path)

    # Print column names for debugging
    print(f"Columns in {source_path}: {df.columns.tolist()}")

    # Create a new DataFrame to store padded data
    new_df = df.copy()

    # Check if observation.state and action columns exist
    if "observation.state" in df.columns:
        # Check the first row of data to confirm if it is a vector
        first_state = df["observation.state"].iloc[0]
        print(
            f"First observation.state type: {type(first_state)}, value: {first_state}"
        )

        # If it's a vector (list or numpy array)
        if isinstance(first_state, (list, np.ndarray)):
            # Check dimension
            state_dim = len(first_state)
            print(f"observation.state dimension: {state_dim}")

            if state_dim < target_dim:
                # Pad vector
                print(
                    f"Padding observation.state from {state_dim} to {target_dim} dimensions"
                )
                new_df["observation.state"] = df["observation.state"].apply(
                    lambda x: np.pad(x, (0, target_dim - len(x)), "constant").tolist()
                )

    # Process action column similarly
    if "action" in df.columns:
        # Check first row data
        first_action = df["action"].iloc[0]
        print(f"First action type: {type(first_action)}, value: {first_action}")

        # If it's a vector
        if isinstance(first_action, (list, np.ndarray)):
            # Check dimension
            action_dim = len(first_action)
            print(f"action dimension: {action_dim}")

            if action_dim < target_dim:
                # Pad vector
                print(f"Padding action from {action_dim} to {target_dim} dimensions")
                new_df["action"] = df["action"].apply(
                    lambda x: np.pad(x, (0, target_dim - len(x)), "constant").tolist()
                )

    # Ensure target directory exists
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    # Save to a new parquet file
    new_df.to_parquet(target_path, index=False)

    print(f"Processed {source_path} and saved to {target_path}")

    return new_df


def merge_datasets(
    source_folders,
    output_folder,
    validate_ts=False,
    tolerance_s=1e-4,
    max_dim=18,
    default_fps=20,
):
    """
    Merge multiple dataset folders into one, handling indices, dimensions, and metadata

    Args:
        source_folders (list): List of source dataset folder paths
        output_folder (str): Output folder path
        validate_ts (bool): Whether to validate timestamps
        tolerance_s (float): Tolerance for timestamp discontinuities in seconds
        max_dim (int): Maximum dimension for vectors
        default_fps (float): Default frame rate

    This function performs the following operations:
    1. Merges all episodes, tasks and stats
    2. Renumbers all indices to maintain continuity
    3. Pads vector dimensions for consistency
    4. Updates metadata files
    5. Copies and processes data and video files
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "meta"), exist_ok=True)

    # Comment out timestamp validation
    # if validate_ts:
    #     issues, fps_values = validate_timestamps(source_folders, tolerance_s)
    #     if issues:
    #         print("Timestamp validation found the following issues:")
    #         for issue in issues:
    #             print(f"  - {issue}")
    #
    #     # Get common FPS value
    #     if fps_values:
    #         fps = max(set(fps_values), key=fps_values.count)
    #         print(f"Using common FPS value: {fps}")
    #     else:
    #         fps = default_fps
    #         print(f"FPS value not found, using default: {default_fps}")
    # else:
    fps = default_fps
    print(f"Using default FPS value: {fps}")

    # Load episodes from all source folders
    all_episodes = []
    all_episodes_stats = []
    all_tasks = []

    folder_to_task_strings = {}
    task_string_to_new_index = {}
    all_unique_tasks = []

    # First, build the mappings from tasks across all datasets
    for folder in source_folders:
        tasks_path = os.path.join(folder, "meta", "tasks.jsonl")
        if not os.path.exists(tasks_path):
            continue

        folder_tasks = load_jsonl(tasks_path)
        current_folder_map = {task["task_index"]: task["task"] for task in folder_tasks}
        folder_to_task_strings[folder] = current_folder_map

        for task in folder_tasks:
            task_desc = task["task"]
            if task_desc not in task_string_to_new_index:
                new_index = len(all_unique_tasks)
                task_string_to_new_index[task_desc] = new_index
                all_unique_tasks.append({"task_index": new_index, "task": task_desc})

    all_tasks = all_unique_tasks

    total_frames = 0
    total_episodes = 0
    total_videos = 0

    # Keep track of episode mapping (old_folder, old_index, new_index)
    episode_mapping = []

    # Collect all stats for proper merging
    all_stats_data = []

    # Track dimensions for each folder
    folder_dimensions = {}

    # Add a variable to track cumulative frames
    cumulative_frame_count = 0

    # Create a mapping to store the starting frame index for each new episode index
    episode_to_frame_index = {}

    # 从info.json获取chunks_size
    info_path = os.path.join(source_folders[0], "meta", "info.json")
    chunks_size = 1000  # 默认值
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
            chunks_size = info.get("chunks_size", 1000)

    for folder in source_folders:
        try:
            # Get total_videos directly from each dataset's info.json
            folder_info_path = os.path.join(folder, "meta", "info.json")
            if os.path.exists(folder_info_path):
                with open(folder_info_path) as f:
                    folder_info = json.load(f)
                    if "total_videos" in folder_info:
                        folder_videos = folder_info["total_videos"]
                        total_videos += folder_videos
                        print(
                            f"Read video count from {folder}'s info.json: {folder_videos}"
                        )

            # Check dimensions of state vectors in this folder
            folder_dim = max_dim  # Use variable instead of hardcoded 18

            # Try to find a parquet file to determine dimensions
            for root, _dirs, files in os.walk(folder):
                for file in files:
                    if file.endswith(".parquet"):
                        try:
                            df = pd.read_parquet(os.path.join(root, file))
                            if "observation.state" in df.columns:
                                first_state = df["observation.state"].iloc[0]
                                if isinstance(first_state, (list, np.ndarray)):
                                    folder_dim = len(first_state)
                                    print(
                                        f"Detected {folder_dim} dimensions in {folder}"
                                    )
                                    break
                        except Exception as e:
                            print(f"Error checking dimensions in {folder}: {e}")
                        break
                if folder_dim != max_dim:  # Use variable instead of hardcoded 18
                    break

            folder_dimensions[folder] = (
                folder_dim  # source_folderをkeyとして, valueには状態のdimを保存
            )

            # Load episodes
            episodes_path = os.path.join(folder, "meta", "episodes.jsonl")
            if not os.path.exists(episodes_path):
                print(f"Warning: Episodes file not found in {folder}, skipping")
                continue

            episodes = load_jsonl(episodes_path)

            # Load episode stats
            episodes_stats_path = os.path.join(folder, "meta", "episodes_stats.jsonl")
            episodes_stats = []
            if os.path.exists(episodes_stats_path):
                episodes_stats = load_jsonl(episodes_stats_path)

            # Create a mapping of episode_index to stats
            stats_map = {}
            for stat in episodes_stats:
                if "episode_index" in stat:
                    stats_map[stat["episode_index"]] = stat

            # Process all episodes from this folder
            for episode in episodes:
                old_index = episode["episode_index"]
                new_index = total_episodes

                # Update episode index
                episode["episode_index"] = new_index
                all_episodes.append(episode)

                # Update stats if available
                if old_index in stats_map:
                    stats = stats_map[old_index]
                    stats["episode_index"] = new_index

                    # # Pad stats data if needed
                    # if "stats" in stats and folder_dimensions[folder] < max_dim:  # Use variable instead of hardcoded 18
                    #     # Pad observation.state and action stats
                    #     for feature in ["observation.state", "action"]:
                    #         if feature in stats["stats"]:
                    #             for stat_type in ["mean", "std", "max", "min"]:
                    #                 if stat_type in stats["stats"][feature]:
                    #                     # Get current values
                    #                     values = stats["stats"][feature][stat_type]

                    #                     # Check if it's a list/array that needs padding
                    #                     if (
                    #                         isinstance(values, list) and len(values) < max_dim
                    #                     ):  # Use variable instead of hardcoded 18
                    #                         # Pad with zeros
                    #                         padded = values + [0.0] * (
                    #                             max_dim - len(values)
                    #                         )  # Use variable instead of hardcoded 18
                    #                         stats["stats"][feature][stat_type] = padded

                    all_episodes_stats.append(stats)

                    # Add to all_stats_data for proper merging
                    if "stats" in stats:
                        all_stats_data.append(stats["stats"])

                # Add to mapping
                episode_mapping.append((folder, old_index, new_index))

                # Update counters
                total_episodes += 1
                total_frames += episode["length"]

                # Collect this information when processing each episode
                episode_to_frame_index[new_index] = cumulative_frame_count
                cumulative_frame_count += episode["length"]

        except Exception as e:
            print(f"Error processing folder {folder}: {e}")
            continue

    print(f"Processed {total_episodes} episodes from {len(source_folders)} folders")

    # Save combined episodes and stats
    save_jsonl(all_episodes, os.path.join(output_folder, "meta", "episodes.jsonl"))
    save_jsonl(
        all_episodes_stats, os.path.join(output_folder, "meta", "episodes_stats.jsonl")
    )
    save_jsonl(all_tasks, os.path.join(output_folder, "meta", "tasks.jsonl"))

    # Merge and save stats
    stats_list = []
    for folder in source_folders:
        stats_path = os.path.join(folder, "meta", "stats.json")
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                stats = json.load(f)
                stats_list.append(stats)

    if stats_list:
        # Merge global stats
        merged_stats = merge_stats(stats_list)

        # Update merged stats with episode-specific stats if available
        if all_stats_data:
            # For each feature in the stats
            for feature in merged_stats:
                if feature in all_stats_data[0]:
                    # Recalculate statistics based on all episodes
                    values = [
                        stat[feature] for stat in all_stats_data if feature in stat
                    ]

                    # Find the maximum dimension for this feature
                    max_dim = max(
                        len(np.array(val.get("mean", [0])).flatten())
                        for val in values
                        if "mean" in val
                    )

                    # Update count
                    if "count" in merged_stats[feature]:
                        merged_stats[feature]["count"] = [
                            sum(
                                stat.get("count", [0])[0]
                                for stat in values
                                if "count" in stat
                            )
                        ]

                    # Update min/max with padding
                    if "min" in merged_stats[feature] and all(
                        "min" in stat for stat in values
                    ):
                        # Pad min values
                        padded_mins = []
                        for val in values:
                            val_array = np.array(val["min"])
                            val_flat = val_array.flatten()
                            if len(val_flat) < max_dim:
                                padded = np.zeros(max_dim)
                                padded[: len(val_flat)] = val_flat
                                padded_mins.append(padded)
                            else:
                                padded_mins.append(val_flat)
                        merged_stats[feature]["min"] = np.minimum.reduce(
                            padded_mins
                        ).tolist()

                    if "max" in merged_stats[feature] and all(
                        "max" in stat for stat in values
                    ):
                        # Pad max values
                        padded_maxs = []
                        for val in values:
                            val_array = np.array(val["max"])
                            val_flat = val_array.flatten()
                            if len(val_flat) < max_dim:
                                padded = np.zeros(max_dim)
                                padded[: len(val_flat)] = val_flat
                                padded_maxs.append(padded)
                            else:
                                padded_maxs.append(val_flat)
                        merged_stats[feature]["max"] = np.maximum.reduce(
                            padded_maxs
                        ).tolist()

                    # Update mean and std (weighted by count if available)
                    if "mean" in merged_stats[feature] and all(
                        "mean" in stat for stat in values
                    ):
                        # Pad mean values
                        padded_means = []
                        for val in values:
                            val_array = np.array(val["mean"])
                            val_flat = val_array.flatten()
                            if len(val_flat) < max_dim:
                                padded = np.zeros(max_dim)
                                padded[: len(val_flat)] = val_flat
                                padded_means.append(padded)
                            else:
                                padded_means.append(val_flat)

                        if all("count" in stat for stat in values):
                            counts = [stat["count"][0] for stat in values]
                            total_count = sum(counts)
                            weighted_means = [
                                mean * count / total_count
                                for mean, count in zip(
                                    padded_means, counts, strict=False
                                )
                            ]
                            merged_stats[feature]["mean"] = np.sum(
                                weighted_means, axis=0
                            ).tolist()
                        else:
                            merged_stats[feature]["mean"] = np.mean(
                                padded_means, axis=0
                            ).tolist()

                    if "std" in merged_stats[feature] and all(
                        "std" in stat for stat in values
                    ):
                        # Pad std values
                        padded_stds = []
                        for val in values:
                            val_array = np.array(val["std"])
                            val_flat = val_array.flatten()
                            if len(val_flat) < max_dim:
                                padded = np.zeros(max_dim)
                                padded[: len(val_flat)] = val_flat
                                padded_stds.append(padded)
                            else:
                                padded_stds.append(val_flat)

                        if all("count" in stat for stat in values):
                            counts = [stat["count"][0] for stat in values]
                            total_count = sum(counts)
                            variances = [std**2 for std in padded_stds]
                            weighted_variances = [
                                var * count / total_count
                                for var, count in zip(variances, counts, strict=False)
                            ]
                            merged_stats[feature]["std"] = np.sqrt(
                                np.sum(weighted_variances, axis=0)
                            ).tolist()
                        else:
                            # Simple average of standard deviations
                            merged_stats[feature]["std"] = np.mean(
                                padded_stds, axis=0
                            ).tolist()

        with open(os.path.join(output_folder, "meta", "stats.json"), "w") as f:
            json.dump(merged_stats, f, indent=4)

    # Update and save info.json
    info_path = os.path.join(source_folders[0], "meta", "info.json")
    with open(info_path) as f:
        info = json.load(f)

    # Update info with correct counts
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_tasks"] = len(all_tasks)
    info["total_chunks"] = (total_episodes + info["chunks_size"] - 1) // info[
        "chunks_size"
    ]  # Ceiling division

    # Update splits
    info["splits"] = {"train": f"0:{total_episodes}"}

    # Update feature dimensions to the maximum dimension
    if "features" in info:
        # Find the maximum dimension across all folders
        actual_max_dim = max_dim  # Use variable instead of hardcoded 18
        for _folder, dim in folder_dimensions.items():
            actual_max_dim = max(actual_max_dim, dim)

        # Update observation.state and action dimensions
        for feature_name in ["observation.state", "action"]:
            if (
                feature_name in info["features"]
                and "shape" in info["features"][feature_name]
            ):
                info["features"][feature_name]["shape"] = [actual_max_dim]
                print(f"Updated {feature_name} shape to {actual_max_dim}")

    # Update total videos
    info["total_videos"] = total_videos
    print(f"Update total videos to: {total_videos}")

    with open(os.path.join(output_folder, "meta", "info.json"), "w") as f:
        json.dump(info, f, indent=4)

    # Copy video and data files
    copy_videos(source_folders, output_folder, episode_mapping)
    copy_data_files(
        source_folders,
        output_folder,
        episode_mapping,
        max_dim=max_dim,
        fps=fps,
        episode_to_frame_index=episode_to_frame_index,
        folder_to_task_strings=folder_to_task_strings,
        task_string_to_new_index=task_string_to_new_index,
        chunks_size=chunks_size,
    )

    print(
        f"Merged {total_episodes} episodes with {total_frames} frames into {output_folder}"
    )


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Merge datasets from multiple sources."
    )

    # Add arguments
    parser.add_argument(
        "--sources", nargs="+", required=True, help="List of source folder paths"
    )
    parser.add_argument("--output", required=True, help="Output folder path")
    parser.add_argument(
        "--max_dim", type=int, default=32, help="Maximum dimension (default: 32)"
    )
    parser.add_argument(
        "--fps", type=int, default=10, help="Your datasets FPS (default: 20)"
    )

    # Parse arguments
    args = parser.parse_args()

    # Use parsed arguments
    merge_datasets(
        args.sources, args.output, max_dim=args.max_dim, default_fps=args.fps
    )
