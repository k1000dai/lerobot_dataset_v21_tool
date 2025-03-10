import pandas as pd
import os
import argparse
import glob
import json
import shutil

FILTERED_SHORT_HORIZON_TASKS = ["Open and Close the oven toaster"]
FILTERED_PRIMITIVE_ACTIONS = [
    "Pick and Place an apple on the floor",
    "Pick up an apple",
]


def read_meta(input_dataset_path):
    episodes_meta_path = os.path.join(input_dataset_path, "meta", "episodes.jsonl")
    with open(episodes_meta_path, "r") as f:
        episodes_meta = [json.loads(line) for line in f]

    episodes_stats_meta_path = os.path.join(
        input_dataset_path, "meta", "episodes_stats.jsonl"
    )
    with open(episodes_stats_meta_path, "r") as f:
        episodes_stats_meta = [json.loads(line) for line in f]

    info_meta_path = os.path.join(input_dataset_path, "meta", "info.json")
    with open(info_meta_path, "r") as f:
        info_meta = json.load(f)

    tasks_meta_path = os.path.join(input_dataset_path, "meta", "tasks.jsonl")
    with open(tasks_meta_path, "r") as f:
        tasks_meta = [json.loads(line) for line in f]

    metas = {
        "episodes": episodes_meta,
        "episodes_stats": episodes_stats_meta,
        "info": info_meta,
        "tasks": tasks_meta,
    }
    return metas


def get_filtered_episode_by_primitive_action_failure(parquet_file_paths, metas):
    filtered_episode_indices = []
    for parquet_file_path in parquet_file_paths:
        df = pd.read_parquet(parquet_file_path)
        success_primitive_action = df["success_primitive_action"].unique()
        if False in success_primitive_action:  # TODO
            filtered_episode_indices.append(int(df["episode_index"].iloc[0]))
    return filtered_episode_indices


def get_filtered_episode_by_robot_types(
    parquet_file_paths, metas, filtered_robot_types
):
    filtered_episode_indices = []
    episodes_meta = metas["episodes"]
    for episode_meta in episodes_meta:
        if episode_meta["hsr_id"] in filtered_robot_types:
            filtered_episode_indices.append(episode_meta["episode_index"])
    return filtered_episode_indices


def get_filtered_episode_by_primitive_action(
    parquet_file_paths, metas, filtered_primitive_actions
):
    # get filtered primitive action indices
    filtered_primitive_action_indices = []
    tasks_meta = metas["tasks"]
    for task in tasks_meta:
        if task["task"] in filtered_primitive_actions:
            filtered_primitive_action_indices.append(task["task_index"])

    # get filtered episode indices
    filtered_episode_indices = []
    for parquet_file_path in parquet_file_paths:
        df = pd.read_parquet(parquet_file_path)
        if df["primitive_action_index"].isin(filtered_primitive_action_indices).any():
            filtered_episode_indices.append(int(df["episode_index"].iloc[0]))
    return filtered_episode_indices


def get_filtered_episode_by_short_horizon_task(
    parquet_file_paths, metas, filtered_short_horizon_tasks
):
    # get filtered short horizon task indices
    filtered_short_horizon_task_indices = []
    tasks_meta = metas["tasks"]
    for task in tasks_meta:
        if task["task"] in filtered_short_horizon_tasks:
            filtered_short_horizon_task_indices.append(task["task_index"])

    # get filtered episode indices
    filtered_episode_indices = []
    for parquet_file_path in parquet_file_paths:
        df = pd.read_parquet(parquet_file_path)
        if (
            df["short_horizon_task_index"]
            .isin(filtered_short_horizon_task_indices)
            .any()
        ):
            filtered_episode_indices.append(int(df["episode_index"].iloc[0]))
    return filtered_episode_indices


def update_meta(metas, parquet_file_paths, filtered_episode_indices, chunk_size):
    new_episodes = []
    new_episodes_stats = []
    new_episode_idx_counter = 0
    new_tasks_list = set()
    for parquet_file_path in parquet_file_paths:
        df = pd.read_parquet(parquet_file_path)
        old_episode_index = df["episode_index"].iloc[0]
        if old_episode_index in filtered_episode_indices:
            continue

        # Update episode meta
        new_episode = metas["episodes"][old_episode_index]
        new_episode["episode_index"] = new_episode_idx_counter
        new_episodes.append(new_episode)

        # Update episode stats meta
        new_episode_stat = metas["episodes_stats"][old_episode_index]
        new_episode_stat["episode_index"] = new_episode_idx_counter
        new_episodes_stats.append(new_episode_stat)

        # get unique task indices
        unique_task_indices = set(df["short_horizon_task_index"].unique()) | set(
            df["primitive_action_index"].unique()
        )
        new_tasks_list.update(unique_task_indices)
        new_episode_idx_counter += 1

    # remove -1 (which is for NaN) task index from new_tasks_list
    new_tasks_list = new_tasks_list - {-1}

    # Update tasks meta
    new_tasks = []
    old_to_new_task_index_map = {}
    for new_task_index, task_index in enumerate(sorted(list(new_tasks_list))):
        task_index_dtype = type(task_index)
        new_task = metas["tasks"][task_index]
        new_task["task_index"] = new_task_index
        new_tasks.append(new_task)
        old_to_new_task_index_map[task_index] = task_index_dtype(new_task_index)
    old_to_new_task_index_map[task_index_dtype(-1)] = task_index_dtype(-1)

    # Update info meta
    new_info = metas["info"]
    new_info["total_episodes"] = len(new_episodes)
    new_info["total_frames"] = sum(episode["length"] for episode in new_episodes)
    new_info["total_tasks"] = len(new_tasks)
    new_info["total_videos"] = len(new_episodes) * 2
    new_info["total_chunks"] = len(new_episodes) // chunk_size + 1
    new_info["chunks_size"] = chunk_size

    updated_metas = {
        "episodes": new_episodes,
        "episodes_stats": new_episodes_stats,
        "tasks": new_tasks,
        "info": new_info,
    }

    return updated_metas, old_to_new_task_index_map


def create_filtered_dataset(
    input_dataset_path, output_dataset_path, metas, filtered_episode_indices, chunk_size
):
    parquet_file_paths = sorted(
        glob.glob(
            os.path.join(input_dataset_path, "data", "chunk-*", "episode_*.parquet")
        )
    )
    updated_metas, old_to_new_task_index_map = update_meta(
        metas, parquet_file_paths, filtered_episode_indices, chunk_size
    )

    new_episode_index = 0
    for parquet_file_path in parquet_file_paths:
        # skip filtered episodes
        if (
            int(os.path.basename(parquet_file_path).split("_")[1].split(".")[0])
            in filtered_episode_indices
        ):
            continue

        df = pd.read_parquet(parquet_file_path)
        # update "episode_index", "short_horizon_task_index", "primitive_action_index", "task_index"
        df["episode_index"] = new_episode_index
        df["short_horizon_task_index"] = df["short_horizon_task_index"].map(
            old_to_new_task_index_map
        )
        df["primitive_action_index"] = df["primitive_action_index"].map(
            old_to_new_task_index_map
        )
        df["task_index"] = df["task_index"].map(old_to_new_task_index_map)

        # save to new parquet file
        new_parquet_file_path = os.path.join(
            output_dataset_path,
            "data",
            f"chunk-{new_episode_index // chunk_size:03d}",
            f"episode_{new_episode_index:06d}.parquet",
        )
        if not os.path.exists(os.path.dirname(new_parquet_file_path)):
            os.makedirs(os.path.dirname(new_parquet_file_path))
        df.to_parquet(new_parquet_file_path)
        new_episode_index += 1

    # metas
    os.makedirs(os.path.join(output_dataset_path, "meta"), exist_ok=True)
    for meta_type in ["episodes", "episodes_stats", "tasks"]:
        with open(
            os.path.join(output_dataset_path, "meta", f"{meta_type}.jsonl"), "w"
        ) as f:
            for meta in updated_metas[meta_type]:
                f.write(json.dumps(meta) + "\n")
    with open(os.path.join(output_dataset_path, "meta", "info.json"), "w") as f:
        json.dump(updated_metas["info"], f, indent=4)

    # copy videos
    for camera_type in ["hand", "head"]:
        new_episode_index = 0
        for video_path in sorted(
            glob.glob(
                os.path.join(
                    input_dataset_path,
                    "videos",
                    "chunk-*",
                    f"observation.image.{camera_type}",
                    "*.mp4",
                )
            )
        ):
            os.makedirs(
                os.path.join(
                    output_dataset_path,
                    "videos",
                    f"chunk-{new_episode_index // chunk_size:03d}",
                    f"observation.image.{camera_type}",
                ),
                exist_ok=True,
            )
            # skip filtered episodes
            if (
                int(os.path.basename(video_path).split("_")[1].split(".")[0])
                in filtered_episode_indices
            ):
                continue
            new_video_path = os.path.join(
                output_dataset_path,
                "videos",
                f"chunk-{new_episode_index // chunk_size:03d}",
                f"observation.image.{camera_type}",
                f"episode_{new_episode_index:06d}.mp4",
            )
            shutil.copy(video_path, new_video_path)
            new_episode_index += 1


def main(input_dataset_path, output_dataset_path, chunk_size):
    parquet_file_paths = sorted(
        glob.glob(
            os.path.join(input_dataset_path, "data", "chunk-*", "episode_*.parquet")
        )
    )
    metas = read_meta(input_dataset_path)

    filtered_episode_indices = set()

    # TODO: get filtered episode indices
    _filtered_episode_indices = get_filtered_episode_by_primitive_action_failure(
        parquet_file_paths, metas
    )
    filtered_episode_indices.update(_filtered_episode_indices)

    # TODO: get filtered episode indices by robot types
    # _filtered_episode_indices = get_filtered_episode_by_robot_types(parquet_file_paths, metas, ["hsrc_049"])
    # filtered_episode_indices.update(_filtered_episode_indices)

    # #TODO: get filtered episode indices by primitive action
    # _filtered_episode_indices = get_filtered_episode_by_primitive_action(parquet_file_paths, metas, FILTERED_PRIMITIVE_ACTIONS)
    # filtered_episode_indices.update(_filtered_episode_indices)

    # #TODO: get filtered episode indices by short horizon task
    # _filtered_episode_indices = get_filtered_episode_by_short_horizon_task(parquet_file_paths, metas, FILTERED_SHORT_HORIZON_TASKS)
    # filtered_episode_indices.update(_filtered_episode_indices)

    filtered_episode_indices = list(filtered_episode_indices)
    print(f"Filtered {len(filtered_episode_indices)} episodes")
    create_filtered_dataset(
        input_dataset_path,
        output_dataset_path,
        metas,
        filtered_episode_indices,
        chunk_size,
    )


if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset_path", type=str, required=True)
    parser.add_argument("--output_dataset_path", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, default=1000)
    args = parser.parse_args()
    input_dataset_path = args.input_dataset_path
    output_dataset_path = args.output_dataset_path
    chunk_size = args.chunk_size

    main(input_dataset_path, output_dataset_path, chunk_size)
