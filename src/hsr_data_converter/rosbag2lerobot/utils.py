"""
Utility functions for rosbag to lerobot conversion
"""

from pathlib import Path
from typing import Dict

import jsonlines


def update_last_jsonline(new_data: Dict, fpath: Path) -> None:
    fpath.parent.mkdir(exist_ok=True, parents=True)

    # read all lines
    if fpath.exists():
        with jsonlines.open(fpath, "r") as reader:
            lines = list(reader)
    else:
        raise FileNotFoundError(f"{fpath} does not exist")

    if not lines:
        raise ValueError(f"{fpath} is empty")

    # add new pairs of key and value
    lines[-1].update(new_data)

    # update the file
    with jsonlines.open(fpath, "w") as writer:
        writer.write_all(lines)


def update_episodes_jsonl(new_data: Dict, fpath: Path) -> None:
    fpath.parent.mkdir(exist_ok=True, parents=True)

    # read all lines
    if fpath.exists():
        with jsonlines.open(fpath, "r") as reader:
            lines = list(reader)
    else:
        raise FileNotFoundError(f"{fpath} does not exist")

    # add new pairs of key and value to each lines based on
    for line in lines:
        episode_index = line["episode_index"]
        new_info_per_episode = new_data[episode_index]
        line.update(new_info_per_episode)

    # update the file
    with jsonlines.open(fpath, "w") as writer:
        writer.write_all(lines)
