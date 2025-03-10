# Rebake: 🍞 Bake your robot data into ML-ready formats

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Robots](https://img.shields.io/badge/Robots-HSR-blue)](https://github.com/airoa-org/rebake)

> 🥖 Take raw robot data, sprinkle in metadata, and rebake it into fresh, standardized LeRobot datasets — ready to serve your machine learning models.

## Overview

**Hot out of the oven and still rising! 🧑‍🍳 Currently under active development. Expect changes in API and command line arguments.**  

Rebake turns the messy leftovers of robotic experiments into neatly baked datasets. By converting and unifying recordings from different robots into the LeRobot format, Rebake makes it easy to share, compare, and train on consistent data.

Think of it as your robot kitchen:

🍪 Raw ingredients → rosbags and metadata

🍳 Recipe → filtering rules and packaging

🍰 Fresh loaf → ML-ready LeRobot datasets

No more format fragmentation — just tasty, standardized data your models will love.

## Links to datasets

> 🚧 **Coming Soon**

## Supported Robots

| Robot Platform | Status        | Data Format | Features                       | Config |
| -------------- | ------------- | ----------- | ------------------------------ | ------ |
| Toyota HSR     | ✅ Production | rosbag      | Full conversion, visualization | `hsr`  |

> **Want your robot supported?** [Open an issue](https://github.com/airoa-org/rebake/issues/new) or contribute a plugin!

### Key Features

* 🔄 **Multi-robot support** (🔮 planned) – Extensible plugin system for adding new robot platforms.
* 📊 **Flexible processing modes** – Convert individual episodes or combine many into larger datasets.
* ☁️ **Cloud-ready** – AWS Batch integration for scaling conversion jobs to big datasets.
* 🎥 **Built-in visualization** – Generate videos and HTML views to quickly inspect your data.
* 📈 **Data management tools** – Filter, merge, and analyze converted datasets with ease.
* 🤖 **Production-tested** – Currently supports Toyota HSR, with more robots on the way.

## Quick Start

### Prerequisites

- Python 3.10 or higher
- Docker and docker-compose
- **Toyota HSR recorded data** (rosbag format with meta.json metadata)
- Currently supports HSR only - other robots coming soon
- pip or uv package manager

### 1-Minute Setup

```bash
# Clone the repository
git clone https://github.com/airoa-org/rebake.git
cd rebake

# Set your HSR dataset directory
export HSR_DATASET_DIR=/path/to/your/rosbag/data

# Build and run the container
cd docker
docker compose build hsr_data_converter
docker compose run hsr_data_converter
```

### Convert Your First Dataset

```bash
# Inside the container, install dependencies
GIT_LFS_SKIP_SMUDGE=1 uv sync

# Convert rosbag to LeRobot format (HSR example)
uv run -m hsr_data_converter.rosbag2lerobot.main \
    --raw_dir /root/datasets \
    --out_dir ./output \
    --fps 10 \
    --robot_type hsr \
    --conversion_type individual
```

🎉 Your LeRobot dataset will be ready in `./output` with videos, metadata, and structured data!

### Verify Your Dataset

```bash
# Visualize the converted dataset
uv run src/hsr_data_converter/visualize/lerobot_dataset.py \
    --repo-id your_dataset_name \
    --root ./output/{Episode directory name} \
    --episode-index 0
```

## Installation

### Using Docker (Recommended)

The easiest way to get started is using the provided Docker environment:

```bash
git clone https://github.com/airoa-org/rebake.git
cd rebake
git submodule update --init --recursive
cd docker
docker compose build hsr_data_converter
docker compose run hsr_data_converter
```

### Local Development Setup

For development or if you prefer local installation:

```bash
# Install dependencies with uv
GIT_LFS_SKIP_SMUDGE=1 uv sync

# Initialize submodules
git submodule update --init --recursive
```

## Usage

### Basic Conversion

Convert HSR recorded data to LeRobot format:

```bash
uv run -m hsr_data_converter.rosbag2lerobot.main \
    --raw_dir /path/to/rosbags \
    --out_dir /path/to/output \
    --fps 10 \
    --robot_type hsr \
    --conversion_type aggregate \
    --separate_per_primitive false
```

### Processing Modes

- **`individual`**: Convert each rosbag to separate datasets
- **`aggregate`**: Combine multiple rosbags into a single dataset

### Data Management

#### Filter Episodes

Remove specific episodes based on criteria:

```bash
uv run src/hsr_data_converter/filter_episodes.py \
    --input_dataset_path ./input_dataset \
    --output_dataset_path ./filtered_dataset \
    --chunk_size 1000
```

#### Merge Datasets

Combine multiple datasets:

```bash
uv run src/hsr_data_converter/merge_dataset.py \
    --sources ./dataset1 ./dataset2 \
    --output ./merged_dataset \
    --fps 10
```

#### Visualize Data

Generate dataset visualization:

```bash
uv run src/hsr_data_converter/visualize/lerobot_dataset.py \
    --repo-id dataset_name \
    --root ./dataset_path \
    --episode-index 0
```

### Data Format Requirements

HSR recorded data should follow this structure:

```
dataset_directory/
├── template-061707-25-04-30-09-01-51/
│   ├── data.bag
│   └── meta.json
├── template-061707-25-04-30-09-02-45/
│   ├── data.bag
│   └── meta.json
├── template-061707-25-04-30-09-03-36/
│   ├── data.bag
│   └── meta.json
├── template-061707-25-04-30-09-04-28/
│   ├── data.bag
│   └── meta.json
└── ...
```

> **Note**: Each episode directory contains a `data.bag` file (rosbag recording) and `meta.json` file (episode metadata) with HSR-specific topic structure.

## Development

### Code Quality

```bash
# Format code
make format

# Run linting (ruff + mypy)
make lint

# Run tests
make test

# Run tests with coverage
make test-coverage
```

### Available Make Commands

- `make format` - Format code with ruff
- `make lint` - Run linting checks (ruff + mypy)
- `make test` - Run all unit tests
- `make test-coverage` - Run tests with coverage report
- `make ruff-check` - Check code style only
- `make ruff-fix` - Fix code style issues
- `make mypy` - Run type checking

### Testing

```bash
# Run specific test
uv run pytest tests/test_rosbag2lerobot.py -v

# Run with coverage
make test-coverage
```

## Troubleshooting

### Common Issues

1. **Docker build fails**: Ensure Docker and nvidia-docker are properly installed
2. **Memory errors**: Increase Docker memory allocation for large datasets
3. **Permission errors**: Check file permissions and Docker volume mounts
4. **Missing dependencies**: Run `git submodule update --init --recursive`

### Getting Help

- 🐛 Report issues on [GitHub Issues](https://github.com/airoa-org/rebake/issues)

## Contributing

We welcome contributions! Whether you're fixing bugs, adding features, or supporting new robot platforms, your help is appreciated.

**Quick start:**

1. Fork the repository
2. Create a feature branch
3. Make your changes and add tests
4. Run quality checks: `make format && make lint && make test`
5. Open a Pull Request

📋 **For detailed instructions, development setup, and guidelines, please see our [Contributing Guide](CONTRIBUTING.md).**

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

Made with ❤️ by the [AIRoA Team](https://github.com/airoa-org)
