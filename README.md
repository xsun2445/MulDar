# MulDar

Multi-monostatic radar data acquisition, processing, and real-time visualization.

## Installation

### Prerequisites

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- CUDA-compatible GPU (required for CuPy)

### Setup

#### Using uv (recommended)

```bash
uv sync
```

This creates a virtual environment and installs all dependencies in one step. (Note: cupy may take longer time for installation)
<!-- 
#### Using pip

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -e .
``` -->

## Usage

```bash
python realtime_vis.py --config-path configs/config_new.yml
```

### Command-line options

| Option | Type | Description |
|---|---|---|
| `--config-path` | str | Path to YAML config file (default: `configs/config_new.yml`) |
| `--flag-save` / `--no-flag-save` | bool | Enable/disable data saving |
| `--flag-visualize` / `--no-flag-visualize` | bool | Enable/disable real-time visualization |
| `--wait-for-threads` / `--no-wait-for-threads` | bool | Wait for threads to finish |
| `--num-trigger` | int | Number of triggers |
| `--period-frame` | int | Frame period |
| `--radar-timeout` | int | Radar timeout |
| `--saving-root-dir` | str | Root directory for saved data |
| `--warmup` / `--no-warmup` | bool | Run warmup before acquisition |

#### Using Docker

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for GPU access.

```bash
docker build -t muldar .
docker run --gpus all muldar --config-path configs/config_new.yml
```