# MulDar

Multi-static COTS radar implementation with TI AWR2243/1243BOOST

## Hardware

TI AWR2243/1243BOOST, DCA1000EVM

![MulDar System](assets/muldar_sys.png)



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

#### Using Docker

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for GPU access.

```bash
docker build -t muldar .
docker run --gpus all -v $(pwd):/app -it muldar bash
```


This creates a virtual environment and installs all dependencies in one step. (Note: cupy may take longer time for installation)


## Usage

```bash
python play.py
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

By configuring the config.yml, radar stream can be played from either real radar or a recorded file.


## Evaluations 

### For single frame 
```bash
python scripts/single_frame_imaging.py
```

For more evaluations please refer to [evaluations/README.md](evaluations/README.md).


## Cite

Consider cite our mobisys paper!

```tex
coming soon.
```



## Folder Structure

```
MulDar/
├── play.py                      # Main entry point for data acquisition
├── configs.yml                  # Default configuration file
├── pyproject.toml               # Python project & dependency config
├── uv.lock                      # Dependency lock file
├── Dockerfile                   # Docker build file
├── LICENSE
│
├── src/muldar/                  # Core library package
│   ├── configs.py               # Configuration loading & dataclasses
│   ├── utils.py                 # Utility functions
│   ├── devices/                 # Hardware device interfaces
│   │   ├── muldar.py            # Multi-radar system controller
│   │   ├── radar.py             # Single radar (DCA1000EVM) interface
│   │   └── motor.py             # Motor control for SAR scanning
│   ├── dsp/                     # Digital signal processing
│   │   ├── dsp.py               # Core DSP pipeline (range/Doppler FFT)
│   │   ├── bistatic.py          # Bistatic radar processing
│   │   ├── sar.py               # SAR imaging algorithms
│   │   ├── cfar.py              # CFAR detection
│   │   └── music.py             # MUSIC algorithm
│   ├── vis/                     # Visualization
│   │   ├── vis.py               # Plotting utilities
│   │   └── network_vis.py       # Radar network visualization
│   └── eval/                    # Evaluation metrics
│       └── chamfer.py           # Chamfer distance metric
│
├── scripts/                     # Standalone utility scripts
│   ├── calibration.py           # Radar calibration
│   ├── config_dca_eeprom.py     # DCA1000 EEPROM configuration
│   ├── radar_config.py          # Radar parameter configuration
│   └── single_frame_imaging.py  # Single-frame imaging script
│
├── evaluations/                 # Evaluation experiments
│   ├── download_dataset.py      # Dataset download script
│   ├── car/                     # Car imaging evaluation
│   ├── chamfer/                 # Chamfer distance evaluation
│   └── common_objects/          # Common object imaging evaluation
│
├── hardware/                    # Hardware resources
│   ├── antenna.ipynb            # Antenna pattern analysis
│   ├── listen_trigger.py        # Trigger listener for sync
│   ├── cadmodels/               # 3D-printable CAD models (.stl)
│   ├── matlab/                  # MATLAB scripts for mmWaveStudio
│   │   ├── mmWaveStudio/        # Lua configs for AWR1243/2243
│   │   ├── studio_server.m      # MATLAB-mmWaveStudio bridge
│   │   └── *.m                  # RSTD connection scripts
│   └── radar/                   # Radar config templates
│
├── adcData/                     # Raw ADC data (not tracked in git)
│   ├── car/                     # Car scene captures
│   ├── curve/                   # Curved surface captures
│   ├── deformable/              # Deformable object captures
│   ├── drywall/                 # Drywall captures
│   ├── fabrics/                 # Fabric captures
│   ├── metal/                   # Metal surface captures
│   ├── plastic/                 # Plastic surface captures
│   ├── real_object/             # Real object captures
│   └── wood/                    # Wood surface captures
│
└── assets/                      # README images
```
