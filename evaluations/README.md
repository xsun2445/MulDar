# Evaluations

This folder contains scripts for downloading the MulDar dataset and reproducing the evaluation results from the paper.

## Downloading the Dataset

The dataset is hosted on Hugging Face at [`xinghs/MulDar_Dataset`](https://huggingface.co/datasets/xinghs/MulDar_Dataset). Use the provided download script to fetch it into the `adcData/` directory.


```bash
python evaluations/download_dataset.py
```

### Download a single subset

```bash
python evaluations/download_dataset.py --subset metal
```

### List available subsets

```bash
python evaluations/download_dataset.py --list
```

Available subsets: `metal`, `plastic`, `fabrics`, `drywall`, `wood`, `curve`, `deformable`, `real_object`, `car`

## Evaluations

### 1. Chamfer Distance Evaluation (`chamfer/`)

Quantitative evaluation of imaging accuracy using Chamfer distance between reconstructed radar point clouds and ground-truth shapes.

**What it evaluates:**
- **Planar targets** — flat surfaces (metal, plastic, fabrics, drywall, wood) placed at angles from -45 to +45 degrees. Compares monostatic-only, multistatic-only, and combined (mono + multi) imaging against a ground-truth line segment.
- **Curvature targets** — curved surfaces with curvatures from -5 to +5. Compares against ground-truth arc shapes.

**Metrics reported:** symmetric Chamfer distance (CD), directional CD (pred-to-GT and GT-to-pred).

**Usage:**

```bash
# Run planar evaluation (metal, plastic, fabrics, drywall, wood)
python evaluations/chamfer/eval_chamfer.py --object planar --mode eval

# Visualize planar results
python evaluations/chamfer/eval_chamfer.py --object planar --mode vis

# Run curvature evaluation
python evaluations/chamfer/eval_chamfer.py --object curvature --mode eval

# Visualize curvature results
python evaluations/chamfer/eval_chamfer.py --object curvature --mode vis
```

**Outputs:** JSON result files (`results_*.json`) and comparison plots in `chamfer/chamfer_results/`.

### 2. Car Imaging Evaluation (`car/`)

Bistatic SAR imaging of a full-size car from multiple viewpoints using two radars.

**What it evaluates:**
- Reconstructs bistatic SAR images from 4 viewpoints around a car (left side, left-side back corner, left back corner, backside).
- For each viewpoint, generates all monostatic and bistatic TX-RX pair images, then combines them into mono-sum, multi-sum, and weighted all-sum images.

**Usage:**

```bash
python evaluations/car/bistatic_car.py
```

**Outputs:** per-viewpoint `.npy` image arrays and visualization PNGs in `car/out/`.

### 3. Common Object Imaging Evaluation (`common_objects/`)

Single-frame bistatic SAR imaging of various common objects.

**What it evaluates:**
- Generates radar images for deformable objects (4 scenes), planar metal surfaces (4 angles), curved surfaces (3 curvatures), and real-world objects like a computer (3 scenes).

**Usage:**

```bash
python evaluations/common_objects/eval_objects.py
```

**Outputs:** visualization images in `common_objects/images/`.

## Folder Structure

```
evaluations/
├── README.md
├── download_dataset.py        # Dataset download script
├── car/
│   ├── bistatic_car.py        # Car imaging evaluation
│   ├── car_labels.py          # Radar poses & data paths for car scenes
│   └── out/                   # Output images and .npy files
├── chamfer/
│   ├── eval_chamfer.py        # Chamfer distance evaluation
│   ├── labels.py              # Material labels, data paths, GT geometry
│   ├── chamfer_results/       # Chamfer distance plots
│   └── img_results/           # Per-angle SAR images and point clouds
└── common_objects/
    ├── eval_objects.py         # Common object evaluation
    ├── labels.py               # Object labels and data paths
    └── images/                 # Output images
```
