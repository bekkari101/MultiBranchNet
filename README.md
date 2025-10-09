# MultiBranchNet

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.9%2B-yellow)](requirements.txt)

## Parallel Branch Neural Network for MNIST (nums)

This project implements a lightweight Parallel Branch Neural Network (PBNN) for MNIST that achieves strong accuracy with very few parameters. The core idea is to split the flattened convolutional features into equal chunks and process each chunk through a tiny MLP branch. The branch outputs are concatenated and classified. This design reduces parameter count by factorizing dense processing across independent branches; concatenating branch outputs recovers global capacity while keeping per-branch compute small.

Inspired by multi-branch network ideas such as GoogLeNet/Inception modules discussed in Dive into Deep Learning. See: https://d2l.ai/chapter_convolutional-modern/googlenet.html

**Quick results:** best run `run_003` (dated 2025-10-09): 7,210 trainable parameters — 98.56% test accuracy (F1 = 0.9856).

### Method Overview
- **Feature extractor**: A compact CNN with two convolution layers and max-pooling produces a spatial feature map. In the current default config: `conv_channels = (12, 16)`, kernel size 3 with padding 1, and 2× max-pooling.
- **Flatten + padding**: The feature map is flattened. If its size is not divisible by the number of branches, zero-padding is added to make it evenly split.
- **Parallel branches**: The flattened vector is sliced into `branches` equal chunks. Each chunk goes through a small MLP stack with `branch_layers` fully connected layers of width `branch_size`.
- **Concatenate and classify**: All branch outputs are concatenated (size = `branches × branch_size`) and passed to a final linear layer to predict 10 MNIST classes.

This design keeps parameter count low because each branch only sees a small subset of features and uses a tiny MLP, yet the concatenation recovers global capacity.

### Why it’s parameter-efficient
- Convolutional parameters are fixed and small (two small convs).
- Each branch’s first linear layer scales with its input chunk size, not the full flattened feature size.
- Internal branch layers are very small (e.g., width 6), and the final classifier only sees the concatenated small branch outputs.

### Current Configuration (key hyperparameters)
Defined in `code/hyperP.py` via dataclasses.

- `ModelConfig`
  - `conv_channels`: `(12, 16)`
  - `branches`: `6`
  - `branch_size`: `6`
  - `branch_layers`: `2`
  - Derived values: `flat_size` after convs, `pad_size`, `input_branch_size`, `hidden_size = branches × branch_size`

- `TrainingConfig`
  - `num_epochs`: `50`, `batch_size`: `128`, `learning_rate`: `1e-3`
  - `optimizer`: `adam`, `weight_decay`: `1e-4`
  - Scheduler: cosine by default
  - Early stopping enabled

- `DataConfig`
  - Uses MNIST organized under `data/train`, `data/valid`, `data/test`
  - Normalization with MNIST mean/std; augmentation disabled by default

- `ExperimentConfig`
  - `seed`: `42`, `device`: `auto`, optional AMP off by default

### Parameter Budget Target (4k–8k)
The project intentionally constrains the model to a small parameter count. You can compute the trainable parameters from `ModelConfig.get_trainable_parameters()` in `code/hyperP.py`. The recent update uses:

- `branches = 6`
- `branch_size = 6`
- `conv_channels = (12, 16)`

This setting typically lands total parameters in the 7k–8k range.

To move within this budget:
- Decrease parameters: reduce `branches` or `branch_size`, or lower `conv_channels`.
- Increase parameters: increase `branch_size` (most effective), then `branches`, then `conv_channels`.

### File Guide
- `code/train_main.py`: Entry point for training and evaluation (training loop, logging, saving).
- `code/hyperP.py`: All hyperparameters and utilities (dataclasses for model/training/data/logging/experiment; parameter counter; device selection; config summary).
- `code/save_model.py`: Utilities to serialize model artifacts.
- `code/plot.py`: Plotting utilities for training curves and results.
- `code/1_download_data.py`, `code/loadata.py`: Data acquisition and loading helpers if needed.
- `code/result/`: Contains run folders with summaries, logs, checkpoints, plots, and training history.

### How to Run
Prerequisites: Python 3.9+ and a working PyTorch installation compatible with your system/GPU.

```bash
# Clone this repository
git clone https://github.com/bekkari101/MultiBranchNet.git
cd MultiBranchNet

# Install dependencies
python -m pip install -r requirements.txt

# Train
python code/train_main.py

# Optional: print the config summary before training
python code/hyperP.py
```

Download and prepare MNIST into `data/train|valid|test` using:

```bash
python code/1_download_data.py
```

### Reproducing Results
- Each training run writes to `code/result/run_xxx/` with `experiment_summary.json`, `training_history.json`, `plots/`, and `models/`.
- To make your run deterministic, keep `ExperimentConfig.deterministic = True` and fixed `seed`.
- Hardware will default to CUDA if available, otherwise CPU or Apple MPS if detected.

Provenance for the headline metrics:

- Results reported from `code/result/run_003/` (dated 2025-10-09). See `experiment_summary.json` for details.

To reproduce `run_003` behavior:

```bash
# Ensure in code/hyperP.py:
# ExperimentConfig.seed = 42
# ExperimentConfig.deterministic = True
python code/train_main.py
```

### Modifying the Model
Tuning tips to stay in 4k–8k params:
- Lower the budget: try `branches=6, branch_size=4` or `conv_channels=(8,12)`.
- Raise the budget: try `branch_size=8` or `branches=8` (watch the final classifier size = `branches × branch_size × num_classes`).
- Keep `branch_layers=2` unless you compensate by reducing `branch_size`.

---

## Results and Comparative Analysis

### Results (example run)
The following metrics come from `code/result/run_003/experiment_summary.json`:

- **Trainable parameters:** `7,210`
- **Final Test Accuracy:** `98.56%`
- **Final Test F1:** `0.9856`
- **Hidden size:** `36` (`branches=6`, `branch_size=6`)

Plots are saved in `code/result/run_003/plots/`:

- Training curves: `code/result/run_003/plots/training_curves.png`
- Confusion matrix: `code/result/run_003/plots/confusion_matrix.png`
- Per-class metrics: `code/result/run_003/plots/class_metrics.png`

### Comparative Table: MultiBranchNet vs Typical MNIST Baselines (examples)

| Model Type                     | Params     | Param Scale | Test Accuracy | Test F1   | Source |
|--------------------------------|-----------:|------------:|--------------:|----------:|:-------|
| **MultiBranchNet (this repo)** | **7,210**  | 7k          | **98.56%**    | **0.9856**| run_003 (this repo) |
| Simple MLP (2-layers)          | ~100,000   | 100k        | ~98%          | ~0.98     | [Arm PyTorch Example][1] |
| Basic CNN (LeNet-5 variant)    | 60k–200k   | 60–200k     | ~99%          | ~0.99     | [ChanMeng666][2], [Colab][3] |
| Deep CNN                       | >1,000,000 | >1M         | >99.5%        | 0.99–0.995| [GeeksforGeeks][4], [Nextjournal][5] |
| FCNN baseline (external)       | ~118,000   | 118k        | 87.22%        | -         | [HandWritten_Digits_FCNN.ipynb][6] |

#### References

- [1]: [Create a PyTorch model for MNIST | Arm Learning Paths](https://learn.arm.com/learning-paths/cross-platform/pytorch-digit-classification-arch-training/model/)
- [2]: [ChanMeng666/mnist-handwritten-digit-recognition-project](https://github.com/ChanMeng666/MNIST-Handwritten-Digit-Recognition-Project)
- [3]: [MNIST CNN Example on Colab](https://colab.research.google.com/github/Deep-Learning-Challenge/challenge-notebooks/blob/master/2.Convolutional%20Neural%20Networks/2.Guided%20Projects/1.Handwritten%20Digit%20Recognition.ipynb)
- [4]: [MNIST Dataset: Practical Applications Using Keras and PyTorch](https://www.geeksforgeeks.org/machine-learning/mnist-dataset/)
- [5]: [MNIST Handwritten Digit Recognition in PyTorch - Nextjournal](https://nextjournal.com/gkoehler/pytorch-mnist)
- [6]: [HandWritten_Digits_FCNN.ipynb](https://github.com/Ahmad-Ali-Rafique/Handwritten-Digit-Recognition-MNIST/blob/main/HandWritten_Digits_FCNN.ipynb)

### Brief Comparison

- **Parameter Efficiency:** MultiBranchNet achieves nearly state-of-the-art F1 and accuracy with only ~7k parameters, far fewer than classic MLPs or CNNs.
- **Performance:** The F1 score and accuracy closely match those of much larger models, validating the efficiency of the parallel-branch design.
- **Literature Comparison:** Most published MNIST models use tens or hundreds of thousands of parameters for similar scores. Deep CNNs reach slightly higher F1/accuracy (0.99+), but with much higher parameter counts.

---
 
 ## Inspiration and Related Work
 
 - Multi-branch architectures: Our approach is inspired by the idea of processing features along parallel paths and concatenating outputs, akin to GoogLeNet/Inception blocks, but simplified to fully connected branches on flattened conv features. Reference: [GoogLeNet (Multi-Branch Networks) in D2L](https://d2l.ai/chapter_convolutional-modern/googlenet.html).
 - Fully-connected MNIST baselines: Simple FCNNs can be effective but tend to use many more parameters for similar or lower accuracy. Example: an FCNN with ~118k parameters reporting 87.22% accuracy [`HandWritten_Digits_FCNN.ipynb`](https://github.com/Ahmad-Ali-Rafique/Handwritten-Digit-Recognition-MNIST/blob/main/HandWritten_Digits_FCNN.ipynb).
 - Classic CNNs (e.g., LeNet-5 variants) often reach >99% accuracy with 60k–200k+ parameters. MultiBranchNet targets competitive accuracy with a fraction of parameters by factorizing dense processing across branches.

## Environment

- Tested on: Python 3.10, PyTorch 2.1.0 (CUDA-enabled). CPU also works.
- Install deps: `python -m pip install -r requirements.txt`

## Limitations

- MNIST is simple; results may not translate to harder datasets (e.g., CIFAR-10/100).
- Some confusions remain (see confusion matrix under `code/result/run_xxx/plots/`).
- Baseline figures are examples and can vary by preprocessing/architecture; see linked sources.


### Notes
- MNIST folders under `data/` follow class-per-directory convention.
- Augmentations are off by default due to dataset simplicity.
- The code logs progress periodically and can save best/last checkpoints and prediction visualizations when enabled in `LoggingConfig`.

### Citation
If you use or extend this codebase, please cite the repository or include a reference to the Parallel Branch Neural Network approach described above.

### Dataset Credits
- MNIST dataset by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges. See `https://www.kaggle.com/datasets/hojjatk/mnist-dataset`.
- The dataset is used here strictly for research and educational purposes. Please refer to the original authors for licensing and usage terms.

### License
This project is open-sourced under the MIT License. See the `LICENSE` file for details.