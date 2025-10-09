# MultiBranchNet

## Parallel Branch Neural Network for MNIST (nums)

This project implements a lightweight Parallel Branch Neural Network (PBNN) for MNIST that achieves strong accuracy with very few parameters. The core idea is to split the flattened convolutional features into equal chunks and process each chunk through a tiny MLP branch. The branch outputs are concatenated and classified. This factorizes the large dense layer found in standard CNNs into parallel, smaller per-branch layers, dramatically reducing parameters while preserving capacity.

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

### Parameter Budget Target (4k–6k)
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

### Reproducing Results
- Each training run writes to `code/result/run_xxx/` with `experiment_summary.json`, `training_history.json`, `plots/`, and `models/`.
- To make your run deterministic, keep `ExperimentConfig.deterministic = True` and fixed `seed`.
- Hardware will default to CUDA if available, otherwise CPU or Apple MPS if detected.

### Modifying the Model
Tuning tips to stay in 4k–6k params:
- Lower the budget: try `branches=6, branch_size=4` or `conv_channels=(8,12)`.
- Raise the budget: try `branch_size=8` or `branches=8` (watch the final classifier size = `branches × branch_size × num_classes`).
- Keep `branch_layers=2` unless you compensate by reducing `branch_size`.

### Results (example run)
The following metrics come from `code/result/run_003/experiment_summary.json`:

- Trainable parameters: `7,210`
- Final Test Accuracy: `98.56%`
- Final Test F1: `0.9856`
- Hidden size: `36` (`branches=6`, `branch_size=6`)

Plots are saved in `code/result/run_003/plots/`:

- Training curves: `code/result/run_003/plots/training_curves.png`
- Confusion matrix: `code/result/run_003/plots/confusion_matrix.png`
- Per-class metrics: `code/result/run_003/plots/class_metrics.png`

### Brief comparison
- Typical small CNN baselines for MNIST often use 20k–60k+ parameters; classic LeNet-5 variants are commonly >60k. This PBNN achieves comparable accuracy using ~7k parameters by parallelizing a small MLP across feature chunks and concatenating their outputs.

#### Comparative numbers

| Model | Params | Test Accuracy | Notes |
| --- | ---:| ---:| --- |
| Parallel Branch NN (this work) | 7,210 | 98.56% | `branches=6`, `branch_size=6`, `conv=(12,16)` |
| FCNN baseline (external) | ~118,000 | 87.22% | Reported in the referenced FCNN notebook [`HandWritten_Digits_FCNN.ipynb`](https://github.com/Ahmad-Ali-Rafique/Handwritten-Digit-Recognition-MNIST/blob/main/HandWritten_Digits_FCNN.ipynb) |

If you want to include additional external baselines, share links and reported metrics and I’ll add them.

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


 
