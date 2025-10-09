import os
from pathlib import Path
from typing import Tuple

from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import random_split, Subset


def prepare_mnist_datasets(download_root: Path) -> Tuple[Subset, Subset, datasets.MNIST]:
    """Download MNIST and create 45k/15k train/valid splits plus the original 10k test.

    Returns (train_subset, valid_subset, test_dataset).
    """
    download_root.mkdir(parents=True, exist_ok=True)
    full_train = datasets.MNIST(root=str(download_root), train=True, download=True)
    test = datasets.MNIST(root=str(download_root), train=False, download=True)

    train_size = 45_000
    valid_size = 15_000
    assert len(full_train) == 60_000, "Expected 60k training images in MNIST"
    assert train_size + valid_size == len(full_train)

    train_subset, valid_subset = random_split(
        full_train, [train_size, valid_size], generator=None
    )
    return train_subset, valid_subset, test


def save_split_images(split_name: str, dataset: Subset | datasets.MNIST, out_root: Path) -> None:
    """Save images from a dataset/subset into label subfolders under out_root/split_name.

    Filenames include the original index when available to keep them stable across runs.
    """
    split_dir = out_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    # Pre-create label directories 0-9
    for label in range(10):
        (split_dir / str(label)).mkdir(parents=True, exist_ok=True)

    has_indices = isinstance(dataset, Subset) and hasattr(dataset, "indices")

    with tqdm(total=len(dataset), desc=f"Saving {split_name}") as progress:
        for i in range(len(dataset)):
            img, label = dataset[i]
            label_dir = split_dir / str(int(label))

            # Prefer original dataset index if available
            orig_idx = dataset.indices[i] if has_indices else i
            filename = f"{split_name}_{orig_idx:05d}.png"
            img_path = label_dir / filename

            # torchvision MNIST returns PIL Images by default when no transform is set
            img.save(img_path)
            progress.update(1)


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    download_root = project_root / "data" / "_torchvision_cache"
    output_root = project_root / "data"

    train_ds, valid_ds, test_ds = prepare_mnist_datasets(download_root)

    save_split_images("train", train_ds, output_root)
    save_split_images("valid", valid_ds, output_root)
    save_split_images("test", test_ds, output_root)


if __name__ == "__main__":
    # Ensure consistent working directory irrespective of how the script is invoked
    os.chdir(Path(__file__).resolve().parent)
    main()


