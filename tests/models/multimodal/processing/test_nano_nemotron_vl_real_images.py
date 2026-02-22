# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Compare slow (PIL BICUBIC) vs fast (cv2) preprocessing on real images.

Usage:
    pytest tests/models/multimodal/processing/test_nano_nemotron_vl_real_images.py \
        --image-dir /path/to/images -v -s
"""

import sys
from pathlib import Path

import pytest
import torch
from tqdm import tqdm

from vllm.model_executor.models.nano_nemotron_vl import dynamic_preprocess

TOLERANCE = 0.06


def collect_image_paths(directory: str) -> list[Path]:
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")
    paths = [p for p in root.rglob("*")
        if p.suffix.lower() == ".jpg"
    ]
    return paths


def preprocess_pair(image, image_size=512, max_num_tiles=12):
    slow = dynamic_preprocess(
        image,
        image_size=image_size,
        max_num_tiles=max_num_tiles,
        use_thumbnail=False,
        fast_preprocess=False,
    )
    fast = dynamic_preprocess(
        image,
        image_size=image_size,
        max_num_tiles=max_num_tiles,
        use_thumbnail=False,
        fast_preprocess=True,
    )
    return slow, fast


def compare_patches(slow_patches, fast_patches, label=""):
    assert len(slow_patches) == len(fast_patches), (
        f"{label}: patch count {len(slow_patches)} vs {len(fast_patches)}"
    )
    max_diff = 0.0
    mean_diff = 0.0
    for s, f in zip(slow_patches, fast_patches):
        assert s.shape == f.shape, f"{label}: shape {s.shape} vs {f.shape}"
        abs_diff = (s - f).abs()
        max_diff = max(max_diff, abs_diff.max().item())
        mean_diff += abs_diff.mean().item()
    mean_diff /= len(slow_patches)
    return max_diff, mean_diff


# ---- pytest fixtures / hooks ----

def pytest_addoption(parser):
    parser.addoption(
        "--image-dir",
        action="store",
        default=None,
        help="Directory containing images to test",
    )


@pytest.fixture(scope="session")
def image_dir(request):
    d = request.config.getoption("--image-dir")
    if d is None:
        pytest.skip("--image-dir not provided")
    return d


@pytest.fixture(scope="session")
def image_paths(image_dir):
    paths = collect_image_paths(image_dir)
    if not paths:
        pytest.skip(f"No images found in {image_dir}")
    return paths


# ---- tests ----

class TestRealImages:

    def test_all_images(self, image_paths):
        from PIL import Image

        results = []
        pbar = tqdm(image_paths, desc="Comparing slow/fast", unit="img")
        for path in pbar:
            pbar.set_postfix_str(path.name[-30:])
            img = Image.open(path).convert("RGB")
            slow, fast = preprocess_pair(img)
            max_d, mean_d = compare_patches(slow, fast, label=path.name)
            results.append((path.name, img.size, len(slow), max_d, mean_d))
            assert max_d < TOLERANCE, (
                f"{path.name}: max_diff {max_d:.6f} >= {TOLERANCE}"
            )

        print(f"\n  Processed {len(results)} images, all within tolerance.")

    def test_per_image(self, image_paths):
        """Parametrize at runtime so each image is a separate sub-test."""
        from PIL import Image

        failures = []
        for path in tqdm(image_paths, desc="Per-image check", unit="img"):
            img = Image.open(path).convert("RGB")
            slow, fast = preprocess_pair(img)
            max_d, _ = compare_patches(slow, fast, label=path.name)
            if max_d >= TOLERANCE:
                failures.append((path.name, max_d))

        assert not failures, (
            f"{len(failures)} image(s) exceeded tolerance:\n"
            + "\n".join(f"  {n}: {d:.6f}" for n, d in failures)
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} /path/to/images")
        sys.exit(1)

    from PIL import Image

    image_dir = sys.argv[1]
    print("Testing images in directory: ", image_dir)
    paths = collect_image_paths(image_dir)
    print(f"Found {len(paths)} images in {image_dir}\n")

    for path in tqdm(paths, desc="Processing", unit="img"):
        img = Image.open(path).convert("RGB")
        slow, fast = preprocess_pair(img)
        max_d, mean_d = compare_patches(slow, fast, label=path.name)
        status = "OK" if max_d < TOLERANCE else "FAIL"
        tqdm.write(
            f"[{status}] {path.name:>40s}  {str(img.size):>14s}  "
            f"patches={len(slow):>2d}  "
            f"max_diff={max_d:.6f}  mean_diff={mean_d:.6f}"
        )
