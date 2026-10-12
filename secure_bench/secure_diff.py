
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tolerance-based screen-change detection for the secure loop.

Compares two masked screenshots (raw PNG bytes) and returns the FRACTION of pixels that changed by
more than a per-channel threshold. Used to tell the actuator when an act had no visible effect.

NOT byte/hash equality: the screenshot composites the cursor, the mask overlay is repainted on
every observe, and first-party pixels animate — so two frames with identical *content* still differ
bit-for-bit. A small area threshold (~0.2%) ignores the cursor and paint noise while catching real
transitions. PIL + numpy are both available in the run venv.
"""
import io


def frac_changed(png_a, png_b, chan_thresh=24):
    """Fraction (0..1) of pixels whose max per-channel abs difference exceeds `chan_thresh`.
    Returns 1.0 (treat as 'changed') if the frames differ in size or cannot be decoded."""
    try:
        import numpy as np
        from PIL import Image
        a = np.asarray(Image.open(io.BytesIO(png_a)).convert("RGB"), dtype=np.int16)
        b = np.asarray(Image.open(io.BytesIO(png_b)).convert("RGB"), dtype=np.int16)
    except Exception:
        return 1.0
    if a.shape != b.shape:
        return 1.0
    diff = np.abs(a - b).max(axis=2)          # per-pixel max channel difference
    return float((diff > chan_thresh).mean())
