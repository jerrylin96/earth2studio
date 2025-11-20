# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CBottle Video Multi-Frame Conditioning Example
===============================================

This example demonstrates how to use the CBottleVideo model with arbitrary frame
conditioning. Unlike the standard autoregressive rollout (create_iterator), this
allows you to condition the model on any subset of the 12 frames in a sequence.

Use cases:
- Infilling: Condition on frames 0 and 11, generate frames 1-10
- Interpolation: Condition on frames 0, 6, 11, generate intermediate frames
- Partial conditioning: Mix known frames with generated ones

Note: This requires the CBottle model package and SST data to be available.
"""

import numpy as np
import torch
from datetime import datetime

from earth2studio.models.px import CBottleVideo


def example_unconditional_generation():
    """Example: Fully unconditional generation (no frame conditioning)"""
    print("=" * 80)
    print("Example 1: Unconditional Generation")
    print("=" * 80)

    # Load model
    package = CBottleVideo.load_default_package()
    model = CBottleVideo.load_model(package, sampler_steps=18)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Unconditional: empty conditions dict
    conditions = {}
    time = np.datetime64("2022-01-01T00:00")

    # Generate 12 frames
    output, output_coords = model.generate_with_conditioning(conditions, time)

    print(f"Output shape: {output.shape}")
    print(f"Lead times: {output_coords['lead_time']}")
    print(f"Generated {len(output_coords['lead_time'])} frames unconditionally\n")

    return output, output_coords


def example_single_frame_conditioning():
    """Example: Condition on first frame only (similar to create_iterator behavior)"""
    print("=" * 80)
    print("Example 2: Single Frame Conditioning")
    print("=" * 80)

    # Load model
    package = CBottleVideo.load_default_package()
    model = CBottleVideo.load_model(package, sampler_steps=18)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create a conditional frame (frame 0)
    # In practice, this would be real data from ERA5 or ICON
    frame_0 = torch.randn(1, 1, 45, 721, 1440, device=device)

    conditions = {0: frame_0}
    time = np.datetime64("2022-01-01T00:00")

    # Generate 12 frames conditioned on frame 0
    output, output_coords = model.generate_with_conditioning(conditions, time)

    print(f"Output shape: {output.shape}")
    print(f"Conditioned on frame: 0")
    print(f"Generated frames: 1-11")
    print(f"Lead times: {output_coords['lead_time']}\n")

    return output, output_coords


def example_multiframe_conditioning():
    """Example: Condition on multiple frames (e.g., infilling)"""
    print("=" * 80)
    print("Example 3: Multi-Frame Conditioning (Infilling)")
    print("=" * 80)

    # Load model
    package = CBottleVideo.load_default_package()
    model = CBottleVideo.load_model(package, sampler_steps=18)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create conditional frames for start, middle, and end
    # Frame 0 (0h), Frame 6 (36h), Frame 11 (66h)
    frame_0 = torch.randn(1, 1, 45, 721, 1440, device=device)
    frame_6 = torch.randn(1, 1, 45, 721, 1440, device=device)
    frame_11 = torch.randn(1, 1, 45, 721, 1440, device=device)

    conditions = {0: frame_0, 6: frame_6, 11: frame_11}
    time = np.datetime64("2022-01-01T00:00")

    # Generate 12 frames with multiple conditioning points
    output, output_coords = model.generate_with_conditioning(conditions, time)

    print(f"Output shape: {output.shape}")
    print(f"Conditioned on frames: 0, 6, 11")
    print(f"Generated/infilled frames: 1-5, 7-10")
    print(f"Lead times: {output_coords['lead_time']}")
    print("This creates temporal interpolation between the conditioned frames\n")

    return output, output_coords


def example_custom_conditioning_pattern():
    """Example: Custom conditioning pattern"""
    print("=" * 80)
    print("Example 4: Custom Conditioning Pattern")
    print("=" * 80)

    # Load model
    package = CBottleVideo.load_default_package()
    model = CBottleVideo.load_model(package, sampler_steps=18)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Condition on every 3rd frame: 0, 3, 6, 9
    conditions = {}
    for frame_idx in [0, 3, 6, 9]:
        conditions[frame_idx] = torch.randn(1, 1, 45, 721, 1440, device=device)

    time = np.datetime64("2022-01-01T00:00")

    # Generate with custom pattern
    output, output_coords = model.generate_with_conditioning(conditions, time)

    print(f"Output shape: {output.shape}")
    print(f"Conditioned on frames: {sorted(conditions.keys())}")
    unconditioned = [i for i in range(12) if i not in conditions]
    print(f"Generated frames: {unconditioned}")
    print(f"Lead times: {output_coords['lead_time']}")
    print("This allows flexible control over which frames are specified\n")

    return output, output_coords


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("CBottle Video Multi-Frame Conditioning Examples")
    print("=" * 80 + "\n")

    try:
        # Example 1: Unconditional
        example_unconditional_generation()

        # Example 2: Single frame
        example_single_frame_conditioning()

        # Example 3: Multi-frame infilling
        example_multiframe_conditioning()

        # Example 4: Custom pattern
        example_custom_conditioning_pattern()

        print("=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    main()
