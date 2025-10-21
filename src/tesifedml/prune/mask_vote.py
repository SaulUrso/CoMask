from typing import Any, Dict, List

import torch


def vote_mask(masks_list: List[Dict[str, Any]], percentage: float) -> Dict[str, Any]:
    """
    Calculate votes for each unit/group across multiple masks and create consolidated mask.

    Args:
        masks_list: List of mask dictionaries from prune() functions (mobilenet, resnet, cnn)
        percentage: Percentage of least voted units to remove (0.0 to 1.0)

    Returns:
        dict: Consolidated mask dictionary with same structure as input masks

    Example:
        >>> mask1 = {"conv1": {"mask": torch.tensor([1., 0., 1., 0.]), "layer_type": "conv",
        ...                    "original_filters": 4, "pruned_filters": 2, "indices_kept": np.array([0, 2])}}
        >>> mask2 = {"conv1": {"mask": torch.tensor([1., 1., 0., 0.]), "layer_type": "conv",
        ...                    "original_filters": 4, "pruned_filters": 2, "indices_kept": np.array([0, 1])}}
        >>> consolidated = vote_mask([mask1, mask2], 0.25)  # Remove 25% least voted
    """
    if not masks_list:
        raise ValueError("masks_list cannot be empty")

    if not (0.0 <= percentage <= 1.0):
        raise ValueError("percentage must be between 0.0 and 1.0")

    # Get all unique layer names across all masks
    all_layer_names = set()
    for mask_dict in masks_list:
        all_layer_names.update(mask_dict.keys())

    # Initialize vote counts for each layer
    layer_votes = {}
    layer_info = {}

    # Calculate votes for each unit in each layer
    for layer_name in all_layer_names:
        # Find the original size by looking at the first mask that has this layer
        original_size = None
        layer_type = None

        for mask_dict in masks_list:
            if layer_name in mask_dict:
                original_size = mask_dict[layer_name]["original_filters"]
                layer_type = mask_dict[layer_name]["layer_type"]
                break

        if original_size is None:
            continue

        # Initialize vote count for this layer
        votes = torch.zeros(original_size, dtype=torch.float32)

        # Count votes from each mask
        for mask_dict in masks_list:
            if layer_name in mask_dict:
                mask_tensor = mask_dict[layer_name]["mask"]
                # Ensure the mask has the correct size (should match original_size)
                if len(mask_tensor) == original_size:
                    votes += mask_tensor
                else:
                    # This shouldn't happen if masks are consistent, but handle gracefully
                    print(
                        f"Warning: mask size mismatch for layer {layer_name}. Expected {original_size}, got {len(mask_tensor)}"
                    )

        layer_votes[layer_name] = votes
        layer_info[layer_name] = {"original_size": original_size, "layer_type": layer_type}

    # Collect all votes across all layers for global ranking
    all_votes = []
    vote_locations: List[tuple[str, int]] = []  # (layer_name, unit_index) for each vote

    for layer_name, votes in layer_votes.items():
        for unit_idx, vote_count in enumerate(votes):
            all_votes.append(vote_count.item())
            vote_locations.append((layer_name, unit_idx))

    # Sort votes in ascending order (least voted first)
    all_votes = torch.tensor(all_votes)
    sorted_indices = torch.argsort(all_votes)

    # Calculate how many units to remove globally
    total_units = len(all_votes)
    units_to_remove = int(total_units * percentage)

    # Determine which units to remove globally
    units_to_remove_set = set()
    for i in range(min(units_to_remove, len(sorted_indices))):
        original_idx = sorted_indices[i].item()
        layer_name, unit_idx = vote_locations[original_idx]
        units_to_remove_set.add((layer_name, unit_idx))

    # Create consolidated masks
    consolidated_masks = {}

    for layer_name in all_layer_names:
        if layer_name not in layer_info:
            continue

        original_size = layer_info[layer_name]["original_size"]
        layer_type = layer_info[layer_name]["layer_type"]

        # Create consolidated mask for this layer
        consolidated_mask = torch.ones(original_size, dtype=torch.float32)

        # Remove units that were selected for removal
        for unit_idx in range(original_size):
            if (layer_name, unit_idx) in units_to_remove_set:
                consolidated_mask[unit_idx] = 0.0

        # Get indices of kept units
        kept_indices = torch.where(consolidated_mask > 0)[0]

        consolidated_masks[layer_name] = {
            "mask": consolidated_mask,
            "layer_type": layer_type,
            "original_filters": original_size,
            "pruned_filters": len(kept_indices),
            "indices_kept": kept_indices.numpy(),
        }

    return consolidated_masks


def print_vote_statistics(masks_list: List[Dict[str, Any]]) -> None:
    """
    Print statistics about votes across multiple masks.

    Args:
        masks_list: List of mask dictionaries from prune() functions
    """
    if not masks_list:
        print("No masks provided")
        return

    print(f"Vote statistics for {len(masks_list)} masks:")
    print("-" * 50)

    # Get all unique layer names
    all_layer_names = set()
    for mask_dict in masks_list:
        all_layer_names.update(mask_dict.keys())

    total_units = 0
    total_votes = 0

    for layer_name in sorted(all_layer_names):
        # Find layer info
        original_size = None
        for mask_dict in masks_list:
            if layer_name in mask_dict:
                original_size = mask_dict[layer_name]["original_filters"]
                break

        if original_size is None:
            continue

        # Calculate votes for this layer
        votes = torch.zeros(original_size, dtype=torch.float32)
        for mask_dict in masks_list:
            if layer_name in mask_dict:
                mask_tensor = mask_dict[layer_name]["mask"]
                if len(mask_tensor) == original_size:
                    votes += mask_tensor

        min_votes = votes.min().item()
        max_votes = votes.max().item()
        mean_votes = votes.mean().item()

        print(f"Layer {layer_name}:")
        print(f"  Units: {original_size}")
        print(f"  Votes per unit - Min: {min_votes:.1f}, Max: {max_votes:.1f}, Mean: {mean_votes:.2f}")
        print(f"  Total votes: {votes.sum().item():.0f}")

        total_units += original_size
        total_votes += votes.sum().item()

    print("-" * 50)
    print(f"Total units across all layers: {total_units}")
    print(f"Total votes across all layers: {total_votes:.0f}")
    print(f"Average votes per unit: {total_votes / total_units:.2f}")
