from typing import Any, Dict, List

import torch


def vote_mask(masks_list: List[Dict[str, Any]], percentage_or_units, min_filters: int = 1) -> Dict[str, Any]:
    """
    Calculate votes for each unit/group across multiple masks and create consolidated mask.

    Args:
        masks_list: List of mask dictionaries from prune() functions (mobilenet, resnet, cnn)
        percentage_or_units: Either percentage of least voted units to remove (float 0.0-1.0)
                           or absolute number of units to remove (int)
        min_filters: Minimum number of filters that must remain in each layer (default: 1)

    Returns:
        dict: Consolidated mask dictionary with same structure as input masks

    Example:
        >>> mask1 = {"conv1": {"mask": torch.tensor([1., 0., 1., 0.]), "layer_type": "conv",
        ...                    "original_filters": 4, "pruned_filters": 2, "indices_kept": np.array([0, 2])}}
        >>> mask2 = {"conv1": {"mask": torch.tensor([1., 1., 0., 0.]), "layer_type": "conv",
        ...                    "original_filters": 4, "pruned_filters": 2, "indices_kept": np.array([0, 1])}}
        >>> consolidated = vote_mask([mask1, mask2], 0.25)  # Remove 25% least voted
        >>> consolidated = vote_mask([mask1, mask2], 2)     # Remove 2 least voted units
    """
    if not masks_list:
        raise ValueError("masks_list cannot be empty")

    # Validate input based on type
    if isinstance(percentage_or_units, int):
        if percentage_or_units < 0:
            raise ValueError("units_to_remove must be non-negative")
    elif isinstance(percentage_or_units, float):
        if not (0.0 <= percentage_or_units <= 1.0):
            raise ValueError("percentage must be between 0.0 and 1.0")
    else:
        raise ValueError("percentage_or_units must be either float (percentage) or int (absolute units)")

    if min_filters < 1:
        raise ValueError("min_filters must be at least 1")

    # Get all unique layer names across all masks
    all_layer_names = set()
    for mask_dict in masks_list:
        all_layer_names.update(mask_dict.keys())

    # Initialize vote counts for each layer
    layer_votes = {}
    layer_info = {}

    # Calculate votes for each unit in each layer
    for layer_name in sorted(list(all_layer_names)):
        # Find the original size by looking at the first mask that has this layer
        original_size = None
        layer_type = None

        for mask_dict in masks_list:
            if layer_name in mask_dict:
                original_size = mask_dict[layer_name]["original_filters"]
                layer_type = mask_dict[layer_name]["layer_type"]
                break

        assert original_size is not None

        # Initialize vote count for this layer
        votes = torch.zeros(original_size, dtype=torch.float32)

        # Count votes from each mask
        for mask_dict in masks_list:
            assert layer_name in mask_dict
            mask_tensor = mask_dict[layer_name]["mask"]
            # Ensure the mask has the correct size (should match original_size)
            assert len(mask_tensor) == original_size
            votes += mask_tensor

        layer_votes[layer_name] = votes
        layer_info[layer_name] = {"original_size": original_size, "layer_type": layer_type}

    # Collect all votes across all layers for global ranking
    all_votes = []
    vote_locations = []  # (layer_name, unit_index) for each vote

    for layer_name, votes in layer_votes.items():
        for unit_idx, vote_count in enumerate(votes):
            all_votes.append(vote_count.item())
            vote_locations.append((layer_name, unit_idx))

    # Sort votes in ascending order (least voted first)
    all_votes = torch.tensor(all_votes)
    sorted_indices = torch.argsort(all_votes)

    # Calculate how many units to remove globally
    total_units = len(all_votes)
    if isinstance(percentage_or_units, int):
        # Absolute number of units to remove
        assert percentage_or_units < total_units - min_filters * len(all_layer_names)
        units_to_remove = percentage_or_units
    else:
        # Percentage of units to remove
        units_to_remove = int(total_units * percentage_or_units) + 1

    # needed to see how many units we are removing, to not remove all of them
    units_to_remove_set = set()
    layer_removal_counts = {layer_name: 0 for layer_name in all_layer_names}
    removed_count = 0
    candidate_idx = 0

    while removed_count < units_to_remove and candidate_idx < len(sorted_indices):
        original_idx = int(sorted_indices[candidate_idx].item())
        layer_name, unit_idx = vote_locations[original_idx]

        # Check if removing this unit would violate min_filters constraint
        original_size = layer_info[layer_name]["original_size"]
        current_removals = layer_removal_counts[layer_name]

        # If removing this unit would leave at least min_filters, remove it
        if (original_size - current_removals - 1) >= min_filters:
            units_to_remove_set.add((layer_name, unit_idx))
            layer_removal_counts[layer_name] += 1
            removed_count += 1

        candidate_idx += 1

    # Create consolidated masks
    consolidated_masks = {}

    for layer_name in all_layer_names:
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
