# Fixed Cluster Count + ssl_coefficient Grid Sweep — Design

Date: 2026-06-04

## Goal

Run `scripts/cluster_prune_script.py` on CIFAR-10 with Dirichlet α=0.5 using
exactly **2, 4, and 8 clusters**, and grid-search `ssl_coefficient` (with
`accuracy_threshold` fixed at 0.2).

Two problems to solve:

1. **AffinityPropagation does not enforce a fixed cluster count.** Today the
   `preference` hyperparameter is tuned by hand until the desired count appears.
   We must keep AffinityPropagation but automate finding the `preference` that
   yields the target number of clusters.
2. **Need a grid sweep** over `ssl_coefficient` at `accuracy_threshold=0.2`,
   `comm_round=250`, `frequency_of_the_test=125`.

## Part 1 — Fix the number of clusters via bisection over `preference`

### New config parameter

`n_clusters` (int), added under `train_args`. Resolution order in
`perform_clustering_svd` / the script:

1. `single_cluster: true` → all clients in cluster 0 (unchanged).
2. `n_clusters` set to a positive int → bisection search for the `preference`
   that yields exactly that many clusters (closest achievable otherwise).
3. Neither set → current behavior: use `preference` (`median` string → `None`,
   or an int). Fully backward-compatible — existing sweeps keep working.

### New function: `find_preference_for_n_clusters`

In `src/comask/clustering/clustering.py`:

```
find_preference_for_n_clusters(sim_mat, target, random_state=42, n_iter=40)
    -> (labels, cluster_centers, used_preference)
```

- The cluster count returned by AffinityPropagation grows (approximately)
  monotonically with `preference`: more negative → fewer clusters, less
  negative → more clusters.
- Search range: `[min, max]` of the off-diagonal similarity values in
  `sim_mat`.
- Each probe calls the existing
  `perform_clustering(sim_mat, "affinity", preference=p)` and counts
  `len(set(labels))`.
- Bisection: if `count < target` raise the lower bound; if `count > target`
  lower the upper bound. Track the best-seen `(preference, labels, centers)` by
  `|count - target|`; early-exit on exact match.
- Returns the closest-count result and **prints a warning** if the exact target
  was not reached.
- Deterministic (`random_state=42`, matching the existing call).

### Wiring

In `perform_clustering_svd`, after `sim_mat` is built (current lines 65–66),
branch on `getattr(args, "n_clusters", None)`:

- If a positive int → call `find_preference_for_n_clusters`.
- Else → existing `perform_clustering(..., preference=preference)` path.

The existing required-fields check still requires `preference` (used by the
fallback path); `n_clusters` is optional.

### Fallback behavior

When bisection cannot hit the exact target (AffinityPropagation plateaus, e.g.
3→5 with no value giving 4), use the **closest achievable** count and log a
warning. The run proceeds.

## Part 2 — Three grid sweep files

Created in `sweeps_configs/`, modeled on
`comask_dir_0.5_cifar_noclust_grid.yaml`:

- `comask_dir_0.5_cifar_2clust_grid.yaml`
- `comask_dir_0.5_cifar_4clust_grid.yaml`
- `comask_dir_0.5_cifar_8clust_grid.yaml`

Differences from the noclust grid template:

| Knob | Value |
|---|---|
| `n_clusters` | `2` / `4` / `8` (per file) |
| `single_cluster` | removed |
| `group_method` | `clusters` |
| `pruning` | `proposal` |
| `ssl_coefficient` | grid axis `[0, 1e-4, 1e-3, 1e-2, 1e-1]` |
| `accuracy_threshold` | `0.2` |
| `comm_round` | `250` |
| `frequency_of_the_test` | `125` |
| `data_split_alpha` | `0.5` |
| `partition_method` / `method_name` / `min_require_size` | `label` / `dirichlet` / `1` |
| `prune_percent` / `consolidation_percentage` | `0.1` / `0.1` (kept equal — utils.py guard) |
| `consensus_percentage` / `prune_counter` | `0.2` / `3` |
| `learning_rate` / `weight_decay` / `client_num_in_total` | `0.1` / `0.0001` / `100` |

Each launched independently:
`wandb sweep sweeps_configs/comask_dir_0.5_cifar_4clust_grid.yaml`.

## Decisions

- Parameter name is `n_clusters` (not `target_clusters`).
- `prune_percent`/`consolidation_percentage` kept at 0.1; this sweep only varies
  `ssl_coefficient` (and fixes `accuracy_threshold=0.2`).
