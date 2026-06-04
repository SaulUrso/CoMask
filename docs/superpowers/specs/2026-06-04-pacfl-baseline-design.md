# PACFL Baseline — Design Spec

**Date:** 2026-06-04
**Goal:** Add PACFL (Vahidian et al., 2022) as a comparison baseline in CoMask, reusing the
existing SVD-signature / principal-angle similarity infrastructure. See `ALGORITHM.md` for the
full PACFL algorithm spec.

## Summary

PACFL is clustered federated learning where clustering is decided **once, up front, from data**
via truncated SVD signatures + principal angles, then training proceeds as **per-cluster FedAvg**.

CoMask already has the two pieces PACFL needs most:
- SVD client signatures + min-principal-angle (Eq. 2) similarity matrix
  (`clustering.py::_compute_client_signatures` loop + `calculating_adjacency`).
- Per-cluster FedAvg aggregation + full logging (`ClusterAPI`).

The PACFL-specific pieces that are **missing**:
1. Agglomerative **hierarchical clustering with a distance threshold β** (current code uses
   AffinityPropagation).
2. **Uniform random client sampling** across the whole population (current `ClusterAPI` uses
   stratified per-cluster quota sampling).
3. A PACFL **entry-point script**.
4. **Four sweep configs** varying β.

## Decisions (from brainstorming)

- **Cluster count control:** support **both** β (primary/faithful) and a fixed `n_clusters`
  (no bisection needed under HC — use `criterion="maxclust"`).
- **Linkage:** default `average`, configurable.
- **Newcomers (PME, Algorithm 2/3):** **skip** for now (YAGNI).
- **Angle measure:** **Eq. 2 only** (reuse existing `calculating_adjacency`).
- **Sweep axis:** sweep **β**; log resulting cluster count for calibration.

## Component 1 — Clustering (`src/comask/clustering/clustering.py`)

**Refactor (must be backwards compatible):** extract the per-client SVD-signature loop currently
inside `perform_clustering_svd` (lines ~25-62) into a shared helper:

```python
def _compute_client_signatures(client_data_list, K) -> list[np.ndarray]:
    # exact code currently inside perform_clustering_svd's loop
    # returns U_clients (one (n_features, K*n_classes) matrix per client)
```

`perform_clustering_svd` then calls this helper and is otherwise **unchanged** — the
AffinityPropagation path, the `-calculating_adjacency(...)` negation, `n_clusters` bisection, and
all return values stay identical. **Regression check:** cluster labels for an existing CoMask run
must be identical before/after the refactor.

**New function:**

```python
def perform_clustering_svd_hc(client_data_list, args):
    U_clients = _compute_client_signatures(client_data_list, args.n_basis)
    dist_mat = calculating_adjacency(range(len(U_clients)), U_clients)  # positive, degrees, Eq.2
    np.fill_diagonal(dist_mat, 0)
    labels = cluster_hierarchical(
        dist_mat,
        beta=getattr(args, "beta", None),
        n_clusters=getattr(args, "n_clusters", None),
        linkage_method=getattr(args, "linkage_method", "average"),
    )
    return labels, None, dist_mat
```

Note: PACFL uses the **positive** distance matrix (NOT negated like the AffinityPropagation path).

**New helper:**

```python
def cluster_hierarchical(dist_mat, beta=None, n_clusters=None, linkage_method="average"):
    from scipy.spatial.distance import squareform
    from scipy.cluster.hierarchy import linkage, fcluster
    condensed = squareform(dist_mat, checks=False)
    Z = linkage(condensed, method=linkage_method)
    if n_clusters is not None and int(n_clusters) > 0:
        labels = fcluster(Z, t=int(n_clusters), criterion="maxclust")
    elif beta is not None:
        labels = fcluster(Z, t=float(beta), criterion="distance")
    else:
        raise ValueError("cluster_hierarchical requires either `beta` or `n_clusters`.")
    return labels
```

β is in **degrees** (units of the existing distance matrix, min principal angle ∈ [0, 90]).

## Component 2 — Trainer (`src/comask/servers/pacfl_trainer.py`)

`PACFLClusterAPI(ClusterAPI)` overriding **only** `_client_sampling`:

```python
def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
    np.random.seed(round_idx)
    sampled = np.random.choice(client_num_in_total, client_num_per_round, replace=False)
    group_to_client_indexes = {}
    for c in sampled:
        g = self.group_indexes[c]
        group_to_client_indexes.setdefault(g, []).append(int(c))
    logging.info("PACFL random sample, by group = {}".format(group_to_client_indexes))
    return group_to_client_indexes
```

Everything else inherited unchanged: per-cluster aggregation, `_test_groups`,
`_local_test_on_all_clients`, and all Server/Cluster/Personal/CommCost logging.

**Invariants that keep the inherited `train()` loop correct:**
- Total sampled clients = `client_num_per_round` = `len(self.client_list)`, so the `idx` cursor
  into `client_list` stays in range.
- `w_groups` is pre-initialized for **all** clusters in `train()`; a cluster with zero sampled
  clients in a round is simply absent from the dict that round and keeps its weights (correct
  PACFL behavior). `_test_groups` still evaluates every cluster.

**Behavioral note:** unlike the stratified base sampler, random sampling does NOT guarantee ≥1
client per cluster per round. With many small clusters and a low sampling rate, some clusters
train rarely. This is faithful to PACFL.

## Component 3 — Entry-point script (`scripts/pacfl_script.py`)

Mirror of `scripts/cluster_svd_script.py` with two swaps:
- clustering call → `perform_clustering_svd_hc(...)` (when not `single_cluster`).
- algorithm flow → `PACFLClusterAPI(args, device, dataset, model)`.

Retain: `initialize_and_override_config()`, data loading/partition/`FedMLAdapter`,
`client_num_per_round` fraction handling, `single_cluster` short-circuit, model selection.

**Add:** after clustering, log the resulting cluster count once so β sweeps are interpretable:

```python
n_found = len(set(cluster_labels))
print(f"  - PACFL HC produced {n_found} clusters")
if getattr(args, "enable_wandb", False):
    import wandb
    wandb.log({"Clustering/NumClusters": n_found})
    wandb.run.summary["Clustering/NumClusters"] = n_found
```

## Component 4 — Sweep configs (`sweeps_configs/`)

Four `method: grid`, `program: scripts/pacfl_script.py` files modeled on `comask_*_grid.yaml`.
Primary swept axis = `beta`. Resulting cluster count logged via `Clustering/NumClusters`.

| File | Base config (`--cf`) | Dataset / partition | per-round | comm_round (initial) |
|---|---|---|---|---|
| `pacfl_dir_0.1_cifar.yaml` | `configs/cifar_config.yaml` | cifar10, label/dirichlet α=0.1 | 0.1 | 250 |
| `pacfl_dir_0.5_cifar.yaml` | `configs/cifar_config.yaml` | cifar10, label/dirichlet α=0.5 | 0.1 | 250 |
| `pacfl_femnist.yaml` | `configs/femnist_config.yaml` | femnist, natural (writer_id) | 30 | 250 |
| `pacfl_har.yaml` | `configs/har_config.yaml` | harbox, natural (user_id) | 10 | 250 |

Each fixes: `group_method: clusters`, `linkage_method: average`, `n_basis: 5`, and
dataset-appropriate `learning_rate` / `weight_decay` / `epochs` (taken from the base configs).
Sweeps `beta` over an **initial** grid `[10, 20, 30, 40, 50, 60]` (degrees).

**Calibration note:** these β values are starting guesses. The distance matrix is min principal
angle in degrees ∈ [0, 90]; the β grid that produces a useful range of cluster counts is
dataset-dependent. The first sweep run logs `Clustering/NumClusters` per β — use that to refine
the grid. The CIFAR α=0.1 sweep follows the existing comask grid conventions
(`data_split_alpha: 0.1`, `min_require_size: 1`).

Metric: `Server/FullTest/Acc`, goal maximize (consistent with existing comask grids). Command
block mirrors existing sweeps (`uv run --no-sync --env-file .env -- ${program} --cf <base>`).

## Out of scope

- Newcomer / PME (Algorithm 2 & 3).
- Eq. 3 (sum of paired angles) similarity measure.
- Changes to existing CoMask / CoMask-prune behavior (the clustering refactor must be a no-op for
  those paths).

## Verification

1. **Refactor regression:** existing `perform_clustering_svd` returns identical labels on a fixed
   input before/after extracting `_compute_client_signatures` (unit test on a small synthetic
   `client_data_list`, or compare against a saved run's cluster assignment).
2. **HC clustering:** `cluster_hierarchical` on a synthetic distance matrix returns the expected
   cluster count for a known β and for a fixed `n_clusters`; raises when both are None.
3. **Random sampler:** `PACFLClusterAPI._client_sampling` returns exactly
   `client_num_per_round` clients total, grouped by cluster, deterministic per round.
4. **Smoke run:** `pacfl_script.py` runs a few rounds on cifar with a chosen β, logs
   `Clustering/NumClusters` and the usual Server/Cluster/Personal metrics without error.
