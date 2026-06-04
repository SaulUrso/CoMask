import logging

import numpy as np

from comask.servers.cluster_trainer import ClusterAPI


class PACFLClusterAPI(ClusterAPI):
    """
    PACFL training loop.

    Identical to :class:`ClusterAPI` (per-cluster FedAvg aggregation + the same Server / Cluster /
    Personal / CommCost logging) except for **client sampling**: PACFL samples ``R * N`` clients
    uniformly at random from the whole population each round, rather than the stratified
    per-cluster quota sampling of the base class. Sampled clients are then grouped by their (fixed,
    one-shot) cluster id for aggregation.

    A cluster with zero sampled clients in a given round is simply not updated that round and keeps
    its previous weights; ``w_groups`` is pre-initialized for all clusters in ``train()``, so every
    cluster is still evaluated at test time.
    """

    def _client_sampling(self, global_round_idx, client_num_in_total, client_num_per_round):
        # Deterministic per round (matches the base class convention).
        np.random.seed(global_round_idx)

        sampled_client_indexes = np.random.choice(
            client_num_in_total, client_num_per_round, replace=False
        )

        group_to_client_indexes = {}
        for client_idx in sampled_client_indexes:
            group_idx = self.group_indexes[client_idx]
            group_to_client_indexes.setdefault(group_idx, []).append(int(client_idx))

        logging.info("PACFL random sample, client_indexes of each group = {}".format(group_to_client_indexes))
        return group_to_client_indexes
