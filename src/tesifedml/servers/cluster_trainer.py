import logging
from typing import Dict

import numpy as np
from fedml.simulation.sp.hierarchical_fl.client import HFLClient
from fedml.simulation.sp.hierarchical_fl.group import Group
from fedml.simulation.sp.hierarchical_fl.trainer import HierarchicalTrainer


class ClusterAPI(HierarchicalTrainer):
    def _setup_clients(
        self,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        model_trainer,
    ):
        logging.info("############setup_clients (START)#############")
        if self.args.group_method == "random":
            self.group_indexes = np.random.randint(0, self.args.group_num, self.args.client_num_in_total)
        elif self.args.group_method == "clusters":
            self.group_indexes = self.args.cluster_indexes
        else:
            raise Exception(self.args.group_method)

        group_to_client_indexes = {}
        for client_idx, group_idx in enumerate(self.group_indexes):
            if group_idx not in group_to_client_indexes:
                group_to_client_indexes[group_idx] = []
            group_to_client_indexes[group_idx].append(client_idx)

        self.group_dict: Dict[int, Group] = {}
        for group_idx, client_indexes in group_to_client_indexes.items():
            self.group_dict[group_idx] = Group(
                group_idx,
                client_indexes,
                train_data_local_dict,
                test_data_local_dict,
                train_data_local_num_dict,
                self.args,
                self.device,
                self.model,
                self.model_trainer,
            )

        # maintain a dummy client to be used in FedAvgTrainer::local_test_on_all_clients()
        client_idx = -1
        self.client_list = [
            HFLClient(
                client_idx,
                train_data_local_dict[0],
                test_data_local_dict[0],
                train_data_local_num_dict[0],
                self.args,
                self.device,
                self.model,
                self.model_trainer,
            )
        ]
        logging.info("############setup_clients (END)#############")

    def _client_sampling(self, global_round_idx, client_num_in_total, client_num_per_round):
        # Fix numpy seed using round
        np.random.seed(global_round_idx)

        # Calculate proportional quotas for each cluster
        group_sizes = {group_idx: len(group.client_dict) for group_idx, group in self.group_dict.items()}
        proportional_quotas = {
            group_idx: client_num_in_total * (size / client_num_in_total) for group_idx, size in group_sizes.items()
        }

        # Perform rounding to determine quotas
        cluster_sampling_quota = {}
        for group_idx, quota in proportional_quotas.items():
            lower_bound = max(1, int(np.floor(quota)))
            cluster_sampling_quota[group_idx] = lower_bound

        print(f"PRE QUOTAS: {cluster_sampling_quota}")

        # Calculate the fractional parts for each cluster
        fractional_parts = {
            group_idx: quota - max(1, int(np.floor(quota))) for group_idx, quota in proportional_quotas.items()
        }

        # Adjust quotas to ensure the total matches total_clients_to_sample
        current_total = sum(cluster_sampling_quota.values())
        while current_total < client_num_per_round:
            # Calculate probabilities for incrementing based on fractional parts
            incrementable_clusters = [i for i, part in fractional_parts.items() if part > 0]
            if not incrementable_clusters:
                raise ValueError(
                    f"Unable to sample more clients. Current total: {current_total}, Target: {client_num_per_round}"
                )
            probabilities = np.array([fractional_parts[i] for i in incrementable_clusters])
            probabilities = probabilities / probabilities.sum()
            selected_cluster = np.random.choice(incrementable_clusters, p=probabilities)
            cluster_sampling_quota[selected_cluster] += 1
            fractional_parts[selected_cluster] = 0  # Remove the selected cluster from further sampling
            current_total += 1

        print(f"POST QUOTAS: {cluster_sampling_quota}")

        # Sample clients from each cluster based on the adjusted quota
        sampled_client_indexes = []
        for group_id, cluster_quota in enumerate(cluster_sampling_quota):
            sampled_client_indexes.extend(
                np.random.choice([*self.group_dict[group_id].client_dict.keys()], cluster_quota, replace=False).tolist()
            )

        # Ensure sampled_clients contains exactly total_clients_to_sample clients
        assert len(sampled_client_indexes) == self.args.client_num_per_round

        group_to_client_indexes = {}
        for client_idx in sampled_client_indexes:
            group_idx = self.group_indexes[client_idx]
            if group_idx not in group_to_client_indexes:
                group_to_client_indexes[group_idx] = []
            group_to_client_indexes[group_idx].append(client_idx)
        logging.info("client_indexes of each group = {}".format(group_to_client_indexes))
        return group_to_client_indexes

    def train(self):
        # TODO: add personal models, add evaluation like fedavg with commcost. Evaluate all "servers"
        # Initialize separate model states for each group
        w_groups_global = {}
        for group_idx in self.group_dict.keys():
            w_groups_global[group_idx] = self.model.state_dict()

        for global_round_idx in range(self.args.comm_round):
            logging.info("################Global Communication Round : {}".format(global_round_idx))
            group_to_client_indexes = self._client_sampling(
                global_round_idx,
                self.args.client_num_in_total,
                self.args.client_num_per_round,
            )

            # train each group with its own model
            for group_idx in sorted(group_to_client_indexes.keys()):
                sampled_client_indexes = group_to_client_indexes[group_idx]
                group = self.group_dict[group_idx]
                w_group_list = group.train(global_round_idx, w_groups_global[group_idx], sampled_client_indexes)

                # aggregate within the group for each epoch
                for global_epoch, w in w_group_list:
                    w_groups_global[group_idx] = w

                    # evaluate performance for this group
                    if (
                        global_epoch % self.args.frequency_of_the_test == 0
                        or global_epoch == self.args.comm_round * self.args.group_comm_round * self.args.epochs - 1
                    ):
                        self.model.load_state_dict(w_groups_global[group_idx])
                        logging.info(f"Evaluating Group {group_idx} at global epoch {global_epoch}")
                        self._local_test_on_all_clients(global_epoch)
