import copy
import logging
from typing import Dict, List

import numpy as np
from fedml import mlops
from fedml.ml.trainer.trainer_creator import create_model_trainer
from fedml.simulation.sp.fedavg.client import Client

import wandb


class ClusterAPI(object):
    def __init__(self, args, device, dataset, model):
        self.device = device
        self.args = args
        [
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ] = dataset

        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num

        self.client_list: List[Client] = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.client_weights = []

        logging.info("model = {}".format(model))
        self.model_trainer = create_model_trainer(model, args)
        self.model = model
        logging.info("self.model_trainer = {}".format(self.model_trainer))

        self._setup_clients(
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            self.model_trainer,
        )

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
            self.group_indexes: List[int] = self.args.cluster_indexes
        else:
            raise Exception(self.args.group_method)

        group_to_client_indexes: Dict[int, List[int]] = {}
        for client_idx, group_idx in enumerate(self.group_indexes):
            if group_idx not in group_to_client_indexes:
                group_to_client_indexes[group_idx] = []
            group_to_client_indexes[group_idx].append(client_idx)

        self.group_dict = group_to_client_indexes

        for client_idx in range(self.args.client_num_per_round):
            c = Client(
                client_idx,
                train_data_local_dict[client_idx],
                test_data_local_dict[client_idx],
                train_data_local_num_dict[client_idx],
                self.args,
                self.device,
                model_trainer,
            )
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def _client_sampling(self, global_round_idx, client_num_in_total, client_num_per_round):
        # Fix numpy seed using round
        np.random.seed(global_round_idx)

        # Calculate proportional quotas for each cluster
        group_sizes = {group_idx: len(group) for group_idx, group in self.group_dict.items()}
        proportional_quotas = {
            group_idx: client_num_per_round * (size / client_num_in_total) for group_idx, size in group_sizes.items()
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
        for group_id, cluster_quota in cluster_sampling_quota.items():
            sampled_client_indexes.extend(
                np.random.choice([*self.group_dict[group_id]], cluster_quota, replace=False).tolist()
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

    def _test_groups(self, round_idx):
        """
        Test each cluster's aggregated model on server side.

        Args:
            round_idx: Current communication round
        """
        logging.info("################_test_clusters : {}".format(round_idx))

        total_test_correct = 0
        total_test_samples = 0
        total_test_loss = 0

        total_sep_test_correct = 0
        total_sep_test_samples = 0
        total_sep_test_loss = 0

        sep_test_loader = getattr(self.args, "sep_test_loader", None)

        # Evaluate each group's model separately
        for group_idx, group_w in self.w_groups.items():
            # Set model to current group's weights
            self.model_trainer.set_model_params(group_w)

            # Test on global test set
            test_metrics = self.model_trainer.test(self.test_global, self.device, self.args)
            test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
            test_loss = test_metrics["test_loss"] / test_metrics["test_total"]

            # Accumulate for overall statistics
            total_test_correct += test_metrics["test_correct"]
            total_test_samples += test_metrics["test_total"]
            total_test_loss += test_metrics["test_loss"]

            # Log individual group metrics
            if self.args.enable_wandb:
                wandb.log({f"Server/Group_{group_idx}/FullTest/Acc": test_acc, "round": round_idx})
                wandb.log({f"Server/Group_{group_idx}/FullTest/Loss": test_loss, "round": round_idx})

            mlops.log({f"Server/Group_{group_idx}/FullTest/Acc": test_acc, "round": round_idx})
            mlops.log({f"Server/Group_{group_idx}/FullTest/Loss": test_loss, "round": round_idx})

            if sep_test_loader is not None:
                sep_test_metrics = self.model_trainer.test(sep_test_loader, self.device, self.args)
                sep_test_acc = sep_test_metrics["test_correct"] / sep_test_metrics["test_total"]
                sep_test_loss = sep_test_metrics["test_loss"] / sep_test_metrics["test_total"]

                # Accumulate for overall statistics
                total_sep_test_correct += sep_test_metrics["test_correct"]
                total_sep_test_samples += sep_test_metrics["test_total"]
                total_sep_test_loss += sep_test_metrics["test_loss"]

                # Log individual group metrics for separate test
                if self.args.enable_wandb:
                    wandb.log({f"Server/Group_{group_idx}/SepTest/Acc": sep_test_acc, "round": round_idx})
                    wandb.log({f"Server/Group_{group_idx}/SepTest/Loss": sep_test_loss, "round": round_idx})

                mlops.log({f"Server/Group_{group_idx}/SepTest/Acc": sep_test_acc, "round": round_idx})
                mlops.log({f"Server/Group_{group_idx}/SepTest/Loss": sep_test_loss, "round": round_idx})

        # Calculate and log aggregate statistics across all groups
        avg_test_acc = total_test_correct / total_test_samples
        avg_test_loss = total_test_loss / total_test_samples

        if self.args.enable_wandb:
            wandb.log({"Server/FullTest/Acc": avg_test_acc, "round": round_idx})
            wandb.log({"Server/FullTest/Loss": avg_test_loss, "round": round_idx})

        mlops.log({"Server/FullTest/Acc": avg_test_acc, "round": round_idx})
        mlops.log({"Server/FullTest/Loss": avg_test_loss, "round": round_idx})

        if sep_test_loader is not None:
            # Calculate and log aggregate statistics across all groups
            avg_sep_test_acc = total_sep_test_correct / total_sep_test_samples
            avg_sep_test_loss = total_sep_test_loss / total_sep_test_samples

            if self.args.enable_wandb:
                wandb.log({"Server/SepTest/Acc": avg_sep_test_acc, "round": round_idx})
                wandb.log({"Server/SepTest/Loss": avg_sep_test_loss, "round": round_idx})

            mlops.log({"Server/SepTest/Acc": avg_sep_test_acc, "round": round_idx})
            mlops.log({"Server/SepTest/Loss": avg_sep_test_loss, "round": round_idx})

    def _count_model_parameters(self, model_params):
        """Count total number of parameters in the model"""
        total_params = 0
        for param_tensor in model_params.values():
            total_params += param_tensor.numel()
        return total_params

    def train(self):
        logging.info("self.model_trainer = {}".format(self.model_trainer))
        w_global = self.model_trainer.get_model_params()
        self.w_groups = {group_idx: copy.deepcopy(w_global) for group_idx in self.group_dict.keys()}
        self.client_weights = [copy.deepcopy(w_global)] * (self.args.client_num_in_total)

        # Calculate model size for communication cost
        total_params = self._count_model_parameters(w_global)
        bits_per_param = 32
        model_size_bits = total_params * bits_per_param

        mlops.log_training_status(mlops.ClientConstants.MSG_MLOPS_CLIENT_STATUS_TRAINING)
        mlops.log_aggregation_status(mlops.ServerConstants.MSG_MLOPS_SERVER_STATUS_RUNNING)
        mlops.log_round_info(self.args.comm_round, -1)
        for round_idx in range(self.args.comm_round):
            logging.info("################Communication round : {}".format(round_idx))

            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset
            """
            group_to_client_indexes = self._client_sampling(
                round_idx, self.args.client_num_in_total, self.args.client_num_per_round
            )
            logging.info("client_indexes = " + str(group_to_client_indexes))

            idx = 0

            for group_idx, client_indexes in group_to_client_indexes.items():
                w_locals = []
                w_curr_group = self.w_groups[group_idx]
                for client_idx in client_indexes:
                    client = self.client_list[idx]

                    client.update_local_dataset(
                        client_idx,
                        self.train_data_local_dict[client_idx],
                        self.test_data_local_dict[client_idx],
                        self.train_data_local_num_dict[client_idx],
                    )

                    # train on new dataset
                    mlops.event("train", event_started=True, event_value="{}_{}".format(str(round_idx), str(idx)))
                    w = client.train(copy.deepcopy(w_curr_group))
                    mlops.event("train", event_started=False, event_value="{}_{}".format(str(round_idx), str(idx)))

                    # self.logging.info("local weights = " + str(w))
                    w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
                    self.client_weights[client_idx] = copy.deepcopy(w)

                    idx += 1

                # update group weights
                mlops.event("agg", event_started=True, event_value=str(round_idx))
                self.w_groups[group_idx] = self._aggregate(w_locals)

                mlops.event("agg", event_started=False, event_value=str(round_idx))

                # Log communication costs
                self._log_communication_cost(round_idx, group_idx, client_indexes, model_size_bits, total_params)

            # at last round
            if round_idx == self.args.comm_round - 1:
                self._local_test_on_all_clients(round_idx)
                self._test_groups(round_idx)
            # per {frequency_of_the_test} round
            elif round_idx % self.args.frequency_of_the_test == 0:
                self._local_test_on_all_clients(round_idx)
                self._test_groups(round_idx)

            mlops.log_round_info(self.args.comm_round, round_idx)

        mlops.log_training_finished_status()
        mlops.log_aggregation_finished_status()

    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params

    def _aggregate_noniid_avg(self, w_locals):
        """
        The old aggregate method will impact the model performance when it comes to Non-IID setting
        Args:
            w_locals:
        Returns:
        """
        (_, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            temp_w = []
            for _, local_w in w_locals:
                temp_w.append(local_w[k])
            averaged_params[k] = sum(temp_w) / len(temp_w)
        return averaged_params

    def _evaluate_client_with_models(self, client, client_idx):
        """Evaluate a single client with both personal and cluster models."""
        client.update_local_dataset(
            0,
            self.train_data_local_dict[client_idx],
            self.test_data_local_dict[client_idx],
            self.train_data_local_num_dict[client_idx],
        )

        # Personal model evaluation
        self.model_trainer.set_model_params(self.client_weights[client_idx])
        personal_train_metrics = client.local_test(False)

        personal_test_metrics = None
        if self.test_data_local_dict[client_idx] is not None:
            personal_test_metrics = client.local_test(True)

        # Cluster model evaluation
        client_group_idx = self.group_indexes[client_idx]
        self.model_trainer.set_model_params(self.w_groups[client_group_idx])

        cluster_train_metrics = client.local_test(False)
        cluster_test_metrics = None
        if self.test_data_local_dict[client_idx] is not None:
            cluster_test_metrics = client.local_test(True)

        return {
            "personal_train": personal_train_metrics,
            "personal_test": personal_test_metrics,
            "cluster_train": cluster_train_metrics,
            "cluster_test": cluster_test_metrics,
            "group_idx": client_group_idx,
        }

    def _aggregate_metrics(self, all_client_results):
        """Aggregate metrics from all client evaluations."""
        # Initialize metric collections
        personal_train_metrics = {"num_samples": [], "num_correct": [], "losses": []}
        personal_test_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        cluster_train_metrics = {}
        cluster_test_metrics = {}
        for group_idx in self.group_dict.keys():
            cluster_train_metrics[group_idx] = {"num_samples": [], "num_correct": [], "losses": []}
            cluster_test_metrics[group_idx] = {"num_samples": [], "num_correct": [], "losses": []}

        # Aggregate results
        for result in all_client_results:
            # Personal model metrics
            self._add_metrics(personal_train_metrics, result["personal_train"])
            if result["personal_test"] is not None:
                self._add_metrics(personal_test_metrics, result["personal_test"])

            # Cluster model metrics
            group_idx = result["group_idx"]
            self._add_metrics(cluster_train_metrics[group_idx], result["cluster_train"])
            if result["cluster_test"] is not None:
                self._add_metrics(cluster_test_metrics[group_idx], result["cluster_test"])

        return personal_train_metrics, personal_test_metrics, cluster_train_metrics, cluster_test_metrics

    def _add_metrics(self, metrics_dict, local_metrics):
        """Helper to add local metrics to aggregated metrics."""
        metrics_dict["num_samples"].append(copy.deepcopy(local_metrics["test_total"]))
        metrics_dict["num_correct"].append(copy.deepcopy(local_metrics["test_correct"]))
        metrics_dict["losses"].append(copy.deepcopy(local_metrics["test_loss"]))

    def _calculate_metrics_with_std(self, metrics):
        """Calculate accuracy, loss and their standard deviations from metrics."""
        if not metrics["num_samples"]:
            return None, None, None, None

        total_acc = sum(metrics["num_correct"]) / sum(metrics["num_samples"])
        total_loss = sum(metrics["losses"]) / sum(metrics["num_samples"])

        accs_per_client = [
            correct / samples for correct, samples in zip(metrics["num_correct"], metrics["num_samples"])
        ]
        losses_per_client = [loss / samples for loss, samples in zip(metrics["losses"], metrics["num_samples"])]

        acc_std = np.std(accs_per_client)
        loss_std = np.std(losses_per_client)

        return total_acc, total_loss, acc_std, loss_std

    def _log_personal_metrics(self, train_metrics, test_metrics, round_idx):
        """Log personal model metrics."""
        train_acc, train_loss, train_acc_std, train_loss_std = self._calculate_metrics_with_std(train_metrics)

        if self.args.enable_wandb:
            wandb.log({"Personal/Train/Acc": train_acc, "round": round_idx})
            wandb.log({"Personal/Train/Loss": train_loss, "round": round_idx})
            wandb.log({"Personal/Train/Acc/Std": train_acc_std, "round": round_idx})
            wandb.log({"Personal/Train/Loss/Std": train_loss_std, "round": round_idx})

        mlops.log({"Personal/Train/Acc": train_acc, "round": round_idx})
        mlops.log({"Personal/Train/Loss": train_loss, "round": round_idx})
        mlops.log({"Personal/Train/Acc/Std": train_acc_std, "round": round_idx})
        mlops.log({"Personal/Train/Loss/Std": train_loss_std, "round": round_idx})

        logging.info({"training_acc": train_acc, "training_loss": train_loss})

        if test_metrics["num_samples"]:
            test_acc, test_loss, test_acc_std, test_loss_std = self._calculate_metrics_with_std(test_metrics)

            if self.args.enable_wandb:
                wandb.log({"Personal/Test/Acc": test_acc, "round": round_idx})
                wandb.log({"Personal/Test/Loss": test_loss, "round": round_idx})
                wandb.log({"Personal/Test/Acc/Std": test_acc_std, "round": round_idx})
                wandb.log({"Personal/Test/Loss/Std": test_loss_std, "round": round_idx})

            mlops.log({"Personal/Test/Acc": test_acc, "round": round_idx})
            mlops.log({"Personal/Test/Loss": test_loss, "round": round_idx})
            mlops.log({"Personal/Test/Acc/Std": test_acc_std, "round": round_idx})
            mlops.log({"Personal/Test/Loss/Std": test_loss_std, "round": round_idx})

            logging.info({"test_acc": test_acc, "test_loss": test_loss})

    def _log_cluster_metrics(self, cluster_train_metrics, cluster_test_metrics, round_idx):
        """Log cluster model metrics for each group."""
        for group_idx in self.group_dict.keys():
            # Train metrics
            train_acc, train_loss, train_acc_std, train_loss_std = self._calculate_metrics_with_std(
                cluster_train_metrics[group_idx]
            )

            if self.args.enable_wandb:
                wandb.log({f"Cluster/Group_{group_idx}/Train/Acc": train_acc, "round": round_idx})
                wandb.log({f"Cluster/Group_{group_idx}/Train/Loss": train_loss, "round": round_idx})
                wandb.log({f"Cluster/Group_{group_idx}/Train/Acc/Std": train_acc_std, "round": round_idx})
                wandb.log({f"Cluster/Group_{group_idx}/Train/Loss/Std": train_loss_std, "round": round_idx})

            mlops.log({f"Cluster/Group_{group_idx}/Train/Acc": train_acc, "round": round_idx})
            mlops.log({f"Cluster/Group_{group_idx}/Train/Loss": train_loss, "round": round_idx})
            mlops.log({f"Cluster/Group_{group_idx}/Train/Acc/Std": train_acc_std, "round": round_idx})
            mlops.log({f"Cluster/Group_{group_idx}/Train/Loss/Std": train_loss_std, "round": round_idx})

            # Test metrics
            if cluster_test_metrics[group_idx]["num_samples"]:
                test_acc, test_loss, test_acc_std, test_loss_std = self._calculate_metrics_with_std(
                    cluster_test_metrics[group_idx]
                )

                if self.args.enable_wandb:
                    wandb.log({f"Cluster/Group_{group_idx}/Test/Acc": test_acc, "round": round_idx})
                    wandb.log({f"Cluster/Group_{group_idx}/Test/Loss": test_loss, "round": round_idx})
                    wandb.log({f"Cluster/Group_{group_idx}/Test/Acc/Std": test_acc_std, "round": round_idx})
                    wandb.log({f"Cluster/Group_{group_idx}/Test/Loss/Std": test_loss_std, "round": round_idx})

                mlops.log({f"Cluster/Group_{group_idx}/Test/Acc": test_acc, "round": round_idx})
                mlops.log({f"Cluster/Group_{group_idx}/Test/Loss": test_loss, "round": round_idx})
                mlops.log({f"Cluster/Group_{group_idx}/Test/Acc/Std": test_acc_std, "round": round_idx})
                mlops.log({f"Cluster/Group_{group_idx}/Test/Loss/Std": test_loss_std, "round": round_idx})

    def _log_server_metrics(self, cluster_train_metrics, cluster_test_metrics, round_idx):
        """Log server-level metrics (aggregated across all clusters)."""
        server_train_total_samples = 0
        server_train_total_correct = 0
        server_train_total_loss = 0

        server_test_total_samples = 0
        server_test_total_correct = 0
        server_test_total_loss = 0

        for group_idx in self.group_dict.keys():
            # Aggregate train metrics
            server_train_total_samples += sum(cluster_train_metrics[group_idx]["num_samples"])
            server_train_total_correct += sum(cluster_train_metrics[group_idx]["num_correct"])
            server_train_total_loss += sum(cluster_train_metrics[group_idx]["losses"])

            # Aggregate test metrics
            if cluster_test_metrics[group_idx]["num_samples"]:
                server_test_total_samples += sum(cluster_test_metrics[group_idx]["num_samples"])
                server_test_total_correct += sum(cluster_test_metrics[group_idx]["num_correct"])
                server_test_total_loss += sum(cluster_test_metrics[group_idx]["losses"])

        # Log train metrics
        server_train_acc = server_train_total_correct / server_train_total_samples
        server_train_loss = server_train_total_loss / server_train_total_samples

        if self.args.enable_wandb:
            wandb.log({"Server/Train/Acc": server_train_acc, "round": round_idx})
            wandb.log({"Server/Train/Loss": server_train_loss, "round": round_idx})

        mlops.log({"Server/Train/Acc": server_train_acc, "round": round_idx})
        mlops.log({"Server/Train/Loss": server_train_loss, "round": round_idx})

        # Log test metrics if available
        if server_test_total_samples > 0:
            server_test_acc = server_test_total_correct / server_test_total_samples
            server_test_loss = server_test_total_loss / server_test_total_samples

            if self.args.enable_wandb:
                wandb.log({"Server/Test/Acc": server_test_acc, "round": round_idx})
                wandb.log({"Server/Test/Loss": server_test_loss, "round": round_idx})

            mlops.log({"Server/Test/Acc": server_test_acc, "round": round_idx})
            mlops.log({"Server/Test/Loss": server_test_loss, "round": round_idx})

    def _local_test_on_all_clients(self, round_idx):
        logging.info("################local_test_on_all_clients : {}".format(round_idx))

        # Evaluate all clients with both personal and cluster models
        client = self.client_list[0]
        all_client_results = []

        for client_idx in range(self.args.client_num_in_total):
            result = self._evaluate_client_with_models(client, client_idx)
            all_client_results.append(result)

        # Aggregate metrics from all evaluations
        personal_train_metrics, personal_test_metrics, cluster_train_metrics, cluster_test_metrics = (
            self._aggregate_metrics(all_client_results)
        )

        # Log all metrics
        self._log_personal_metrics(personal_train_metrics, personal_test_metrics, round_idx)
        self._log_cluster_metrics(cluster_train_metrics, cluster_test_metrics, round_idx)
        self._log_server_metrics(cluster_train_metrics, cluster_test_metrics, round_idx)

    def _log_communication_cost(
        self, round_idx, group_idx, participating_client_indexes, model_size_bits, total_params
    ):
        """
        Log communication costs for all clients in the current round for a specific group.

        Args:
            round_idx: Current communication round
            group_idx: Group/cluster index
            participating_client_indexes: List of client indices that participated in this round
            model_size_bits: Size of the model in bits
            total_params: Total number of parameters in the model
        """
        # Communication cost tracking for all clients
        total_upload_cost = 0  # Total client to server
        total_download_cost = 0  # Total server to client

        # Log communication costs for participating clients in this group
        for client_idx in participating_client_indexes:
            # Participating client
            client_upload = model_size_bits
            client_download = model_size_bits
            total_upload_cost += client_upload
            total_download_cost += client_download

            client_total = client_upload + client_download

            if self.args.enable_wandb:
                wandb.log({f"CommCost/Group_{group_idx}/Client_{client_idx}/Upload": client_upload, "round": round_idx})
                wandb.log(
                    {f"CommCost/Group_{group_idx}/Client_{client_idx}/Download": client_download, "round": round_idx}
                )
                wandb.log({f"CommCost/Group_{group_idx}/Client_{client_idx}/Total": client_total, "round": round_idx})

            mlops.log({f"CommCost/Group_{group_idx}/Client_{client_idx}/Upload": client_upload, "round": round_idx})
            mlops.log({f"CommCost/Group_{group_idx}/Client_{client_idx}/Download": client_download, "round": round_idx})
            mlops.log({f"CommCost/Group_{group_idx}/Client_{client_idx}/Total": client_total, "round": round_idx})

        # Log total costs for the group in this round
        total_round_cost = total_upload_cost + total_download_cost

        if self.args.enable_wandb:
            wandb.log({f"CommCost/Group_{group_idx}/Total/Upload": total_upload_cost, "round": round_idx})
            wandb.log({f"CommCost/Group_{group_idx}/Total/Download": total_download_cost, "round": round_idx})
            wandb.log({f"CommCost/Group_{group_idx}/Total/Combined": total_round_cost, "round": round_idx})
            wandb.log({f"CommCost/Group_{group_idx}/ModelSize/Bits": model_size_bits, "round": round_idx})
            wandb.log({f"CommCost/Group_{group_idx}/ModelSize/Parameters": total_params, "round": round_idx})

        mlops.log({f"CommCost/Group_{group_idx}/Total/Upload": total_upload_cost, "round": round_idx})
        mlops.log({f"CommCost/Group_{group_idx}/Total/Download": total_download_cost, "round": round_idx})
        mlops.log({f"CommCost/Group_{group_idx}/Total/Combined": total_round_cost, "round": round_idx})
        mlops.log({f"CommCost/Group_{group_idx}/ModelSize/Bits": model_size_bits, "round": round_idx})
        mlops.log({f"CommCost/Group_{group_idx}/ModelSize/Parameters": total_params, "round": round_idx})

        logging.info(
            f"Communication costs - Round {round_idx}, Group {group_idx}: Total Upload={total_upload_cost} bits, Total Download={total_download_cost} bits, Combined={total_round_cost} bits"
        )
