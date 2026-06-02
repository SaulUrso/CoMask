import copy
import logging
import random

import numpy as np
import torch
import wandb

from fedml import mlops
from fedml.ml.trainer.trainer_creator import create_model_trainer
from .client import Client


class FedAvgAPI(object):
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

        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.client_weights = []

        logging.info("model = {}".format(model))
        self.model_trainer = create_model_trainer(model, args)
        self.model = model
        logging.info("self.model_trainer = {}".format(self.model_trainer))

        self._setup_clients(
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, self.model_trainer,
        )

    def _setup_clients(
        self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer,
    ):
        logging.info("############setup_clients (START)#############")
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

    def _test_server(self, round_idx):
        """
        Test the aggregated model on server side.

        Args:
            round_idx: Current communication round
            use_global_test: If True, use global test data. If False, average local client test results.
        """
        logging.info("################_test_server : {}".format(round_idx))


        test_metrics = self.model_trainer.test(self.test_global, self.device, self.args)

        test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
        test_loss = test_metrics["test_loss"] / test_metrics["test_total"]

        sep_test_loader = getattr(self.args,"sep_test_loader",None)

        if sep_test_loader is not None:
            sep_test_metrics = self.model_trainer.test(sep_test_loader, self.device, self.args)
            sep_test_acc = sep_test_metrics["test_correct"] / sep_test_metrics["test_total"]
            sep_test_loss = sep_test_metrics["test_loss"] / sep_test_metrics["test_total"]

            if self.args.enable_wandb:
                wandb.log({"Server/SepTest/Acc": sep_test_acc, "round": round_idx})
                wandb.log({"Server/SepTest/Loss": sep_test_loss, "round": round_idx})

            mlops.log({"Server/SepTest/Acc": sep_test_acc, "round": round_idx})
            mlops.log({"Server/SepTest/Loss": sep_test_loss, "round": round_idx})

            logging.info({"server_sep_test_acc": sep_test_acc, "server_sep_test_loss": sep_test_loss})

        stats = {"server_test_acc": test_acc, "server_test_loss": test_loss}

        if self.args.enable_wandb:
            wandb.log({"Server/Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Server/Test/Loss": test_loss, "round": round_idx})

        mlops.log({"Server/Test/Acc": test_acc, "round": round_idx})
        mlops.log({"Server/Test/Loss": test_loss, "round": round_idx})

        logging.info(stats)
        return stats

    def _count_model_parameters(self, model_params):
        """Count total number of parameters in the model"""
        total_params = 0
        for param_tensor in model_params.values():
            total_params += param_tensor.numel()
        return total_params

    def train(self):
        logging.info("self.model_trainer = {}".format(self.model_trainer))
        w_global = self.model_trainer.get_model_params()
        self.client_weights = [copy.deepcopy(w_global)] * (self.args.client_num_in_total + 1)
        
        # Calculate model size for communication cost
        total_params = self._count_model_parameters(w_global)
        bits_per_param = 32
        model_size_bits = total_params * bits_per_param
        
        mlops.log_training_status(mlops.ClientConstants.MSG_MLOPS_CLIENT_STATUS_TRAINING)
        mlops.log_aggregation_status(mlops.ServerConstants.MSG_MLOPS_SERVER_STATUS_RUNNING)
        mlops.log_round_info(self.args.comm_round, -1)
        for round_idx in range(self.args.comm_round):

            logging.info("################Communication round : {}".format(round_idx))

            w_locals = []
            
            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset
            """
            client_indexes = self._client_sampling(
                round_idx, self.args.client_num_in_total, self.args.client_num_per_round
            )
            logging.info("client_indexes = " + str(client_indexes))

            for idx, client in enumerate(self.client_list):
                # update dataset
                client_idx = client_indexes[idx]
                client.update_local_dataset(
                    client_idx,
                    self.train_data_local_dict[client_idx],
                    self.test_data_local_dict[client_idx],
                    self.train_data_local_num_dict[client_idx],
                )

                # train on new dataset
                mlops.event("train", event_started=True, event_value="{}_{}".format(str(round_idx), str(idx)))
                w = client.train(copy.deepcopy(w_global))
                mlops.event("train", event_started=False, event_value="{}_{}".format(str(round_idx), str(idx)))
                
                # self.logging.info("local weights = " + str(w))
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
                self.client_weights[client_idx] = copy.deepcopy(w)

            # Log communication costs
            self._log_communication_cost(round_idx, client_indexes, model_size_bits, total_params)

            # update global weights
            mlops.event("agg", event_started=True, event_value=str(round_idx))
            w_global = self._aggregate(w_locals)
            self.client_weights[-1] = w_global

            self.model_trainer.set_model_params(w_global)
            mlops.event("agg", event_started=False, event_value=str(round_idx))

            # test results
            # at last round
            if round_idx == self.args.comm_round - 1:
                self._local_test_on_all_clients(round_idx)
                self._test_server(round_idx)
            # per {frequency_of_the_test} round
            elif round_idx % self.args.frequency_of_the_test == 0:

                self._local_test_on_all_clients(round_idx)
                self._test_server(round_idx)

            mlops.log_round_info(self.args.comm_round, round_idx)

        mlops.log_training_finished_status()
        mlops.log_aggregation_finished_status()

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

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
            for (_, local_w) in w_locals:
                temp_w.append(local_w[k])
            averaged_params[k] = sum(temp_w) / len(temp_w)
        return averaged_params

    def _local_test_on_all_clients(self, round_idx):


        logging.info("################local_test_on_all_clients : {}".format(round_idx))

        train_metrics = {"num_samples": [], "num_correct": [], "losses": []}
        test_metrics = {"num_samples": [], "num_correct": [], "losses": []}
        
        # Add global model metrics
        global_train_metrics = {"num_samples": [], "num_correct": [], "losses": []}
        # global_test_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        client = self.client_list[0]

        for client_idx in range(self.args.client_num_in_total):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """


            client.update_local_dataset(
                0,
                self.train_data_local_dict[client_idx],
                self.test_data_local_dict[client_idx],
                self.train_data_local_num_dict[client_idx],
            )

            # set to local model
            self.model_trainer.set_model_params(self.client_weights[client_idx])

            # train data
            train_local_metrics = client.local_test(False)
            train_metrics["num_samples"].append(copy.deepcopy(train_local_metrics["test_total"]))
            train_metrics["num_correct"].append(copy.deepcopy(train_local_metrics["test_correct"]))
            train_metrics["losses"].append(copy.deepcopy(train_local_metrics["test_loss"]))

            # test data
            if self.test_data_local_dict[client_idx] is not None:
                test_local_metrics = client.local_test(True)
                test_metrics["num_samples"].append(copy.deepcopy(test_local_metrics["test_total"]))
                test_metrics["num_correct"].append(copy.deepcopy(test_local_metrics["test_correct"]))
                test_metrics["losses"].append(copy.deepcopy(test_local_metrics["test_loss"]))

            # Evaluate global model on this client's data
            self.model_trainer.set_model_params(self.client_weights[-1])  # Set to global model
            
            # Global model on train data
            global_train_local_metrics = client.local_test(False)
            global_train_metrics["num_samples"].append(copy.deepcopy(global_train_local_metrics["test_total"]))
            global_train_metrics["num_correct"].append(copy.deepcopy(global_train_local_metrics["test_correct"]))
            global_train_metrics["losses"].append(copy.deepcopy(global_train_local_metrics["test_loss"]))

            # Global model on test data
            # global_test_local_metrics = client.local_test(True)
            # global_test_metrics["num_samples"].append(copy.deepcopy(global_test_local_metrics["test_total"]))
            # global_test_metrics["num_correct"].append(copy.deepcopy(global_test_local_metrics["test_correct"]))
            # global_test_metrics["losses"].append(copy.deepcopy(global_test_local_metrics["test_loss"]))

        # test on training dataset (local models)
        train_acc = sum(train_metrics["num_correct"]) / sum(train_metrics["num_samples"])
        train_loss = sum(train_metrics["losses"]) / sum(train_metrics["num_samples"])

        # Calculate standard deviations for local models on training data
        train_accs_per_client = [correct/samples for correct, samples in zip(train_metrics["num_correct"], train_metrics["num_samples"])]
        train_losses_per_client = [loss/samples for loss, samples in zip(train_metrics["losses"], train_metrics["num_samples"])]
        train_acc_std = np.std(train_accs_per_client)
        train_loss_std = np.std(train_losses_per_client)

        # test on test dataset (local models)
        test_acc = sum(test_metrics["num_correct"]) / sum(test_metrics["num_samples"])
        test_loss = sum(test_metrics["losses"]) / sum(test_metrics["num_samples"])

        # Calculate standard deviations for local models on test data
        test_accs_per_client = [correct/samples for correct, samples in zip(test_metrics["num_correct"], test_metrics["num_samples"])]
        test_losses_per_client = [loss/samples for loss, samples in zip(test_metrics["losses"], test_metrics["num_samples"])]
        test_acc_std = np.std(test_accs_per_client)
        test_loss_std = np.std(test_losses_per_client)

        # Global model metrics on training dataset
        global_train_acc = sum(global_train_metrics["num_correct"]) / sum(global_train_metrics["num_samples"])
        global_train_loss = sum(global_train_metrics["losses"]) / sum(global_train_metrics["num_samples"])

        # Calculate standard deviations for global model on training data
        global_train_accs_per_client = [correct/samples for correct, samples in zip(global_train_metrics["num_correct"], global_train_metrics["num_samples"])]
        global_train_losses_per_client = [loss/samples for loss, samples in zip(global_train_metrics["losses"], global_train_metrics["num_samples"])]
        global_train_acc_std = np.std(global_train_accs_per_client)
        global_train_loss_std = np.std(global_train_losses_per_client)

        # # Global model metrics on test dataset
        # global_test_acc = sum(global_test_metrics["num_correct"]) / sum(global_test_metrics["num_samples"])
        # global_test_loss = sum(global_test_metrics["losses"]) / sum(global_test_metrics["num_samples"])

        stats = {"training_acc": train_acc, "training_loss": train_loss}
        if self.args.enable_wandb:
            wandb.log({"Train/Acc": train_acc, "round": round_idx})
            wandb.log({"Train/Loss": train_loss, "round": round_idx})
            wandb.log({"Train/Acc/Std": train_acc_std, "round": round_idx})
            wandb.log({"Train/Loss/Std": train_loss_std, "round": round_idx})

        mlops.log({"Train/Acc": train_acc, "round": round_idx})
        mlops.log({"Train/Loss": train_loss, "round": round_idx})
        mlops.log({"Train/Acc/Std": train_acc_std, "round": round_idx})
        mlops.log({"Train/Loss/Std": train_loss_std, "round": round_idx})
        logging.info(stats)

        stats = {"test_acc": test_acc, "test_loss": test_loss}
        if self.args.enable_wandb:
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
            wandb.log({"Test/Acc/Std": test_acc_std, "round": round_idx})
            wandb.log({"Test/Loss/Std": test_loss_std, "round": round_idx})

        mlops.log({"Test/Acc": test_acc, "round": round_idx})
        mlops.log({"Test/Loss": test_loss, "round": round_idx})
        mlops.log({"Test/Acc/Std": test_acc_std, "round": round_idx})
        mlops.log({"Test/Loss/Std": test_loss_std, "round": round_idx})
        logging.info(stats)

        # Log global model metrics
        global_stats_train = {"global_training_acc": global_train_acc, "global_training_loss": global_train_loss}
        if self.args.enable_wandb:
            wandb.log({"Global/Train/Acc": global_train_acc, "round": round_idx})
            wandb.log({"Global/Train/Loss": global_train_loss, "round": round_idx})
            wandb.log({"Global/Train/Acc/Std": global_train_acc_std, "round": round_idx})
            wandb.log({"Global/Train/Loss/Std": global_train_loss_std, "round": round_idx})

        mlops.log({"Global/Train/Acc": global_train_acc, "round": round_idx})
        mlops.log({"Global/Train/Loss": global_train_loss, "round": round_idx})
        mlops.log({"Global/Train/Acc/Std": global_train_acc_std, "round": round_idx})
        mlops.log({"Global/Train/Loss/Std": global_train_loss_std, "round": round_idx})
        logging.info(global_stats_train)

        # global_stats_test = {"global_test_acc": global_test_acc, "global_test_loss": global_test_loss}
        # if self.args.enable_wandb:
        #     wandb.log({"Global/Test/Acc": global_test_acc, "round": round_idx})
        #     wandb.log({"Global/Test/Loss": global_test_loss, "round": round_idx})

        # mlops.log({"Global/Test/Acc": global_test_acc, "round": round_idx})
        # mlops.log({"Global/Test/Loss": global_test_loss, "round": round_idx})
        # logging.info(global_stats_test)

    def _log_communication_cost(self, round_idx, participating_client_indexes, model_size_bits, total_params):
        """
        Log communication costs for all clients in the current round.
        
        Args:
            round_idx: Current communication round
            participating_client_indexes: List of client indices that participated in this round
            model_size_bits: Size of the model in bits
            total_params: Total number of parameters in the model
        """
        # Communication cost tracking for all clients
        total_upload_cost = 0  # Total client to server
        total_download_cost = 0  # Total server to client

        # Log communication costs for ALL clients
        for client_idx in range(self.args.client_num_in_total):
            if client_idx in participating_client_indexes:
                # Participating client
                client_upload = model_size_bits
                client_download = model_size_bits
                total_upload_cost += client_upload
                total_download_cost += client_download

                
                client_total = client_upload + client_download
                
                if self.args.enable_wandb:
                    wandb.log({f"CommCost/Client_{client_idx}/Upload": client_upload, "round": round_idx})
                    wandb.log({f"CommCost/Client_{client_idx}/Download": client_download, "round": round_idx})
                    wandb.log({f"CommCost/Client_{client_idx}/Total": client_total, "round": round_idx})
                
                mlops.log({f"CommCost/Client_{client_idx}/Upload": client_upload, "round": round_idx})
                mlops.log({f"CommCost/Client_{client_idx}/Download": client_download, "round": round_idx})
                mlops.log({f"CommCost/Client_{client_idx}/Total": client_total, "round": round_idx})
        
        # Log total costs for the round
        total_round_cost = total_upload_cost + total_download_cost
        
        if self.args.enable_wandb:
            wandb.log({"CommCost/Total/Upload": total_upload_cost, "round": round_idx})
            wandb.log({"CommCost/Total/Download": total_download_cost, "round": round_idx})
            wandb.log({"CommCost/Total/Combined": total_round_cost, "round": round_idx})
            wandb.log({"CommCost/ModelSize/Bits": model_size_bits, "round": round_idx})
            wandb.log({"CommCost/ModelSize/Parameters": total_params, "round": round_idx})
        
        mlops.log({"CommCost/Total/Upload": total_upload_cost, "round": round_idx})
        mlops.log({"CommCost/Total/Download": total_download_cost, "round": round_idx})
        mlops.log({"CommCost/Total/Combined": total_round_cost, "round": round_idx})
        mlops.log({"CommCost/ModelSize/Bits": model_size_bits, "round": round_idx})
        mlops.log({"CommCost/ModelSize/Parameters": total_params, "round": round_idx})
        
        logging.info(f"Communication costs - Round {round_idx}: Total Upload={total_upload_cost} bits, Total Download={total_download_cost} bits, Combined={total_round_cost} bits")

