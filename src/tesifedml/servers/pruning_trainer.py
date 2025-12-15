import copy
import logging
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from fedml import mlops
from fedml.simulation.sp.fedavg.fedavg_api import FedAvgAPI
from torch import nn

import wandb
from tesifedml.aggregators.unified_aggregate import aggregate_model
from tesifedml.clients.prune_client import ModelTrainerSSL, PruneClient
from tesifedml.prune.unified_prune import prune_model
from tesifedml.prune.utils import combine_mask


class PruningTrainerAPI(FedAvgAPI):
    """
    Federated learning with immediate client-side pruning (no clustering).

    This approach extends FedAvgAPI to support per-client pruning decisions.
    Each client independently decides when to prune based on validation accuracy.

    Key features:
    - Each client maintains its own prune counter (default=3)
    - Clients prune when: counter > 0 AND validation accuracy < threshold
    - Different clients can have different pruned models at the same time
    - Uses unified_aggregate to handle heterogeneous model structures
    - Client masks are cumulative (combined using combine_mask) to track pruning w.r.t. original model

    Methods overridden from FedAvgAPI:
    - __init__: Add validation data support and pruning state tracking
    - _setup_clients: Use PruneClient instead of Client, initialize pruning counters
    - train: Implement pruning decision logic and use unified aggregation
    - _local_test_on_all_clients: Use _get_client_training_weights to apply pruning masks

    Required args:
    - pruning: Must be set to "immediate"
    - accuracy_threshold: Threshold below which clients will prune
    - prune_percent: Percentage of filters/channels to prune
    - client_prune_counter: (optional) Number of times each client can prune (default=3)
    - prune_way: (optional) Pruning method (default="group_lasso")
    - minimum_channels: (optional) Minimum channels to keep (default=1)
    - divisor: (optional) Divisor for channel rounding (default=1)
    """

    def __init__(self, args, device, dataset, model: nn.Module):
        # Initialize parent class first
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
            val_data_local_dict,
        ] = dataset

        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num

        self.client_list: List[PruneClient] = []  # type:ignore
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.val_data_local_dict = val_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.client_weights = []

        logging.info("model = {}".format(model))

        if args.ssl_coefficient > 0.0:
            self.model_trainer = ModelTrainerSSL(model, args)
            logging.info("Using ModelTrainerSSL with ssl_coefficient = {}".format(args.ssl_coefficient))
        else:
            from fedml.ml.trainer.trainer_creator import create_model_trainer

            self.model_trainer = create_model_trainer(model, args)
            logging.info("Using standard model trainer")

        self.model = model

        # Track client-specific information
        self.client_prune_counters: Dict[int, int] = {}  # client_id -> remaining prunes
        self.client_masks: Dict[int, Optional[Dict]] = {}  # client_id -> mask (relative to original)
        self.client_pruned_models: Dict[int, Optional[nn.Module]] = {}  # client_id -> pruned model structure

        logging.info("self.model_trainer = {}".format(self.model_trainer))

        self._setup_clients(
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            self.model_trainer,
            val_data_local_dict=val_data_local_dict,
        )

    def run(self):
        self.train()

    def _setup_clients(
        self,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        model_trainer,
        val_data_local_dict=None,
    ):
        """Setup clients with validation data support and initialize pruning counters."""
        assert val_data_local_dict is not None, "Validation data is required for pruning decisions"

        logging.info("############setup_clients (START)#############")

        # Initialize client pruning counters and masks
        prune_counter_default = getattr(self.args, "client_prune_counter", 3)
        for client_idx in range(self.args.client_num_in_total):
            self.client_prune_counters[client_idx] = prune_counter_default
            self.client_masks[client_idx] = None
            # Keep client models on CPU
            self.client_pruned_models[client_idx] = copy.deepcopy(self.model).cpu()

        for client_idx in range(self.args.client_num_per_round):
            c = PruneClient(
                client_idx,
                train_data_local_dict[client_idx],
                val_data_local_dict[client_idx],
                test_data_local_dict[client_idx],
                train_data_local_num_dict[client_idx],
                self.args,
                self.device,
                model_trainer,
            )
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def train(self):
        """Main training loop with per-client pruning decisions."""
        logging.info("self.model_trainer = {}".format(self.model_trainer))
        w_global = self.model_trainer.get_model_params()
        self.client_weights = [copy.deepcopy(w_global)] * (self.args.client_num_in_total)

        mlops.log_training_status(mlops.ClientConstants.MSG_MLOPS_CLIENT_STATUS_TRAINING)
        mlops.log_aggregation_status(mlops.ServerConstants.MSG_MLOPS_SERVER_STATUS_RUNNING)
        mlops.log_round_info(self.args.comm_round, -1)

        for round_idx in range(self.args.comm_round):
            round_start_time = time.time()
            logging.info("################Communication round : {}".format(round_idx))

            # Profile: Client sampling
            t_start = time.time()
            client_indexes = self._client_sampling(
                round_idx, self.args.client_num_in_total, self.args.client_num_per_round
            )
            t_sampling = time.time() - t_start
            logging.info("client_indexes = " + str(client_indexes))

            w_locals = []
            masks_locals = []
            counters_locals = []
            pruned_this_round = {}  # Track which clients pruned this round and their model sizes before/after

            # Profile: Client training times
            client_times = {}
            total_prune_decision_time = 0
            total_prune_execution_time = 0
            total_model_update_time = 0
            total_training_time = 0

            for idx, client_idx in enumerate(client_indexes):
                client_start_time = time.time()
                client = self.client_list[idx]

                client.update_local_dataset(
                    client_idx,
                    self.train_data_local_dict[client_idx],
                    self.test_data_local_dict[client_idx],
                    self.train_data_local_num_dict[client_idx],
                    local_val_data=self.val_data_local_dict[client_idx],
                )

                # Check if client should prune before training
                t_prune_decision_start = time.time()
                should_prune = self._should_client_prune(client_idx, round_idx)
                t_prune_decision = time.time() - t_prune_decision_start
                total_prune_decision_time += t_prune_decision

                if should_prune:
                    t_prune_exec_start = time.time()
                    # Record model size before pruning for communication cost calculation
                    model_size_before = self._count_model_parameters(self.client_pruned_models[client_idx].state_dict())  # type: ignore
                    # Client prunes immediately before training
                    logging.info(f"Client {client_idx} pruning at round {round_idx}")
                    self._prune_client_model(client_idx)

                    # Record model size after pruning
                    model_size_after = self._count_model_parameters(self.client_pruned_models[client_idx].state_dict())  # type: ignore
                    pruned_this_round[client_idx] = {
                        "before": model_size_before,
                        "after": model_size_after,
                    }
                    t_prune_exec = time.time() - t_prune_exec_start
                    total_prune_execution_time += t_prune_exec

                # Update client's model trainer to use the correct structure (pruned or full)
                t_model_update_start = time.time()
                self._update_client_model_structure(client, client_idx)

                # Move client's model to GPU for training
                client.model_trainer.model = client.model_trainer.model.to(self.device)

                # Get the client's training weights (for the correct structure)
                client_train_weights = self._get_client_training_weights(client_idx, w_global)
                t_model_update = time.time() - t_model_update_start
                total_model_update_time += t_model_update

                # Train on the client's (possibly pruned) model
                t_train_start = time.time()
                mlops.event("train", event_started=True, event_value="{}_{}".format(str(round_idx), str(idx)))
                w = client.train(copy.deepcopy(client_train_weights))
                mlops.event("train", event_started=False, event_value="{}_{}".format(str(round_idx), str(idx)))
                t_train = time.time() - t_train_start
                total_training_time += t_train

                # Move model back to CPU after training
                client.model_trainer.model = client.model_trainer.model.cpu()

                # set back client's model trainer to use the full structure (necessary for next round)
                self._update_client_model_structure(client, None)

                # Store client weights and mask for aggregation
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
                masks_locals.append(self.client_masks[client_idx])
                counters_locals.append(client.get_sample_number())
                self.client_weights[client_idx] = copy.deepcopy(w)

                client_total_time = time.time() - client_start_time
                client_times[client_idx] = client_total_time

            # Log communication costs
            # self._log_communication_cost(round_idx, client_indexes, model_size_bits, total_params) #TODO: put comm cost

            # for training you changed the structure of the model in the model trainer, so now you need to restore it
            # self.model_trainer.model = self.model

            t_agg_start = time.time()
            mlops.event("agg", event_started=True, event_value=str(round_idx))
            w_global = self._aggregate_with_masks(w_locals, masks_locals, counters_locals, client_indexes)
            self.model_trainer.set_model_params(w_global)
            mlops.event("agg", event_started=False, event_value=str(round_idx))
            t_aggregation = time.time() - t_agg_start

            t_test_start = time.time()
            if round_idx == self.args.comm_round - 1:
                self._local_test_on_all_clients(round_idx)
                self._test_server(round_idx)
            elif round_idx % self.args.frequency_of_the_test == 0:
                self._local_test_on_all_clients(round_idx)
                self._test_server(round_idx)
            t_testing = time.time() - t_test_start

            mlops.log_round_info(self.args.comm_round, round_idx)

            # Log communication costs
            t_comm_cost_start = time.time()
            self._log_communication_cost(round_idx, client_indexes, pruned_this_round)
            t_comm_cost = time.time() - t_comm_cost_start

            # Calculate and log profiling metrics
            round_total_time = time.time() - round_start_time

            # Log timing information
            timing_info = {
                "Profile/Round/Total": round_total_time,
                "Profile/Round/ClientSampling": t_sampling,
                "Profile/Round/PruneDecision": total_prune_decision_time,
                "Profile/Round/PruneExecution": total_prune_execution_time,
                "Profile/Round/ModelUpdate": total_model_update_time,
                "Profile/Round/Training": total_training_time,
                "Profile/Round/Aggregation": t_aggregation,
                "Profile/Round/Testing": t_testing,
                "Profile/Round/CommCost": t_comm_cost,
            }

            # Calculate percentages
            if round_total_time > 0:
                timing_info["Profile/Percent/Training"] = (total_training_time / round_total_time) * 100
                timing_info["Profile/Percent/Aggregation"] = (t_aggregation / round_total_time) * 100
                timing_info["Profile/Percent/Testing"] = (t_testing / round_total_time) * 100
                timing_info["Profile/Percent/PruneDecision"] = (total_prune_decision_time / round_total_time) * 100
                timing_info["Profile/Percent/PruneExecution"] = (total_prune_execution_time / round_total_time) * 100

            # Log to wandb and mlops
            for key, value in timing_info.items():
                if self.args.enable_wandb:
                    wandb.log({key: value, "round": round_idx})
                mlops.log({key: value, "round": round_idx})

            # Log per-client times
            for client_idx_log, client_time in client_times.items():
                if self.args.enable_wandb:
                    wandb.log({f"Profile/Client_{client_idx_log}/Time": client_time, "round": round_idx})
                mlops.log({f"Profile/Client_{client_idx_log}/Time": client_time, "round": round_idx})

            # Log summary to console
            logging.info(
                f"Round {round_idx} Profiling - Total: {round_total_time:.2f}s, "
                f"Training: {total_training_time:.2f}s ({total_training_time / round_total_time * 100:.1f}%), "
                f"Aggregation: {t_aggregation:.2f}s ({t_aggregation / round_total_time * 100:.1f}%), "
                f"Testing: {t_testing:.2f}s ({t_testing / round_total_time * 100:.1f}%), "
                f"Prune Decision: {total_prune_decision_time:.2f}s, "
                f"Prune Execution: {total_prune_execution_time:.2f}s"
            )

        mlops.log_training_finished_status()
        mlops.log_aggregation_finished_status()

    def _should_client_prune(self, client_idx: int, round_idx: int) -> bool:
        """
        Determine if a client should prune in this round.

        Conditions:
        1. Counter > 0
        2. Validation accuracy < threshold

        Note: Only checks after round 0 to ensure client has trained at least once.
        """
        # Don't prune in first round - need at least one training round
        if round_idx == 0:
            return False

        if self.client_prune_counters[client_idx] <= 0:
            return False

        # Evaluate on validation set
        val_loader = self.val_data_local_dict[client_idx]
        assert val_loader is not None

        # Get client's current model for evaluation
        client_train_weights = self._get_client_training_weights(client_idx, self.model_trainer.get_model_params())

        # Temporarily update model structure for evaluation and move to GPU
        original_model = self.model_trainer.model
        self.model_trainer.model = self.client_pruned_models[client_idx].to(self.device)  # type: ignore

        # Set model and evaluate
        self.model_trainer.set_model_params(client_train_weights)

        val_metrics = self.model_trainer.test(val_loader, self.device, self.args)
        test_total = val_metrics["test_total"]
        assert test_total > 0

        val_acc = val_metrics["test_correct"] / test_total
        threshold = self.args.accuracy_threshold

        # Move model back to CPU and restore original model structure
        self.model_trainer.model = self.model_trainer.model.cpu()
        self.model_trainer.model = original_model

        if val_acc >= threshold:
            logging.info(f"Client {client_idx} validation accuracy {val_acc:.4f} >= threshold {threshold}, will prune")
            return True
        else:
            logging.info(f"Client {client_idx} validation accuracy {val_acc:.4f} < threshold {threshold}, pruning")
            return False

    def _prune_client_model(self, client_idx: int):
        """
        Prune a client's model immediately.

        The mask is generated w.r.t. the current (possibly already pruned) model,
        then combined with existing mask to get cumulative mask w.r.t. original model.
        """
        # Get client's current model (might be already pruned)
        client_train_weights = self._get_client_training_weights(client_idx, self.model_trainer.get_model_params())

        # Create temporary model for pruning (keep on CPU)
        temp_model = copy.deepcopy(self.client_pruned_models[client_idx]).cpu()  # type: ignore
        temp_model.load_state_dict(client_train_weights)

        # Read pruning parameters
        prune_way = getattr(self.args, "prune_way", "group_lasso")
        minimum_channels = getattr(self.args, "minimum_channels", 1)
        divisor = getattr(self.args, "divisor", 1)
        prune_percent = self.args.prune_percent

        # Prune the model (pruning happens on CPU)
        pruned_model, param_ratio, group_ratio, threshold, new_mask, units_pruned = prune_model(
            temp_model,
            percent=prune_percent,
            prune_way=prune_way,
            minimum_channels=minimum_channels,
            divisor=divisor,
        )

        # Combine masks if client was already pruned
        if self.client_masks[client_idx] is not None:
            # Combine previous mask with new mask to get cumulative effect
            combined_mask = combine_mask(self.client_masks[client_idx], new_mask)
            self.client_masks[client_idx] = combined_mask
            logging.info(
                f"Client {client_idx} combined masks - pruned {units_pruned} units "
                f"(param reduction: {param_ratio:.2%}, group reduction: {group_ratio:.2%})"
            )
        else:
            # First pruning - just store the mask
            self.client_masks[client_idx] = new_mask
            logging.info(
                f"Client {client_idx} first pruning - pruned {units_pruned} units "
                f"(param reduction: {param_ratio:.2%}, group reduction: {group_ratio:.2%})"
            )

        # Store the pruned model structure for this client (keep on CPU)
        self.client_pruned_models[client_idx] = pruned_model.cpu()

        # Decrement pruning counter
        self.client_prune_counters[client_idx] -= 1
        logging.info(f"Client {client_idx} prune counter: {self.client_prune_counters[client_idx]} remaining")

    def _update_client_model_structure(self, client: PruneClient, client_idx: Optional[int]):
        """
        Update the client's model_trainer to use the correct model structure.

        CRITICAL: The client's model_trainer.model must match the structure of the weights
        we're about to load. If the client has been pruned, we need to use the pruned
        model structure, otherwise there will be a size mismatch when loading weights.

        If the client has been pruned, update to use the pruned model structure.
        Otherwise, use the original full model structure.

        Note: Model is kept on CPU, will be moved to GPU when needed for operations.
        """
        if client_idx is not None and self.client_pruned_models[client_idx] is not None:
            # Client has been pruned, use pruned model structure (keep on CPU)
            client.model_trainer.model = copy.deepcopy(self.client_pruned_models[client_idx]).cpu()  # type: ignore
        else:
            # Client not pruned, use original model structure (keep on CPU)
            client.model_trainer.model = copy.deepcopy(self.model).cpu()

    def _get_client_training_weights(self, client_idx: int, w_global: Dict) -> Dict:
        """
        Get the weights for client training by applying mask to global weights.

        If client has been pruned, extract only the relevant weights from w_global
        that correspond to the pruned model structure.
        Otherwise, return the global model weights directly.

        The pruned model structure is already stored in self.client_pruned_models,
        so we just need to extract the matching weights from w_global.
        """
        if self.client_masks[client_idx] is None:
            # Client hasn't been pruned yet, use full global model
            return w_global

        # Client has been pruned - need to extract relevant weights from global model
        # Create a temporary full model and apply mask to get pruned structure (on CPU)
        temp_global_model = copy.deepcopy(self.model).cpu()
        temp_global_model.load_state_dict(w_global)

        # Apply mask to get pruned version with global weights
        prune_way = getattr(self.args, "prune_way", "group_lasso")
        minimum_channels = getattr(self.args, "minimum_channels", 1)
        divisor = getattr(self.args, "divisor", 1)
        prune_percent = self.args.prune_percent

        pruned_model_with_global_weights, _, _, _, _, _ = prune_model(
            temp_global_model,
            percent=prune_percent,
            prune_way=prune_way,
            minimum_channels=minimum_channels,
            divisor=divisor,
            with_mask=self.client_masks[client_idx],
        )

        return pruned_model_with_global_weights.state_dict()

    def _aggregate_with_masks(
        self,
        w_locals: List[tuple],  # List of (sample_num, weights) tuples
        masks_locals: List[Optional[Dict]],
        counters_locals: List[int],
        client_indexes: List[int],
    ) -> Dict:
        """
        Aggregate client models using unified_aggregate.

        Handles clients with different model structures due to different pruning.
        """
        # Extract just the weights from the tuples
        weights_only = [w for _, w in w_locals]

        # Handle case where all clients have no masks (no pruning yet)
        if all(m is None for m in masks_locals):
            # Use standard weighted averaging from parent class
            return self._aggregate(w_locals)

        # Find a reference mask from a client that has been pruned
        reference_mask = None
        for mask in masks_locals:
            if mask is not None:
                reference_mask = mask
                break

        # Build models from state dicts (keep on CPU for aggregation)
        models = []
        masks = []
        counters = []

        for w, mask, counter, client_idx in zip(weights_only, masks_locals, counters_locals, client_indexes):
            # If client has a mask, use the already stored pruned model structure
            if mask is not None:
                # Use the already pruned model structure stored for this client (keep on CPU)
                pruned_model = copy.deepcopy(self.client_pruned_models[client_idx]).cpu()  # type: ignore

                # Load client's trained weights into pruned structure
                pruned_model.load_state_dict(w)
                models.append(pruned_model)
                masks.append(mask)
            else:
                # Client has full model (keep on CPU)
                model = copy.deepcopy(self.model).cpu()
                model.load_state_dict(w)
                models.append(model)
                # Create identity mask based on reference mask structure
                masks.append(self._create_identity_mask_from_reference(reference_mask))  # type: ignore

            counters.append(counter)

        # Aggregate using unified_aggregate (happens on CPU)
        aggregated_model = aggregate_model(models, masks, counters)

        return aggregated_model.state_dict()

    def _create_identity_mask_from_reference(self, reference_mask: Dict) -> Dict:
        """
        Create an identity mask (all filters kept) based on a reference mask structure.

        This uses the original_filters from each layer in the reference mask to create
        a mask where all filters are kept. This approach is model-agnostic.

        Args:
            reference_mask: A mask dictionary from a pruned client to use as structure reference

        Returns:
            An identity mask with the same layer structure but all filters kept
        """
        identity_mask = {}

        for layer_name, mask_info in reference_mask.items():
            original_filters = mask_info["original_filters"]

            identity_mask[layer_name] = {
                "mask": torch.ones(original_filters),
                "layer_type": mask_info["layer_type"],
                "original_filters": original_filters,
                "pruned_filters": original_filters,
                "indices_kept": np.arange(original_filters),
            }

        return identity_mask

    def _local_test_on_all_clients(self, round_idx):
        """
        Override to use _get_client_training_weights for proper pruned model handling.

        The parent class uses self.client_weights[client_idx] directly, but we need
        to apply pruning masks via _get_client_training_weights().
        """
        logging.info("################local_test_on_all_clients : {}".format(round_idx))

        train_metrics = {"num_samples": [], "num_correct": [], "losses": []}
        test_metrics = {"num_samples": [], "num_correct": [], "losses": []}
        global_train_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        client = self.client_list[0]
        w_global = self.model_trainer.get_model_params()

        for client_idx in range(self.args.client_num_in_total):
            client.update_local_dataset(
                0,
                self.train_data_local_dict[client_idx],
                self.test_data_local_dict[client_idx],
                self.train_data_local_num_dict[client_idx],
            )

            # Test personal model with proper mask application
            personal_weights = self._get_client_training_weights(client_idx, w_global)

            # Temporarily set the correct model structure for this client and move to GPU
            original_model = self.model_trainer.model
            if self.client_pruned_models[client_idx] is not None:
                self.model_trainer.model = copy.deepcopy(self.client_pruned_models[client_idx]).to(self.device)  # type: ignore
            else:
                self.model_trainer.model = copy.deepcopy(self.model).to(self.device)

            self.model_trainer.set_model_params(self.client_weights[client_idx])
            train_local_metrics = client.local_test(False)

            train_metrics["num_samples"].append(copy.deepcopy(train_local_metrics["test_total"]))
            train_metrics["num_correct"].append(copy.deepcopy(train_local_metrics["test_correct"]))
            train_metrics["losses"].append(copy.deepcopy(train_local_metrics["test_loss"]))

            if self.test_data_local_dict[client_idx] is not None:
                test_local_metrics = client.local_test(True)
                test_metrics["num_samples"].append(copy.deepcopy(test_local_metrics["test_total"]))
                test_metrics["num_correct"].append(copy.deepcopy(test_local_metrics["test_correct"]))
                test_metrics["losses"].append(copy.deepcopy(test_local_metrics["test_loss"]))

                # Test global model
                self.model_trainer.set_model_params(personal_weights)
                global_train_local_metrics = client.local_test(True)

                global_train_metrics["num_samples"].append(copy.deepcopy(global_train_local_metrics["test_total"]))
                global_train_metrics["num_correct"].append(copy.deepcopy(global_train_local_metrics["test_correct"]))
                global_train_metrics["losses"].append(copy.deepcopy(global_train_local_metrics["test_loss"]))

            # Move model back to CPU and restore original structure
            self.model_trainer.model = self.model_trainer.model.cpu()
            self.model_trainer.model = original_model

        # Log personal (pruned) model metrics
        train_acc = sum(train_metrics["num_correct"]) / sum(train_metrics["num_samples"])
        train_loss = sum(train_metrics["losses"]) / sum(train_metrics["num_samples"])
        train_accs = [c / s for c, s in zip(train_metrics["num_correct"], train_metrics["num_samples"])]
        train_losses = [loss / s for loss, s in zip(train_metrics["losses"], train_metrics["num_samples"])]

        if self.args.enable_wandb:
            wandb.log({"Train/Acc": train_acc, "round": round_idx})
            wandb.log({"Train/Loss": train_loss, "round": round_idx})
            wandb.log({"Train/Acc/Std": np.std(train_accs), "round": round_idx})
            wandb.log({"Train/Loss/Std": np.std(train_losses), "round": round_idx})

        mlops.log({"Train/Acc": train_acc, "round": round_idx})
        mlops.log({"Train/Loss": train_loss, "round": round_idx})
        mlops.log({"Train/Acc/Std": np.std(train_accs), "round": round_idx})
        mlops.log({"Train/Loss/Std": np.std(train_losses), "round": round_idx})
        logging.info({"training_acc": train_acc, "training_loss": train_loss})

        if len(test_metrics["num_samples"]) > 0:
            test_acc = sum(test_metrics["num_correct"]) / sum(test_metrics["num_samples"])
            test_loss = sum(test_metrics["losses"]) / sum(test_metrics["num_samples"])
            test_accs = [c / s for c, s in zip(test_metrics["num_correct"], test_metrics["num_samples"])]
            test_losses = [loss / s for loss, s in zip(test_metrics["losses"], test_metrics["num_samples"])]

            if self.args.enable_wandb:
                wandb.log({"Test/Acc": test_acc, "round": round_idx})
                wandb.log({"Test/Loss": test_loss, "round": round_idx})
                wandb.log({"Test/Acc/Std": np.std(test_accs), "round": round_idx})
                wandb.log({"Test/Loss/Std": np.std(test_losses), "round": round_idx})

            mlops.log({"Test/Acc": test_acc, "round": round_idx})
            mlops.log({"Test/Loss": test_loss, "round": round_idx})
            mlops.log({"Test/Acc/Std": np.std(test_accs), "round": round_idx})
            mlops.log({"Test/Loss/Std": np.std(test_losses), "round": round_idx})
            logging.info({"test_acc": test_acc, "test_loss": test_loss})

            # Log global model metrics
            global_train_acc = sum(global_train_metrics["num_correct"]) / sum(global_train_metrics["num_samples"])
            global_train_loss = sum(global_train_metrics["losses"]) / sum(global_train_metrics["num_samples"])
            global_train_accs = [
                c / s for c, s in zip(global_train_metrics["num_correct"], global_train_metrics["num_samples"])
            ]
            global_train_losses = [
                loss / s for loss, s in zip(global_train_metrics["losses"], global_train_metrics["num_samples"])
            ]

            if self.args.enable_wandb:
                wandb.log({"Global/Test/Acc": global_train_acc, "round": round_idx})
                wandb.log({"Global/Test/Loss": global_train_loss, "round": round_idx})
                wandb.log({"Global/Test/Acc/Std": np.std(global_train_accs), "round": round_idx})
                wandb.log({"Global/Test/Loss/Std": np.std(global_train_losses), "round": round_idx})

            mlops.log({"Global/Test/Acc": global_train_acc, "round": round_idx})
            mlops.log({"Global/Test/Loss": global_train_loss, "round": round_idx})
            mlops.log({"Global/Test/Acc/Std": np.std(global_train_accs), "round": round_idx})
            mlops.log({"Global/Test/Loss/Std": np.std(global_train_losses), "round": round_idx})
            logging.info({"global_test_acc": global_train_acc, "global_test_loss": global_train_loss})

    def _count_model_parameters(self, model_params):
        """Count total number of parameters in the model"""
        total_params = 0
        for param_tensor in model_params.values():
            total_params += param_tensor.numel()
        return total_params

    def _log_communication_cost(self, round_idx, participating_client_indexes, pruned_this_round):  # type: ignore
        """
        Log communication costs for all clients in the current round.

        For clients that pruned this round:
        - Download cost: based on the model size before pruning
        - Upload cost: based on the model size after pruning

        For clients that did not prune this round:
        - Download and upload costs are the same (client's current local model size)

        Args:
            round_idx: Current communication round
            participating_client_indexes: List of client indices that participated in this round
            pruned_this_round: Dict mapping client_idx to {"before": size, "after": size} for clients that pruned
        """
        bits_per_param = 32

        # Communication cost tracking for all clients
        total_upload_cost = 0
        total_download_cost = 0

        for client_idx in range(self.args.client_num_in_total):
            if client_idx in participating_client_indexes:
                if client_idx in pruned_this_round:
                    # Client pruned this round
                    # Download cost: model size before pruning
                    client_download = pruned_this_round[client_idx]["before"] * bits_per_param
                    # Upload cost: model size after pruning
                    client_upload = pruned_this_round[client_idx]["after"] * bits_per_param
                else:
                    # Client did not prune this round - use current local model size
                    client_model_params = self._count_model_parameters(
                        self.client_pruned_models[client_idx].state_dict()  # type: ignore
                    )
                    client_download = client_model_params * bits_per_param
                    client_upload = client_model_params * bits_per_param

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

        mlops.log({"CommCost/Total/Upload": total_upload_cost, "round": round_idx})
        mlops.log({"CommCost/Total/Download": total_download_cost, "round": round_idx})
        mlops.log({"CommCost/Total/Combined": total_round_cost, "round": round_idx})

        logging.info(
            f"Communication costs - Round {round_idx}: Total Upload={total_upload_cost} bits, "
            f"Total Download={total_download_cost} bits, Combined={total_round_cost} bits"
        )
