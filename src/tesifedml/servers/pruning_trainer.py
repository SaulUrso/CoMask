import copy
import logging
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
            self.client_pruned_models[client_idx] = None

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
            logging.info("################Communication round : {}".format(round_idx))

            client_indexes = self._client_sampling(
                round_idx, self.args.client_num_in_total, self.args.client_num_per_round
            )
            logging.info("client_indexes = " + str(client_indexes))

            w_locals = []
            masks_locals = []
            counters_locals = []

            for idx, client_idx in enumerate(client_indexes):
                client = self.client_list[idx]

                client.update_local_dataset(
                    client_idx,
                    self.train_data_local_dict[client_idx],
                    self.test_data_local_dict[client_idx],
                    self.train_data_local_num_dict[client_idx],
                    local_val_data=self.val_data_local_dict[client_idx],
                )

                # Check if client should prune before training
                should_prune = self._should_client_prune(client_idx, round_idx)

                if should_prune:
                    # Client prunes immediately before training
                    logging.info(f"Client {client_idx} pruning at round {round_idx}")
                    self._prune_client_model(client_idx)

                # Update client's model trainer to use the correct structure (pruned or full)
                self._update_client_model_structure(client, client_idx)

                # Get the client's training weights (for the correct structure)
                client_train_weights = self._get_client_training_weights(client_idx, w_global)

                # Train on the client's (possibly pruned) model
                mlops.event("train", event_started=True, event_value="{}_{}".format(str(round_idx), str(idx)))
                w = client.train(copy.deepcopy(client_train_weights))
                mlops.event("train", event_started=False, event_value="{}_{}".format(str(round_idx), str(idx)))

                # set back client's model trainer to use the full structure (necessary for next round)
                self._update_client_model_structure(client, None)

                # Store client weights and mask for aggregation
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
                masks_locals.append(self.client_masks[client_idx])
                counters_locals.append(client.get_sample_number())
                self.client_weights[client_idx] = copy.deepcopy(w)

            # Log communication costs
            # self._log_communication_cost(round_idx, client_indexes, model_size_bits, total_params) #TODO: put comm cost

            # for training you changed the structure of the model in the model trainer, so now you need to restore it
            # self.model_trainer.model = self.model

            mlops.event("agg", event_started=True, event_value=str(round_idx))
            w_global = self._aggregate_with_masks(w_locals, masks_locals, counters_locals)
            self.model_trainer.set_model_params(w_global)
            mlops.event("agg", event_started=False, event_value=str(round_idx))

            if round_idx == self.args.comm_round - 1:
                self._local_test_on_all_clients(round_idx)
                self._test_server(round_idx)
            elif round_idx % self.args.frequency_of_the_test == 0:
                self._local_test_on_all_clients(round_idx)
                self._test_server(round_idx)

            mlops.log_round_info(self.args.comm_round, round_idx)

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
        if val_loader is None:
            logging.warning(f"Client {client_idx} has no validation data, skipping pruning check")
            return False

        # Get client's current model for evaluation
        client_train_weights = self._get_client_training_weights(client_idx, self.model_trainer.get_model_params())

        # Temporarily update model structure for evaluation
        original_model = self.model_trainer.model
        if self.client_pruned_models[client_idx] is not None:
            self.model_trainer.model = self.client_pruned_models[client_idx]

        # Set model and evaluate
        self.model_trainer.set_model_params(client_train_weights)

        val_metrics = self.model_trainer.test(val_loader, self.device, self.args)
        test_total = val_metrics["test_total"]
        assert test_total > 0

        val_acc = val_metrics["test_correct"] / test_total
        threshold = self.args.accuracy_threshold

        # Restore original model structure
        self.model_trainer.model = original_model

        if val_acc < threshold: #TODO: change this, only kept this way for testing
            logging.info(f"Client {client_idx} validation accuracy {val_acc:.4f} < threshold {threshold}, will prune")
            return True
        else:
            logging.info(f"Client {client_idx} validation accuracy {val_acc:.4f} >= threshold {threshold}, no pruning")
            return False

    def _prune_client_model(self, client_idx: int):
        """
        Prune a client's model immediately.

        The mask is generated w.r.t. the current (possibly already pruned) model,
        then combined with existing mask to get cumulative mask w.r.t. original model.
        """
        # Get client's current model (might be already pruned)
        client_train_weights = self._get_client_training_weights(client_idx, self.model_trainer.get_model_params())

        # Create temporary model for pruning
        temp_model = copy.deepcopy(self.model).cpu()
        temp_model.load_state_dict(client_train_weights)

        # Read pruning parameters
        prune_way = getattr(self.args, "prune_way", "group_lasso")
        minimum_channels = getattr(self.args, "minimum_channels", 1)
        divisor = getattr(self.args, "divisor", 1)
        prune_percent = self.args.prune_percent

        # Prune the model
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

        # Store the pruned model structure for this client
        self.client_pruned_models[client_idx] = pruned_model

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
        """
        if client_idx is not None and self.client_pruned_models[client_idx] is not None:
            # Client has been pruned, use pruned model structure
            client.model_trainer.model = self.client_pruned_models[client_idx]
        else:
            # Client not pruned, use original model structure
            client.model_trainer.model = self.model

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
        # Create a temporary full model and apply mask to get pruned structure
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
    ) -> Dict:
        """
        Aggregate client models using unified_aggregate.

        Handles clients with different model structures due to different pruning.
        """
        # Extract just the weights from the tuples
        weights_only = [w for _, w in w_locals]

        # Build models from state dicts
        models = []
        masks = []
        counters = []

        for w, mask, counter in zip(weights_only, masks_locals, counters_locals):
            # Create model instance
            model = copy.deepcopy(self.model).cpu()

            # If client has a mask, we need to create the pruned model structure
            if mask is not None:
                # Apply mask to get the pruned model structure
                prune_way = getattr(self.args, "prune_way", "group_lasso")
                minimum_channels = getattr(self.args, "minimum_channels", 1)
                divisor = getattr(self.args, "divisor", 1)
                prune_percent = self.args.prune_percent

                temp_model = copy.deepcopy(self.model).cpu()
                temp_model.load_state_dict(self.model_trainer.get_model_params())

                pruned_model, _, _, _, _, _ = prune_model(
                    temp_model,
                    percent=prune_percent,
                    prune_way=prune_way,
                    minimum_channels=minimum_channels,
                    divisor=divisor,
                    with_mask=mask,
                )

                # Load client's trained weights into pruned structure
                pruned_model.load_state_dict(w)
                models.append(pruned_model)
                masks.append(mask)
            else:
                # Client has full model
                model.load_state_dict(w)
                models.append(model)
                # Create identity mask (all filters kept)
                masks.append(None)

            counters.append(counter)

        # Handle case where all clients have no masks (no pruning yet)
        if all(m is None for m in masks):
            # Use standard weighted averaging from parent class
            return self._aggregate(w_locals)

        # Replace None masks with identity masks for aggregation
        for i, mask in enumerate(masks):
            if mask is None:
                masks[i] = self._create_identity_mask(models[i])

        # Aggregate using unified_aggregate
        aggregated_model = aggregate_model(models, masks, counters)

        return aggregated_model.state_dict()

    def _create_identity_mask(self, model: nn.Module) -> Dict:
        """
        Create an identity mask (all filters kept) for a model.

        This is used when a client hasn't been pruned yet but we need a mask
        for the unified aggregation function.
        """
        identity_mask = {}

        # Determine model type and create appropriate masks
        from fedml.model.cv.resnet_cifar import ResNet

        from tesifedml.models.cnn import HARBox_CNN
        from tesifedml.models.mobilenet import MobileNet

        if isinstance(model, HARBox_CNN):
            # HARBox_CNN has conv1 and conv2
            for layer_idx in range(2):
                conv_name = f"conv{layer_idx + 1}"
                conv_seq = getattr(model, conv_name)
                conv_layer = conv_seq[0]
                num_filters = conv_layer.out_channels

                mask_key = f"conv_{layer_idx}"
                identity_mask[mask_key] = {
                    "mask": torch.ones(num_filters),
                    "layer_type": "conv",
                    "original_filters": num_filters,
                    "pruned_filters": num_filters,
                    "indices_kept": np.arange(num_filters),
                }
        elif isinstance(model, MobileNet):
            # MobileNet has multiple conv layers
            layer_idx = 0
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d) and "depthwise" not in name:
                    num_filters = module.out_channels
                    mask_key = f"conv_{layer_idx}"
                    identity_mask[mask_key] = {
                        "mask": torch.ones(num_filters),
                        "layer_type": "conv",
                        "original_filters": num_filters,
                        "pruned_filters": num_filters,
                        "indices_kept": np.arange(num_filters),
                    }
                    layer_idx += 1
        elif isinstance(model, ResNet):
            # ResNet has multiple layers
            layer_idx = 0
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d) and "downsample" not in name:
                    num_filters = module.out_channels
                    mask_key = f"conv_{layer_idx}"
                    identity_mask[mask_key] = {
                        "mask": torch.ones(num_filters),
                        "layer_type": "conv",
                        "original_filters": num_filters,
                        "pruned_filters": num_filters,
                        "indices_kept": np.arange(num_filters),
                    }
                    layer_idx += 1
        else:
            raise ValueError(f"Unsupported model type for identity mask creation: {type(model)}")

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

            # Temporarily set the correct model structure for this client
            original_model = self.model_trainer.model
            if self.client_pruned_models[client_idx] is not None:
                self.model_trainer.model = self.client_pruned_models[client_idx]

            self.model_trainer.set_model_params(personal_weights)

            train_local_metrics = client.local_test(False)
            train_metrics["num_samples"].append(copy.deepcopy(train_local_metrics["test_total"]))
            train_metrics["num_correct"].append(copy.deepcopy(train_local_metrics["test_correct"]))
            train_metrics["losses"].append(copy.deepcopy(train_local_metrics["test_loss"]))

            if self.test_data_local_dict[client_idx] is not None:
                test_local_metrics = client.local_test(True)
                test_metrics["num_samples"].append(copy.deepcopy(test_local_metrics["test_total"]))
                test_metrics["num_correct"].append(copy.deepcopy(test_local_metrics["test_correct"]))
                test_metrics["losses"].append(copy.deepcopy(test_local_metrics["test_loss"]))

            # Restore original model structure and test global model
            self.model_trainer.model = original_model
            self.model_trainer.set_model_params(w_global)

            global_train_local_metrics = client.local_test(False)
            global_train_metrics["num_samples"].append(copy.deepcopy(global_train_local_metrics["test_total"]))
            global_train_metrics["num_correct"].append(copy.deepcopy(global_train_local_metrics["test_correct"]))
            global_train_metrics["losses"].append(copy.deepcopy(global_train_local_metrics["test_loss"]))

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
            wandb.log({"Global/Train/Acc": global_train_acc, "round": round_idx})
            wandb.log({"Global/Train/Loss": global_train_loss, "round": round_idx})
            wandb.log({"Global/Train/Acc/Std": np.std(global_train_accs), "round": round_idx})
            wandb.log({"Global/Train/Loss/Std": np.std(global_train_losses), "round": round_idx})

        mlops.log({"Global/Train/Acc": global_train_acc, "round": round_idx})
        mlops.log({"Global/Train/Loss": global_train_loss, "round": round_idx})
        mlops.log({"Global/Train/Acc/Std": np.std(global_train_accs), "round": round_idx})
        mlops.log({"Global/Train/Loss/Std": np.std(global_train_losses), "round": round_idx})
        logging.info({"global_training_acc": global_train_acc, "global_training_loss": global_train_loss})
