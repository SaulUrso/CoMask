import copy
import logging
from typing import Dict, List, Optional

import numpy as np
from fedml import mlops
from fedml.ml.trainer.trainer_creator import create_model_trainer
from torch import nn

from comask.aggregators.unified_aggregate import aggregate_model
from comask.clients.prune_client import ModelTrainerSSL, PruneClient
from comask.prune.unified_prune import prune_model
from comask.prune.utils import combine_mask

from .cluster_trainer import ClusterAPI


class PruneClusterImmediateAPI(ClusterAPI):
    """
    Cluster-based federated learning with immediate client-side pruning.
    
    In this approach, clients prune their own models immediately (not through proposal/voting).
    Each client maintains a prune counter (default=3) and prunes when:
    1. Counter > 0
    2. Validation accuracy < threshold
    
    Since clients have different pruned models, aggregation uses unified_aggregate.py.
    Client masks are combined across pruning operations to maintain reference to original model.
    
    Key differences from PruneClusterAPI:
    - Pruning happens immediately per client (not cluster-wide)
    - Each client can have different pruning schedules based on validation accuracy
    - Uses unified_aggregate for aggregating heterogeneous models
    - Masks are cumulative (combined using combine_mask) to track pruning w.r.t. original model
    
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
            self.model_trainer = create_model_trainer(model, args)
            logging.info("Using standard model trainer")

        self.model = model
        
        # Store cluster-specific models (unpruned versions)
        self.cluster_models: Dict[int, nn.Module] = {}
        
        # Track client-specific information
        self.client_prune_counters: Dict[int, int] = {}  # client_id -> remaining prunes
        self.client_masks: Dict[int, Optional[Dict]] = {}  # client_id -> mask (relative to original)
        
        logging.info("self.model_trainer = {}".format(self.model_trainer))

        self._setup_clients(
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            self.model_trainer,
            val_data_local_dict=val_data_local_dict,
        )

    def _setup_clients(
        self,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        model_trainer,
        val_data_local_dict=None,
    ):
        assert val_data_local_dict is not None
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

        # Initialize client pruning counters and masks
        prune_counter_default = getattr(self.args, "client_prune_counter", 3)
        for client_idx in range(self.args.client_num_in_total):
            self.client_prune_counters[client_idx] = prune_counter_default
            self.client_masks[client_idx] = None

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
        logging.info("self.model_trainer = {}".format(self.model_trainer))
        w_global = self.model_trainer.get_model_params()
        self.w_groups = {group_idx: copy.deepcopy(w_global) for group_idx in self.group_dict.keys()}

        # Initialize cluster-specific models
        for group_idx in self.group_dict.keys():
            self.cluster_models[group_idx] = copy.deepcopy(self.model)

        self.client_weights = [copy.deepcopy(w_global)] * (self.args.client_num_in_total)

        mlops.log_training_status(mlops.ClientConstants.MSG_MLOPS_CLIENT_STATUS_TRAINING)
        mlops.log_aggregation_status(mlops.ServerConstants.MSG_MLOPS_SERVER_STATUS_RUNNING)
        mlops.log_round_info(self.args.comm_round, -1)
        
        for round_idx in range(self.args.comm_round):
            logging.info("################Communication round : {}".format(round_idx))

            group_to_client_indexes = self._client_sampling(
                round_idx, self.args.client_num_in_total, self.args.client_num_per_round
            )
            logging.info("client_indexes = " + str(group_to_client_indexes))

            idx = 0

            for group_idx, client_indexes in group_to_client_indexes.items():
                w_locals = []
                masks_locals = []
                counters_locals = []
                
                for client_idx in client_indexes:
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
                        self._prune_client_model(client_idx, group_idx)
                    
                    # Get the client's training model (pruned version of cluster model)
                    client_train_weights = self._get_client_training_weights(client_idx, group_idx)
                    
                    # Train on the client's (possibly pruned) model
                    mlops.event("train", event_started=True, event_value="{}_{}".format(str(round_idx), str(idx)))
                    w = client.train(copy.deepcopy(client_train_weights))
                    mlops.event("train", event_started=False, event_value="{}_{}".format(str(round_idx), str(idx)))

                    # Store client weights and mask for aggregation
                    w_locals.append(copy.deepcopy(w))
                    masks_locals.append(self.client_masks[client_idx])
                    counters_locals.append(client.get_sample_number())
                    self.client_weights[client_idx] = copy.deepcopy(w)

                    idx += 1

                # Aggregate using unified_aggregate (handles different model structures)
                mlops.event("agg", event_started=True, event_value=str(round_idx))
                self.w_groups[group_idx] = self._aggregate_with_masks(
                    w_locals, masks_locals, counters_locals, group_idx
                )
                mlops.event("agg", event_started=False, event_value=str(round_idx))

            # Testing
            if round_idx == self.args.comm_round - 1:
                self._local_test_on_all_clients(round_idx)
                self._test_groups(round_idx)
            elif round_idx % self.args.frequency_of_the_test == 0:
                self._local_test_on_all_clients(round_idx)
                self._test_groups(round_idx)

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
        group_idx = self.group_indexes[client_idx]
        client_train_weights = self._get_client_training_weights(client_idx, group_idx)
        
        # Set model and evaluate
        self.model_trainer.model = self.cluster_models[group_idx]
        self.model_trainer.set_model_params(client_train_weights)
        
        val_metrics = self.model_trainer.test(val_loader, self.device, self.args)
        test_total = val_metrics["test_total"]
        assert test_total > 0
        
        val_acc = val_metrics["test_correct"] / test_total
        threshold = self.args.accuracy_threshold
        
        if val_acc < threshold:
            logging.info(
                f"Client {client_idx} validation accuracy {val_acc:.4f} < threshold {threshold}, will prune"
            )
            return True
        else:
            logging.info(
                f"Client {client_idx} validation accuracy {val_acc:.4f} >= threshold {threshold}, no pruning"
            )
            return False

    def _prune_client_model(self, client_idx: int, group_idx: int):
        """
        Prune a client's model immediately.
        
        The mask is generated w.r.t. the current (possibly already pruned) model,
        then combined with existing mask to get cumulative mask w.r.t. original model.
        """
        # Get client's current model (might be already pruned)
        client_train_weights = self._get_client_training_weights(client_idx, group_idx)
        
        # Create temporary model for pruning
        temp_model = copy.deepcopy(self.cluster_models[group_idx]).cpu()
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
        
        # Decrement pruning counter
        self.client_prune_counters[client_idx] -= 1
        logging.info(f"Client {client_idx} prune counter: {self.client_prune_counters[client_idx]} remaining")

    def _get_client_training_weights(self, client_idx: int, group_idx: int) -> Dict:
        """
        Get the weights for client training.
        
        If client has been pruned, apply the client's mask to the cluster model.
        Otherwise, return the cluster model weights directly.
        """
        if self.client_masks[client_idx] is None:
            # Client hasn't been pruned yet, use full cluster model
            return self.w_groups[group_idx]
        
        # Client has been pruned - apply mask to cluster model to get pruned version
        cluster_model = copy.deepcopy(self.cluster_models[group_idx]).cpu()
        cluster_model.load_state_dict(self.w_groups[group_idx])
        
        # Apply client's mask to get pruned model
        prune_way = getattr(self.args, "prune_way", "group_lasso")
        minimum_channels = getattr(self.args, "minimum_channels", 1)
        divisor = getattr(self.args, "divisor", 1)
        prune_percent = self.args.prune_percent
        
        pruned_client_model, _, _, _, _, _ = prune_model(
            cluster_model,
            percent=prune_percent,
            prune_way=prune_way,
            minimum_channels=minimum_channels,
            divisor=divisor,
            with_mask=self.client_masks[client_idx],
        )
        
        return pruned_client_model.state_dict()

    def _aggregate_with_masks(
        self, 
        w_locals: List[Dict], 
        masks_locals: List[Optional[Dict]], 
        counters_locals: List[int],
        group_idx: int
    ) -> Dict:
        """
        Aggregate client models using unified_aggregate.
        
        Handles clients with different model structures due to different pruning.
        """
        # Build models from state dicts
        models = []
        masks = []
        counters = []
        
        for w, mask, counter in zip(w_locals, masks_locals, counters_locals):
            # Create model instance
            model = copy.deepcopy(self.cluster_models[group_idx]).cpu()
            
            # If client has a mask, we need to create the pruned model structure
            if mask is not None:
                # Apply mask to get the pruned model structure
                prune_way = getattr(self.args, "prune_way", "group_lasso")
                minimum_channels = getattr(self.args, "minimum_channels", 1)
                divisor = getattr(self.args, "divisor", 1)
                prune_percent = self.args.prune_percent
                
                temp_model = copy.deepcopy(self.cluster_models[group_idx]).cpu()
                temp_model.load_state_dict(self.w_groups[group_idx])
                
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
            # Use standard weighted averaging
            return self._aggregate_standard(w_locals, counters)
        
        # Replace None masks with identity masks for aggregation
        for i, mask in enumerate(masks):
            if mask is None:
                masks[i] = self._create_identity_mask(models[i])
        
        # Aggregate using unified_aggregate
        aggregated_model = aggregate_model(models, masks, counters)
        
        return aggregated_model.state_dict()

    def _aggregate_standard(self, w_locals: List[Dict], counters: List[int]) -> Dict:
        """Standard weighted averaging when no pruning has occurred."""
        training_num = sum(counters)
        
        averaged_params = copy.deepcopy(w_locals[0])
        for k in averaged_params.keys():
            averaged_params[k] = averaged_params[k] * (counters[0] / training_num)
            for i in range(1, len(w_locals)):
                w = counters[i] / training_num
                averaged_params[k] += w_locals[i][k] * w
        
        return averaged_params

    def _create_identity_mask(self, model: nn.Module) -> Dict:
        """
        Create an identity mask (all filters kept) for a model.
        
        This is used when a client hasn't been pruned yet but we need a mask
        for the unified aggregation function.
        """
        import torch
        
        identity_mask = {}
        
        # Determine model type and create appropriate masks
        from comask.models.cnn import HARBox_CNN
        from comask.models.mobilenet import MobileNet
        from fedml.model.cv.resnet_cifar import ResNet
        
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

    def _evaluate_client_with_models(self, client: PruneClient, client_idx):
        """Evaluate a single client with both personal and cluster models."""
        # Get client's cluster
        client_group_idx = self.group_indexes[client_idx]
        
        client.update_local_dataset(
            0,
            self.train_data_local_dict[client_idx],
            self.test_data_local_dict[client_idx],
            self.train_data_local_num_dict[client_idx],
            local_val_data=self.val_data_local_dict[client_idx],
        )

        # Update client's model trainer to use cluster-specific model structure
        client.model_trainer.model = self.cluster_models[client_group_idx]

        # Personal model evaluation (client's own pruned model)
        client_train_weights = self._get_client_training_weights(client_idx, client_group_idx)
        self.model_trainer.set_model_params(client_train_weights)
        personal_train_metrics = client.local_test(False)

        personal_test_metrics = None
        if self.test_data_local_dict[client_idx] is not None:
            personal_test_metrics = client.local_test(True)

        # Cluster model evaluation (full cluster model)
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
