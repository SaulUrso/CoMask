import copy
import logging
from typing import Dict, List, Optional

import numpy as np
from fedml import mlops
from fedml.ml.trainer.trainer_creator import create_model_trainer
from torch import nn

import wandb
from comask.clients.prune_client import ModelTrainerSSL, PruneClient
from comask.prune.mask_vote import vote_mask
from comask.prune.unified_prune import prune_model
from comask.prune.utils import jaccard_similarity, shuffle_mask

from .cluster_trainer import ClusterAPI

#NOTE: in the code we use mask proposal and mask-vote interchangeably


class PruneClusterAPI(ClusterAPI):
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

        # Initialize wandb table for mask proposal events
        if args.enable_wandb:
            self.proposal_table = wandb.Table(columns=["round", "cluster_id", "client_id"])
        else:
            self.proposal_table = None

        logging.info("model = {}".format(model))

        if args.ssl_coefficient > 0.0:
            self.model_trainer = ModelTrainerSSL(model, args)
            logging.info("Using ModelTrainerSSL with ssl_coefficient = {}".format(args.ssl_coefficient))
        else:
            self.model_trainer = create_model_trainer(model, args)
            logging.info("Using standard model trainer")

        self.model = model
        # Store cluster-specific models
        self.cluster_models: Dict[int, nn.Module] = {}
        # Proposal mechanism for pruning
        self.cluster_prune_counters: Dict[int, int] = {}  # How many times each cluster can prune
        self.cluster_mask_proposals: Dict[int, Dict[int, Dict]] = {}  # cluster_id -> {client_id: mask}
        # Track units pruned in first pruning for consistency
        self.cluster_first_prune_units: Dict[int, Optional[int]] = {}  # cluster_id -> units_pruned
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

        # Log cluster sizes to wandb
        if self.args.enable_wandb:
            cluster_sizes = {
                f"cluster_{group_idx}_num_clients": len(client_idxs)
                for group_idx, client_idxs in group_to_client_indexes.items()
            }
            wandb.config.update(cluster_sizes)
            logging.info(f"Logged cluster sizes to wandb config: {cluster_sizes}")

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
            # Initialize pruning counters for each cluster (starts at 3)
            prune_counter = getattr(self.args, "prune_counter", 3)
            self.cluster_prune_counters[group_idx] = prune_counter
            
            self.cluster_mask_proposals[group_idx] = {}

            self.cluster_first_prune_units[group_idx] = None

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
                w_curr_group = self.w_groups[group_idx]
                for client_idx in client_indexes:
                    client = self.client_list[idx]

                    client.model_trainer.model = self.cluster_models[group_idx]

                    client.update_local_dataset(
                        client_idx,
                        self.train_data_local_dict[client_idx],
                        self.test_data_local_dict[client_idx],
                        self.train_data_local_num_dict[client_idx],
                        local_val_data=self.val_data_local_dict[client_idx],
                    )

                    # MASK PROPOSAL PHASE: Before training, client proposes mask if using proposal mechanism
                    if self.args.pruning == "proposal" and self.cluster_prune_counters[group_idx] > 0:
                        if client_idx not in self.cluster_mask_proposals[group_idx]:
                            mask_proposal = self._generate_client_mask_proposal(group_idx, client_idx)
                            if mask_proposal is not None:
                                self.cluster_mask_proposals[group_idx][client_idx] = mask_proposal
                                logging.info(
                                    f"Client {client_idx} in cluster {group_idx} proposed a mask before training"
                                )

                                if self.proposal_table is not None:
                                    self.proposal_table.add_data(round_idx, group_idx, client_idx)
                                    logging.info(
                                        f"Logged proposal event: round={round_idx}, cluster={group_idx}, client={client_idx}"
                                    )

                    # train on new dataset
                    mlops.event("train", event_started=True, event_value="{}_{}".format(str(round_idx), str(idx)))
                    # If this cluster finished pruning (counter == 0), disable SSL regularization
                    # by passing a temporary args override with ssl_coefficient = 0.0.
                    if self.cluster_prune_counters[group_idx] <= 0:
                        args_override = copy.deepcopy(self.args)
                        args_override.ssl_coefficient = 0.0
                    else:
                        args_override = None

                    w = client.train(copy.deepcopy(w_curr_group), args_override)
                    mlops.event("train", event_started=False, event_value="{}_{}".format(str(round_idx), str(idx)))

                    w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
                    self.client_weights[client_idx] = copy.deepcopy(w)

                    idx += 1

                # update group weights
                mlops.event("agg", event_started=True, event_value=str(round_idx))
                self.w_groups[group_idx] = self._aggregate(w_locals)
                self.cluster_models[group_idx].load_state_dict(self.w_groups[group_idx])

                mlops.event("agg", event_started=False, event_value=str(round_idx))

                # Calculate cluster-specific model size for communication cost
                cluster_total_params = self._count_model_parameters(self.w_groups[group_idx])
                bits_per_param = 32
                cluster_model_size_bits = cluster_total_params * bits_per_param

                # Log communication costs
                self._log_communication_cost(
                    round_idx, group_idx, client_indexes, cluster_model_size_bits, cluster_total_params
                )

                if self.args.pruning == "random" and round_idx % 50 == 0 and round_idx != 0:
                    # don't consider this option for reproducing CoMask
                    self._prune_cluster_clients(group_idx, round_idx)
                elif self.args.pruning == "proposal":
                    self._handle_mask_voting_and_pruning(group_idx, round_idx)

            # logging
            if round_idx == self.args.comm_round - 1:
                self._local_test_on_all_clients(round_idx)
                self._test_groups(round_idx)
            elif round_idx % self.args.frequency_of_the_test == 0:
                self._local_test_on_all_clients(round_idx)
                self._test_groups(round_idx)

            mlops.log_round_info(self.args.comm_round, round_idx)

        mlops.log_training_finished_status()
        mlops.log_aggregation_finished_status()

        # Log the proposal events table to wandb at the end of training
        if self.proposal_table is not None:
            wandb.log({"mask_proposals": self.proposal_table})
            logging.info("Logged mask proposal events table to wandb")

    def _prune_cluster_clients(self, group_idx: int, round_idx: int):
        """Apply random pruning mask to all client models in the specified cluster."""
        logging.info(f"################Random pruning for cluster {group_idx} at round {round_idx}")

        cluster_seed = round_idx * (group_idx + 1)

        prune_way = getattr(self.args, "prune_way", "group_lasso")
        minimum_channels = getattr(self.args, "minimum_channels", 1)
        divisor = getattr(self.args, "divisor", 1)

        # Determine pruning amount: use same units as first pruning or percentage
        if self.cluster_first_prune_units[group_idx] is None:
            # First pruning for this cluster - use percentage
            prune_param = self.args.prune_percent
            logging.info(f"First pruning for cluster {group_idx} using percentage: {prune_param}")
        else:
            # Subsequent pruning - use same number of units as first pruning
            prune_param = self.cluster_first_prune_units[group_idx]
            logging.info(f"Subsequent pruning for cluster {group_idx} using same units: {prune_param}")

        # Generate baseline mask using cluster's current model
        temp_model = copy.deepcopy(self.cluster_models[group_idx]).cpu()
        temp_model.load_state_dict(self.w_groups[group_idx])
        _, _, group_pruning_ratio, _, baseline_mask, units_pruned = prune_model(
            temp_model,
            percent=prune_param,
            prune_way=prune_way,
            minimum_channels=minimum_channels,
            divisor=divisor,
        )

        # Track units pruned in first pruning
        if self.cluster_first_prune_units[group_idx] is None:
            self.cluster_first_prune_units[group_idx] = units_pruned
            logging.info(f"Recorded {units_pruned} units pruned for cluster {group_idx} first pruning")

        # Generate random mask with same pruning ratio
        random_mask = shuffle_mask(baseline_mask, cluster_seed)

        # Apply random mask to cluster's model
        cluster_model = copy.deepcopy(self.cluster_models[group_idx]).cpu()
        cluster_model.load_state_dict(self.w_groups[group_idx])
        pruned_cluster_model, _, _, _, _, _ = prune_model(
            cluster_model,
            prune_param,
            prune_way,
            minimum_channels,
            divisor,
            with_mask=random_mask,
        )
        self.w_groups[group_idx] = pruned_cluster_model.state_dict()

        # Apply same random mask to each client's individual model
        for client_idx in self.group_dict[group_idx]:
            client_model = copy.deepcopy(self.cluster_models[group_idx]).cpu()
            client_model.load_state_dict(self.client_weights[client_idx])
            pruned_client_model, _, _, _, _, _ = prune_model(
                client_model,
                prune_param,
                prune_way,
                minimum_channels,
                divisor,
                with_mask=random_mask,
            )
            self.client_weights[client_idx] = pruned_client_model.state_dict()

        self.cluster_models[group_idx] = pruned_cluster_model

        logging.info(
            f"Applied random pruning to cluster {group_idx} with {group_pruning_ratio:.2%} parameter reduction"
        )

    def _handle_mask_voting_and_pruning(self, group_idx: int, round_idx: int):
        """Handle mask voting and pruning for a cluster after aggregation."""
        
        # NOTE: the current implementation could be relatively more efficient
        # as we could just keep track of the sum of the masks, instead of storing all of them
        # and then summing only once the voting percentage has been reached

        if self.cluster_prune_counters[group_idx] <= 0:
            return

        total_clients_in_cluster = len(self.group_dict[group_idx])
        current_proposals = len(self.cluster_mask_proposals[group_idx])
        required_proposals = max(int(total_clients_in_cluster * self.args.consensus_percentage), 1)

        if current_proposals >= required_proposals:
            logging.info(
                f"Cluster {group_idx} reached consensus ({current_proposals}/{total_clients_in_cluster} proposals, needed {required_proposals})"
            )
            self._apply_voted_mask(group_idx, round_idx)

            # prepare for next consolidation
            self.cluster_prune_counters[group_idx] -= 1
            self.cluster_mask_proposals[group_idx] = {}
            logging.info(f"Cluster {group_idx} has {self.cluster_prune_counters[group_idx]} prune attempts remaining")

    def _generate_client_mask_proposal(self, group_idx: int, client_idx: int) -> Optional[Dict]:
        """Generate a mask proposal from a client."""

        val_loader = self.val_data_local_dict[client_idx]

        assert val_loader is not None, f"Client {client_idx} has val_loader {val_loader}"

        # Build a temporary trainer for evaluation using the cluster model structure
        self.model_trainer.model = self.cluster_models[group_idx]
        self.model_trainer.set_model_params(self.w_groups[group_idx])

        val_metrics = self.model_trainer.test(val_loader, self.device, self.args)
        test_total = val_metrics["test_total"]
        assert test_total > 0

        val_acc = val_metrics["test_correct"] / test_total
        threshold = self.args.accuracy_threshold
        if val_acc < threshold:
            logging.info(
                f"Client {client_idx} validation acc {val_acc:.4f} below threshold {threshold:.4f}; skipping proposal"
            )
            return None

        # Use the cluster's current model for mask-vote generation (copy to CPU for pruning)
        temp_model = copy.deepcopy(self.cluster_models[group_idx]).cpu()
        temp_model.load_state_dict(self.w_groups[group_idx])

        prune_way = getattr(self.args, "prune_way", "group_lasso")
        minimum_channels = getattr(self.args, "minimum_channels", 1)
        divisor = getattr(self.args, "divisor", 1)

        # Determine pruning amount: use same units as first pruning or percentage
        if self.cluster_first_prune_units[group_idx] is None:
            prune_param = self.args.prune_percent
        else:
            prune_param = self.cluster_first_prune_units[group_idx]

        # Generate mask-vote
        _, _, _, _, mask_proposal, _ = prune_model(
            temp_model,
            percent=prune_param,
            prune_way=prune_way,
            minimum_channels=minimum_channels,
            divisor=divisor,
        )

        return mask_proposal

    def _apply_voted_mask(self, group_idx: int, round_idx: int):
        """Apply the voted mask to cluster model and all clients in the cluster."""
        # Collect all mask proposals for this cluster
        mask_proposals = list(self.cluster_mask_proposals[group_idx].values())

        if self.cluster_first_prune_units[group_idx] is not None:
            consolidation_param = self.cluster_first_prune_units[group_idx]
            logging.info(f"Using {consolidation_param} units for mask voting in cluster {group_idx}")
        else:
            consolidation_param = self.args.consolidation_percentage
            logging.info(f"Using {consolidation_param} percentage for mask voting in cluster {group_idx}")

        prune_percent = self.args.prune_percent
        prune_way = getattr(self.args, "prune_way", "group_lasso")
        minimum_channels = getattr(self.args, "minimum_channels", 1)
        divisor = getattr(self.args, "divisor", 1)

        # Check if test_jaccard is enabled and we have enough proposals
        test_jaccard = getattr(self.args, "test_jaccard", False)

        # Validate consolidation_param for vote_mask
        # If using absolute units, check if it's feasible; otherwise fall back to percentage
        voting_param = consolidation_param
        if isinstance(consolidation_param, int) and len(mask_proposals) > 0:
            # Calculate total units available in the masks
            sample_mask = mask_proposals[0]
            total_units = sum(mask_info["original_filters"] for mask_info in sample_mask.values())
            num_layers = len(sample_mask)
            min_filters = 1

            # Check if absolute number is feasible
            max_removable = total_units - (min_filters * num_layers)
            if consolidation_param >= max_removable:
                # Fall back to percentage
                voting_param = self.args.consolidation_percentage
                logging.warning(
                    f"consolidation_param={consolidation_param} is too large (max_removable={max_removable}). "
                    f"Falling back to percentage={voting_param}"
                )

        if test_jaccard and len(mask_proposals) >= 10:  # Need at least 10 proposals for 20% groups
            #NOTE: ignore this branch for normal behavior
            logging.info(f"Testing Jaccard similarity for cluster {group_idx} with {len(mask_proposals)} proposals")

            # Split proposals into first 20% (old) and last 20% (new)
            num_proposals = len(mask_proposals)
            old_count = max(1, int(num_proposals * 0.2))
            new_count = max(1, int(num_proposals * 0.2))

            old_proposals = mask_proposals[:old_count]
            new_proposals = mask_proposals[-new_count:]

            logging.info(f"Old proposals: {len(old_proposals)}, New proposals: {len(new_proposals)}")

            # Compute masks for both groups
            old_voted_mask = vote_mask(old_proposals, voting_param)
            new_voted_mask = vote_mask(new_proposals, voting_param)

            # Measure Jaccard similarity between the two masks
            jaccard_results = jaccard_similarity(old_voted_mask, new_voted_mask)
            logging.info(
                f"Jaccard similarity - Overall: {jaccard_results['overall']:.4f}, Mean: {jaccard_results['mean']:.4f}"
            )

            # Log Jaccard similarity to wandb
            if self.args.enable_wandb:
                wandb.log(
                    {
                        f"Cluster_{group_idx}/Jaccard/Overall": jaccard_results["overall"],
                        f"Cluster_{group_idx}/Jaccard/Mean": jaccard_results["mean"],
                        "round": round_idx,
                    }
                )

            # Test accuracy of both masks on the test set
            # Test old mask
            cluster_model_old = copy.deepcopy(self.cluster_models[group_idx]).cpu()
            cluster_model_old.load_state_dict(self.w_groups[group_idx])
            pruned_old_model, _, _, _, _, _ = prune_model(
                cluster_model_old,
                prune_percent,
                prune_way,
                minimum_channels,
                divisor,
                with_mask=old_voted_mask,
            )

            # Set up temporary trainer for old mask
            self.model_trainer.model = pruned_old_model.to(self.device)
            self.model_trainer.set_model_params(pruned_old_model.state_dict())
            old_test_metrics = self.model_trainer.test(self.test_global, self.device, self.args)
            old_test_acc = old_test_metrics["test_correct"] / old_test_metrics["test_total"]
            old_test_loss = old_test_metrics["test_loss"] / old_test_metrics["test_total"]

            logging.info(f"Old mask test accuracy: {old_test_acc:.4f}, loss: {old_test_loss:.4f}")

            # Test new mask
            cluster_model_new = copy.deepcopy(self.cluster_models[group_idx]).cpu()
            cluster_model_new.load_state_dict(self.w_groups[group_idx])
            pruned_new_model, _, _, _, _, _ = prune_model(
                cluster_model_new,
                prune_percent,
                prune_way,
                minimum_channels,
                divisor,
                with_mask=new_voted_mask,
            )

            # Set up temporary trainer for new mask
            self.model_trainer.model = pruned_new_model.to(self.device)
            self.model_trainer.set_model_params(pruned_new_model.state_dict())
            new_test_metrics = self.model_trainer.test(self.test_global, self.device, self.args)
            new_test_acc = new_test_metrics["test_correct"] / new_test_metrics["test_total"]
            new_test_loss = new_test_metrics["test_loss"] / new_test_metrics["test_total"]

            logging.info(f"New mask test accuracy: {new_test_acc:.4f}, loss: {new_test_loss:.4f}")

            # Log both accuracies to wandb
            if self.args.enable_wandb:
                wandb.log(
                    {
                        f"Cluster_{group_idx}/MaskComparison/OldMask/Acc": old_test_acc,
                        f"Cluster_{group_idx}/MaskComparison/OldMask/Loss": old_test_loss,
                        f"Cluster_{group_idx}/MaskComparison/NewMask/Acc": new_test_acc,
                        f"Cluster_{group_idx}/MaskComparison/NewMask/Loss": new_test_loss,
                        "round": round_idx,
                    }
                )

            # Adopt the mask with higher accuracy
            if old_test_acc >= new_test_acc:
                voted_mask = old_voted_mask
                pruned_cluster_model = pruned_old_model
                chosen_mask = "old"
                logging.info(f"Adopted OLD mask with accuracy {old_test_acc:.4f} (new: {new_test_acc:.4f})")
            else:
                voted_mask = new_voted_mask
                pruned_cluster_model = pruned_new_model
                chosen_mask = "new"
                logging.info(f"Adopted NEW mask with accuracy {new_test_acc:.4f} (old: {old_test_acc:.4f})")

            # Log which mask was chosen
            if self.args.enable_wandb:
                wandb.log(
                    {
                        f"Cluster_{group_idx}/MaskComparison/ChosenMask": 0 if chosen_mask == "old" else 1,
                        "round": round_idx,
                    }
                )

        else:
            # Original behavior: use all proposals
            voted_mask = vote_mask(mask_proposals, voting_param)
            logging.info(f"Consolidated {len(mask_proposals)} mask proposals for cluster {group_idx}")

            # Apply voted mask to cluster's model
            cluster_model = copy.deepcopy(self.cluster_models[group_idx]).cpu()
            cluster_model.load_state_dict(self.w_groups[group_idx])

            pruned_cluster_model, _, _, _, _, _ = prune_model(
                cluster_model,
                prune_percent,
                prune_way,
                minimum_channels,
                divisor,
                with_mask=voted_mask,
            )

        # Apply the chosen mask to cluster model and all clients
        self.w_groups[group_idx] = pruned_cluster_model.state_dict()
        logging.info(f"Applied voted mask to cluster {group_idx} at round {round_idx}")

        for client_idx in self.group_dict[group_idx]:
            client_model = copy.deepcopy(self.cluster_models[group_idx]).cpu()
            client_model.load_state_dict(self.client_weights[client_idx])
            pruned_client_model, _, _, _, _, _ = prune_model(
                client_model,
                prune_percent,
                prune_way,
                minimum_channels,
                divisor,
                with_mask=voted_mask,
            )
            self.client_weights[client_idx] = pruned_client_model.state_dict()

        self.cluster_models[group_idx] = pruned_cluster_model

        if self.cluster_first_prune_units[group_idx] is None:
            units_pruned = sum(mask_info["pruned_filters"] for mask_info in voted_mask.values())
            self.cluster_first_prune_units[group_idx] = units_pruned
            logging.info(f"Recorded {units_pruned} units pruned for cluster {group_idx} first voted mask pruning ")

    def _evaluate_client_with_models(self, client: PruneClient, client_idx):
        """Evaluate a single client with both personal and cluster models."""
        # Get client's cluster and switch to correct model structure
        client_group_idx = self.group_indexes[client_idx]
        client.model_trainer.model = self.cluster_models[client_group_idx]
        # self.model_trainer = create_model_trainer(self.model, self.args)

        client.update_local_dataset(
            0,
            self.train_data_local_dict[client_idx],
            self.test_data_local_dict[client_idx],
            self.train_data_local_num_dict[client_idx],
            local_val_data=self.val_data_local_dict[client_idx],
        )

        # Personal model evaluation
        self.model_trainer.set_model_params(self.client_weights[client_idx])
        personal_train_metrics = client.local_test(False)

        personal_test_metrics = None
        if self.test_data_local_dict[client_idx] is not None:
            personal_test_metrics = client.local_test(True)

        # Cluster model evaluation
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

        # Collect all metrics in a single dictionary for wandb
        wandb_metrics = {"round": round_idx}
        mlops_metrics = {"round": round_idx}

        # Evaluate each group's model separately
        for group_idx, group_w in self.w_groups.items():
            # Set model to current group's weights
            self.model_trainer.model = self.cluster_models[group_idx]
            self.model_trainer.set_model_params(group_w)

            # Test on global test set
            test_metrics = self.model_trainer.test(self.test_global, self.device, self.args)
            test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
            test_loss = test_metrics["test_loss"] / test_metrics["test_total"]

            # Accumulate for overall statistics
            total_test_correct += test_metrics["test_correct"]
            total_test_samples += test_metrics["test_total"]
            total_test_loss += test_metrics["test_loss"]

            # Add individual group metrics to batch
            wandb_metrics[f"Server/Group_{group_idx}/FullTest/Acc"] = test_acc
            wandb_metrics[f"Server/Group_{group_idx}/FullTest/Loss"] = test_loss
            mlops_metrics[f"Server/Group_{group_idx}/FullTest/Acc"] = test_acc
            mlops_metrics[f"Server/Group_{group_idx}/FullTest/Loss"] = test_loss

            if sep_test_loader is not None:
                sep_test_metrics = self.model_trainer.test(sep_test_loader, self.device, self.args)
                sep_test_acc = sep_test_metrics["test_correct"] / sep_test_metrics["test_total"]
                sep_test_loss = sep_test_metrics["test_loss"] / sep_test_metrics["test_total"]

                # Accumulate for overall statistics
                total_sep_test_correct += sep_test_metrics["test_correct"]
                total_sep_test_samples += sep_test_metrics["test_total"]
                total_sep_test_loss += sep_test_metrics["test_loss"]

                # Add individual group metrics for separate test to batch
                wandb_metrics[f"Server/Group_{group_idx}/SepTest/Acc"] = sep_test_acc
                wandb_metrics[f"Server/Group_{group_idx}/SepTest/Loss"] = sep_test_loss
                mlops_metrics[f"Server/Group_{group_idx}/SepTest/Acc"] = sep_test_acc
                mlops_metrics[f"Server/Group_{group_idx}/SepTest/Loss"] = sep_test_loss

        # Calculate and add aggregate statistics across all groups
        avg_test_acc = total_test_correct / total_test_samples
        avg_test_loss = total_test_loss / total_test_samples

        wandb_metrics["Server/FullTest/Acc"] = avg_test_acc
        wandb_metrics["Server/FullTest/Loss"] = avg_test_loss
        mlops_metrics["Server/FullTest/Acc"] = avg_test_acc
        mlops_metrics["Server/FullTest/Loss"] = avg_test_loss

        if sep_test_loader is not None:
            # Calculate and add aggregate statistics for separate test
            avg_sep_test_acc = total_sep_test_correct / total_sep_test_samples
            avg_sep_test_loss = total_sep_test_loss / total_sep_test_samples

            wandb_metrics["Server/SepTest/Acc"] = avg_sep_test_acc
            wandb_metrics["Server/SepTest/Loss"] = avg_sep_test_loss
            mlops_metrics["Server/SepTest/Acc"] = avg_sep_test_acc
            mlops_metrics["Server/SepTest/Loss"] = avg_sep_test_loss

        # Single wandb.log call with all metrics
        if self.args.enable_wandb:
            wandb.log(wandb_metrics)

        # Single mlops.log call with all metrics
        mlops.log(mlops_metrics)
