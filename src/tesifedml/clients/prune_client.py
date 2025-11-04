import logging
from typing import Union

import torch
from fedml.ml.trainer.my_model_trainer_classification import ModelTrainerCLS
from fedml.ml.trainer.my_model_trainer_nwp import ModelTrainerNWP
from fedml.ml.trainer.my_model_trainer_tag_prediction import ModelTrainerTAGPred
from fedml.simulation.sp.fedavg.client import Client
from torch import nn

from tesifedml.prune.keywords import KEY_FILTER
from tesifedml.prune.utils import ssl_loss


class ModelTrainerSSL(ModelTrainerCLS):
    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)  # pylint: disable=E1102
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

        # Get SSL loss coefficients from args
        ssl_coefficient = args.ssl_coefficient
        ssl_loss_type = getattr(args, "ssl_loss_type", KEY_FILTER)

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []

            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                labels = labels.long()

                # Classification loss
                cls_loss = criterion(log_probs, labels)  # pylint: disable=E1102

                # SSL loss
                total_loss = cls_loss
                if ssl_coefficient > 0.0:
                    ssl_reg_loss = ssl_loss(
                        model, loss_type=ssl_loss_type, lambda_n=ssl_coefficient, lambda_c=ssl_coefficient
                    )
                    total_loss = cls_loss + ssl_reg_loss

                total_loss.backward()
                optimizer.step()

                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # logging.info(
                #     "Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                #         epoch,
                #         (batch_idx + 1) * args.batch_size,
                #         len(train_data) * args.batch_size,
                #         100.0 * (batch_idx + 1) / len(train_data),
                #         loss.item(),
                #     )
                # )

                batch_loss.append(total_loss.item())
            if len(batch_loss) == 0:
                epoch_loss.append(0.0)
            else:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info(
                "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(self.id, epoch, sum(epoch_loss) / len(epoch_loss))
            )

    def train_iterations(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)  # pylint: disable=E1102
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

        # Get SSL loss coefficients from args
        ssl_coefficient = args.ssl_coefficient
        ssl_loss_type = getattr(args, "ssl_loss_type", KEY_FILTER)

        epoch_loss = []

        current_steps = 0
        current_epoch = 0
        while current_steps < args.local_iterations:
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                labels = labels.long()

                # Classification loss
                cls_loss = criterion(log_probs, labels)  # pylint: disable=E1102

                # SSL loss
                total_loss = cls_loss
                if ssl_coefficient > 0.0:
                    ssl_reg_loss = ssl_loss(
                        model, loss_type=ssl_loss_type, lambda_n=ssl_coefficient, lambda_c=ssl_coefficient
                    )
                    total_loss = cls_loss + ssl_reg_loss

                total_loss.backward()

                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                # logging.info(
                #     "Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                #         epoch,
                #         (batch_idx + 1) * args.batch_size,
                #         len(train_data) * args.batch_size,
                #         100.0 * (batch_idx + 1) / len(train_data),
                #         loss.item(),
                #     )
                # )
                batch_loss.append(total_loss.item())
                current_steps += 1
                if current_steps == args.local_iterations:
                    break
            current_epoch += 1
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info(
                "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
                    self.id, current_epoch, sum(epoch_loss) / len(epoch_loss)
                )
            )


class PruneClient(Client):
    def __init__(
        self,
        client_idx,
        local_training_data,
        local_val_data,
        local_test_data,
        local_sample_number,
        args,
        device,
        model_trainer: Union[ModelTrainerTAGPred, ModelTrainerNWP, ModelTrainerCLS],
    ):
        super().__init__(
            client_idx, local_training_data, local_test_data, local_sample_number, args, device, model_trainer
        )
        self.model_trainer = model_trainer  # also done in super, but pylance won't show me types otherwhise
        self.local_val_data = local_val_data

    def update_local_dataset(
        self, client_idx, local_training_data, local_test_data, local_sample_number, local_val_data=None
    ):
        self.local_val_data = local_val_data
        return super().update_local_dataset(client_idx, local_training_data, local_test_data, local_sample_number)

    def train(self, w_global, args_override=None):
        """Train the client's model starting from w_global.

        Args:
            w_global: state_dict of global weights to load
            args_override: optional argparse/namespace-like object to use instead of self.args
                           (useful to temporarily disable SSL without mutating global args)
        """
        # download parameters
        self.model_trainer.set_model_params(w_global)

        # if condition_of_pruning -> then get the mask and return also that, else return none as mask
        if self.args.pruning != "random":
            pass

        # decide which args object to pass to the trainer (allow temporary override)
        trainer_args = args_override if args_override is not None else self.args

        # finetuning
        self.model_trainer.train(self.local_training_data, self.device, trainer_args)
        weights = self.model_trainer.get_model_params()
        return weights
