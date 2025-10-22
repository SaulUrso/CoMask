from typing import Union

from fedml.ml.trainer.my_model_trainer_classification import ModelTrainerCLS
from fedml.ml.trainer.my_model_trainer_nwp import ModelTrainerNWP
from fedml.ml.trainer.my_model_trainer_tag_prediction import ModelTrainerTAGPred
from fedml.simulation.sp.fedavg.client import Client


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

    def train(self, w_global):
        # download parameters
        self.model_trainer.set_model_params(w_global)

        # if condition_of_pruning -> then get the mask and return also that, else return none as mask
        if self.args.pruning != "random":
            pass

        # finetuning
        self.model_trainer.train(self.local_training_data, self.device, self.args)
        weights = self.model_trainer.get_model_params()
        return weights
