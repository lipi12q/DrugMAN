from pathlib import Path

import numpy as np
import pandas as pd
import joblib


class ResultSaver:
    """Class for saving results."""

    def __init__(self, save_folder: Path):
        self.save_folder = save_folder

    def save_model(self, model):
        """Saves model in a .pkl file."""
        joblib.dump(model, self.save_folder + "/model.pkl")

    def save_train_result(self, train_losses):
        """Saves training loss data in a .csv file."""
        train_losses = np.array(train_losses)
        train_losses = pd.DataFrame(train_losses, columns=["epoch", "train_loss"])
        train_losses[["epoch"]] = train_losses[["epoch"]].astype(int)
        train_losses[["train_loss"]] = round(train_losses[["train_loss"]], 5)
        train_losses.to_csv(self.save_folder + '/train_list.csv', index=False)

    def save_val_result(self, val_list):
        """save validate result in a .csv file. """
        val_result = np.array(val_list)
        data = pd.DataFrame(val_result, columns=["epoch", "auroc", "auprc", "loss"])
        data.to_csv(self.save_folder + "/val_list.csv", index=False)

    def save_test_result(self, test_list):
        """save test evaluation result in a .csv file. """
        test_result = np.array(test_list)
        data = pd.DataFrame(test_result, columns=["auroc", "auprc", "f1", "loss"])
        data.to_csv(self.save_folder + "/test_list.csv", index=False)

    def save_pred(self, test_result):
        """save test predicted result in a .csv file. """
        data = pd.DataFrame(test_result, columns=["pubchem_cid", "gene_id", "label", "pred"])
        data.to_csv(self.save_folder + "/test_pred.csv", index=False)

















