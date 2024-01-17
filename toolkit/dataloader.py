import pandas as pd
import numpy as np
import torch.utils.data as Data
import torch
from sklearn.preprocessing import StandardScaler


class DrugMANDataset:
    def __init__(self, file_csv, file_emb):
        self.file_csv = file_csv
        self.file_emb = file_emb

    def load_data(self):
        train = pd.read_csv(self.file_csv + "/bind_train_part1.csv")
        val = pd.read_csv(self.file_csv + "/bind_val_part1.csv")
        test = pd.read_csv(self.file_csv + "/bind_test_part1.csv")

        return train, val, test

    def load_embed(self):
        drug_emb = pd.read_csv(self.file_emb + "/drug_features.tsv", index_col=0, delimiter="\t")
        target_emb = pd.read_csv(self.file_emb + "/target_features.tsv", index_col=0, delimiter="\t")

        return drug_emb, target_emb

    def get_dataloader(self):
        train, val, test = self.load_data()
        drug_emb, target_emb = self.load_embed()

        # Based on the paired drug-target pair,
        # the corresponding drug embeddings and target embeddings were matched
        train_drug_emb = drug_emb.loc[train['pubchem_cid'], ]
        train_target_emb = target_emb.loc[train['gene_id'], ]

        val_drug_emb = drug_emb.loc[val["pubchem_cid"], ]
        val_target_emb = target_emb.loc[val['gene_id'], ]

        test_drug_emb = drug_emb.loc[test["pubchem_cid"], ]
        test_target_emb = target_emb.loc[test['gene_id'], ]

        # normalized by z-score and convert to tensor type
        scaler = StandardScaler()
        train_drug_emb = scaler.fit_transform(np.array(train_drug_emb))
        train_target_emb = scaler.fit_transform(np.array(train_target_emb))
        train_drug_emb = torch.FloatTensor(train_drug_emb)
        train_target_emb = torch.FloatTensor(train_target_emb)
        train_label = torch.FloatTensor(np.array(train['label']))

        val_drug_emb = scaler.fit_transform(np.array(val_drug_emb))
        val_target_emb = scaler.fit_transform(np.array(val_target_emb))
        val_drug_emb = torch.FloatTensor(val_drug_emb)
        val_target_emb = torch.FloatTensor(val_target_emb)
        val_label = torch.FloatTensor(np.array(val['label']))

        test_drug_emb = scaler.fit_transform(np.array(test_drug_emb))
        test_target_emb = scaler.fit_transform(np.array(test_target_emb))
        test_drug_emb = torch.FloatTensor(test_drug_emb)
        test_target_emb = torch.FloatTensor(test_target_emb)
        test_label = torch.FloatTensor(np.array(test['label']))

        # create dataloader
        train_dataset = Data.TensorDataset(train_drug_emb, train_target_emb, train_label)
        val_dataset = Data.TensorDataset(val_drug_emb, val_target_emb, val_label)
        test_dataset = Data.TensorDataset(test_drug_emb, test_target_emb, test_label)

        params = {'batch_size': 512, 'shuffle': True, 'num_workers': 0, 'drop_last': True}
        if train_dataset or val_dataset:
            train_loader = Data.DataLoader(train_dataset, **params)
            val_loader = Data.DataLoader(val_dataset, **params)
        if test_dataset:
            params['shuffle'] = False
            params['drop_last'] = False
            params['batch_size'] = len(test_dataset)
            test_loader = Data.DataLoader(test_dataset, **params)
            test_bcs = len(test_dataset)

        return train_loader, val_loader, test_loader, test_bcs

























