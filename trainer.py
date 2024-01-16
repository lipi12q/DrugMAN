import torch
import torch.nn as nn
import torch.optim as optim

import time
import copy
from math import cos, pi
import numpy as np

from sklearn.metrics import roc_curve, precision_recall_curve, auc
from model import DrugMAN


class Trainer:
    def __init__(self, test_bcs, train_generator, val_generator, test_generator, device):

        self.epochs = 4
        self.batch_size = 512
        self.test_bcs = test_bcs

        self.device = device

        self.train_generator = train_generator
        self.val_generator = val_generator
        self.test_generator = test_generator


    def BCE_loss(self, input, target):
        BCE_loss = nn.BCELoss()
        m = nn.Sigmoid()
        loss = BCE_loss(torch.squeeze(m(input)), target)
        return loss
        
    def adjust_lr(self, optimizer, current_epoch, max_epoch, lr_min, lr_max, warmup=True):
        warmup_epoch = 20 if warmup else 0
        if current_epoch < warmup_epoch:
            lr = lr_max * current_epoch / warmup_epoch
        else:
            lr = lr_min + (lr_max - lr_min) * (1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train(self):
        self.model = DrugMAN().to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=3e-5, weight_decay=0.02)  # 这里调整参数，来训练模型
        best_val_auroc = 0
        train_list = []
        val_list = []
        test_list = []
        for epoch in range(self.epochs):
            time_start = time.time()
            self.model.train()
            current_epoch = epoch+1
            loss_sum = 0
            float2str = lambda x: '%0.6f' % x
            self.adjust_lr(optimizer, epoch, self.epochs, lr_min=0, lr_max=3e-5, warmup=True)
            for step, (v_d, v_p, batch_label) in enumerate(self.train_generator):
                v_d, v_p, batch_label = v_d.to(self.device), v_p.to(self.device), batch_label.to(self.device)
                v_dp = torch.cat((v_d, v_p), axis=1)
                v_dp = v_dp.view(self.batch_size, 2, 512)
                v_dp = v_dp.to(self.device)
                optimizer.zero_grad()
                y_pred = self.model(v_dp)
                batch_loss = self.BCE_loss(y_pred, batch_label)
                loss_sum += batch_loss.item()
                batch_loss.backward()
                optimizer.step()

            epoch_loss = loss_sum/len(self.train_generator)
            train_lst = [current_epoch] + list(map(float2str, [epoch_loss]))
            train_list.append(train_lst)
            print("Epoch " + str(current_epoch))
            print("train_loss: %.8f" % (epoch_loss))
            val_auroc, val_auprc, val_loss = self.test(dataloader="val")
            val_lst = [current_epoch] + list(map(float2str, [val_auroc, val_auprc, val_loss]))
            times = time.time() - time_start
            print("val_AUROC: %.4f, val_AUPRC: %.4f, val_loss: %.4f, times: %.4f" % (val_auroc, val_auprc, val_loss, times))
            val_list.append(val_lst)
            if val_auroc > best_val_auroc:
                self.best_model = copy.deepcopy(self.model)
                best_val_auroc = val_auroc
                self.best_epoch = current_epoch
        # scheduler.step()
        print("Best_epoch  " + str(self.best_epoch))
        test_auroc, test_auprc, test_f1_score, test_loss, test_pred = self.test(dataloader="test")
        print("test_auroc: %.4f,test_auprc: %.4f,test_loss: %.4f" % (test_auroc, test_auprc, test_loss))
        test_lst = list(map(float2str, [test_auroc, test_auprc, test_f1_score, test_loss]))
        test_list.append(test_lst)
        save_model = self.best_model

        return train_list, val_list, test_list, test_pred, save_model

    def test(self, dataloader="test"):
        if dataloader == "val":
            val_losses = 0
            y_pred, y_label = [], []
            with torch.no_grad():
                self.model.eval()
                for step, (v_d, v_p, batch_label) in enumerate(self.val_generator):
                    v_d, v_p, batch_label = v_d.to(self.device), v_p.to(self.device), batch_label.to(self.device)
                    v_dp = torch.cat((v_d, v_p), axis=1)
                    v_dp = v_dp.view(self.batch_size, 2, 512)
                    v_dp = v_dp.to(self.device)
                    val_pred = self.model(v_dp)
                    val_loss = self.BCE_loss(val_pred, batch_label)
                    val_losses += val_loss.item()
                    m = nn.Sigmoid()
                    val_pred = m(val_pred)
                    y_pred = y_pred + val_pred.tolist()
                    y_label = y_label + batch_label.tolist()
            val_roc_auc, val_pr_auc, val_f1 = self.evaluate(y_label, y_pred)
            val_ave_loss = val_losses / len(self.val_generator)

        elif dataloader == "test":
            for step, (v_d, v_p, batch_label) in enumerate(self.test_generator):
                v_d, v_p, batch_label = v_d.to(self.device), v_p.to(self.device), batch_label.to(self.device)
                v_dp = torch.cat((v_d, v_p), axis=1)
                v_dp = v_dp.view(self.test_bcs, 2, 512)
                v_dp = v_dp.to(self.device)
                test_pred = self.best_model(v_dp)
                test_loss = self.BCE_loss(test_pred, batch_label)
                test_loss = test_loss.item()
                m = nn.Sigmoid()
                test_pred = m(test_pred)
                test_pred = test_pred.tolist()
                test_label = batch_label.tolist()
                test_roc_auc, test_pr_auc, test_f1 = self.evaluate(test_label, test_pred)
        if dataloader == "test":
            return test_roc_auc, test_pr_auc, test_f1, test_loss, test_pred
        else:
            return val_roc_auc, val_pr_auc, val_ave_loss

    def evaluate(self, y_true, y_pred):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        pr_auc = auc(recall, precision)
        f1 = (2 * precision * recall)/(precision+recall)
        best_f1 = np.max(f1[np.isfinite(f1)])
        return roc_auc, pr_auc, best_f1