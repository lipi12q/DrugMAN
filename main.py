import numpy as np

import torch
from dataloader import DrugMANDataset
from trainer import Trainer
from saver import ResultSaver
from plotter import plot_losses, plot_roc_curve, plot_pr_curve


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    # load data and get dataloader
    dataFolder = 'data/warm_start'
    embFolder = 'data/bionic_embed'
    dataset = DrugMANDataset(dataFolder, embFolder)
    df_train, df_val, df_test = dataset.load_data()
    train_dataloader, val_dataloader, test_dataloader, test_bcs = dataset.get_dataloader()

    # start train
    drugman_trainer = Trainer(test_bcs, train_dataloader, val_dataloader, test_dataloader, device)
    train_list, val_list, test_list, test_pred, best_model = drugman_trainer.train()

    # result
    test_pred = np.array(test_pred)
    test_data = np.array(df_test)
    pred = np.concatenate((test_data, test_pred), axis=1)
    y_true = np.array(pred[:, 2]).astype(int)
    y_score = np.array(pred[:, 3])

    # save result
    save_fileFoder = "result"
    saver = ResultSaver(save_fileFoder)
    saver.save_model(best_model)
    saver.save_train_result(train_list)
    saver.save_val_result(val_list)
    saver.save_test_result(test_list)
    saver.save_pred(pred)

    # plot and save loss, roc, pr picture
    plot_losses(train_list, val_list, save_fileFoder)
    plot_roc_curve(y_true, y_score, save_fileFoder)
    plot_pr_curve(y_true, y_score, save_fileFoder)


if __name__ == '__main__':
    main()
    print("finish")
