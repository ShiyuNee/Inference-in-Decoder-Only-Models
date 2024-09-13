import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
from collect import read_json
from sklearn.metrics import roc_auc_score
import os
import json

def arrange_data(batch, mode, device):
    batch_size = batch[0].shape[0]
    if mode == 'cnn':
        inputs = batch[0].to(device).unsqueeze(1) # for cnn, 加一维
    elif mode == 'mlp':
        inputs = batch[0].to(device).view(batch_size, -1) # (batch_size, dim)
    elif mode == 'lstm':
        inputs = batch[0].to(device)
    elif mode == 'transformer':
        inputs = batch[0].to(device)
    elif mode == 'multi-mlp':
        inputs = batch[0].to(device)
    else:
        raise ValueError("Specify the wrong mode")
    return inputs

class Generator:
    def __init__(self, train_data, test_data, ood_data, batch_size, model):
        self.train_data = train_data
        self.test_data = test_data
        self.odd_data = ood_data
        self.train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
        self.test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
        self.ood_loader = DataLoader(ood_data, shuffle=False, batch_size=batch_size)
        self.model = model

    def finetune(self, epochs=100, mode='cnn', lr_rate=5e-5):
        device = torch.device('cuda')
        self.model = self.model.to(device)
        optimizer = Adam(self.model.parameters(), lr=lr_rate)
        criterion = torch.nn.CrossEntropyLoss()
        self.model.train()
        dev_acc_list = []
        test_acc_list = []
        dev_pred_list = []
        test_pred_list = []
        dev_auroc_list = []
        ood_auroc_list = []
        avg_dev_loss, dev_pred, dev_auroc = self.evaluate('test', mode, device)
        avg_ood_loss, test_pred, ood_auroc = self.evaluate('ood', mode, device)
        print('Begin Average test acc ' + str(avg_dev_loss) + ', Auroc test ' + str(dev_auroc))
        print('Begin Average ood acc ' + str(avg_ood_loss) + ', Auroc ood ' + str(ood_auroc))
        # avg_dev_loss = self.evaluate('cuda')
        # print(f'Begin, Average acc ' + str(avg_dev_loss))
        for epoch in range(0, epochs):
            print('Epoch ' + str(epoch) + ' start')
            total_train_loss = 0.0
            acc = 0.0
            self.model.train()

            for step, batch in enumerate(self.train_loader):
                # inputs = get_contrastive_layer(batch[0]).to(device).view(batch_size, -1)
                inputs = arrange_data(batch, mode, device)
                target = batch[1].to(device)
                outputs = self.model(inputs)
                acc += sum(torch.argmax(outputs, axis=1) == torch.argmax(target, axis=1))
                loss = criterion(outputs, target)
                total_train_loss += loss
                loss.backward()
                optimizer.step()
                self.model.zero_grad()
            
            avg_train_loss = total_train_loss / len(self.train_loader)
            acc = acc / len(self.train_data)
            print('Epoch ' + str(epoch) + ', Average Train acc ' + str(acc.item()))
            # eval
            avg_dev_loss, dev_pred, dev_auroc = self.evaluate('test', mode, device)
            avg_ood_loss, test_pred, ood_auroc = self.evaluate('ood', mode, device)
            dev_acc_list.append(avg_dev_loss)
            test_acc_list.append(avg_ood_loss)
            dev_pred_list.append(dev_pred)
            dev_auroc_list.append(dev_auroc)
            test_pred_list.append(test_pred)
            ood_auroc_list.append(ood_auroc)
            print('Epoch ' + str(epoch) + ', Average dev acc ' + str(avg_dev_loss.item()) + ', Auroc dev ' + str(dev_auroc))
            print('Epoch ' + str(epoch) + ', Average test acc ' + str(avg_ood_loss.item()) + ', Auroc test ' + str(ood_auroc))
            print(f'pred right: {sum(dev_pred)/len(dev_pred)}')
            print(f'pred right: {sum(test_pred)/len(test_pred)}')
        dev_acc_list = torch.tensor(dev_acc_list)
        test_acc_list = torch.tensor(test_acc_list)
        best_dev_score, dev_idx = torch.max(dev_acc_list, dim=0)
        best_test_score, test_idx = torch.max(test_acc_list, dim=0)
        test_score = test_acc_list[dev_idx]
        test_pred = test_pred_list[dev_idx]
        best_test_pred = test_pred_list[test_idx]
        # print(f'test score: {test_score}, idx: {dev_idx}')
        # print(f'test max score: {best_test_score}, idx: {test_idx}')
        return test_score, dev_idx, best_test_score, test_idx, test_pred, best_test_pred

    def evaluate(self, test_mode, mode, device):
        self.model.eval()
        acc = 0
        data_loader = self.test_loader if test_mode == "test" else self.ood_loader
        data_set = self.test_data if test_mode == "test" else self.odd_data
        pred_list = []
        prob_list = []
        label_list = []
        for batch in data_loader:
            inputs = arrange_data(batch, mode, device)
            target = batch[1].to(device)
            # batch_size = data.shape[0]
            # data = get_contrastive_layer(data).to(mode).view(batch_size, -1)
            target = target.to("cuda")
            target = torch.argmax(target, axis=1)
            pred = torch.argmax(self.model(inputs), axis=1)
            prob = self.model(inputs)[:, 1]
            acc += sum(pred == target)
            pred_list += pred.tolist()
            prob_list += prob.tolist()
            label_list += target.tolist()
        return acc / len(data_set), pred_list, roc_auc_score(label_list, prob_list)
