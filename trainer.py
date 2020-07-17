import time
from datetime import timedelta
import torch
from torch import optim
import numpy as np


def format_time(avg_time):
    avg_time = timedelta(seconds=avg_time)
    total_seconds = int(avg_time.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d}.{str(avg_time.microseconds)[:3]}"


class Trainer:

    def __init__(self, model, loss_function, device='cuda'):
        self.model = model
        self.loss_fun = loss_function()
        self.device = torch.device(device)

    def _train(self, train_loader, optimizer, clipping=None):
        model = self.model.to(self.device)

        model.train()

        loss_all = 0
        acc_all = 0
        dig_loss_all = 0
        for data, neg_data in train_loader:

            data = data.to(self.device)
            neg_data = data[torch.randperm(data.size()[0])]

            optimizer.zero_grad()
            output, dig_loss = model(data, neg_data)

            if not isinstance(output, tuple):
                output = (output,)

            loss, acc = self.loss_fun(data.y, *output)

            final_loss = loss + dig_loss

            final_loss.backward()

            try:
                num_graphs = data.num_graphs
            except TypeError:
                num_graphs = data.adj.size(0)

            loss_all += loss.item() * num_graphs
            acc_all += acc.item() * num_graphs
            dig_loss_all += dig_loss.item() * num_graphs

            if clipping is not None:  # Clip gradient before updating weights
                torch.nn.utils.clip_grad_norm_(model.parameters(), clipping)
            optimizer.step()
            optimizer.zero_grad()

        return acc_all / len(train_loader.dataset), loss_all / len(train_loader.dataset), dig_loss_all/ len(train_loader.dataset)


    def _eval(self, loader):
        model = self.model.to(self.device)
        model.eval()

        loss_all = 0
        acc_all = 0
        for data, neg_data in loader:
            data = data.to(self.device)
            neg_data = data[torch.randperm(data.size()[0])]
            output, _ = model(data, neg_data)

            if not isinstance(output, tuple):
                output = (output,)

            loss, acc = self.loss_fun(data.y, *output)

            try:
                num_graphs = data.num_graphs
            except TypeError:
                num_graphs = data.adj.size(0)

            loss_all += loss.item() * num_graphs
            acc_all += acc.item() * num_graphs

        return acc_all / len(loader.dataset), loss_all / len(loader.dataset)


    def train(self, train_loader, fold_no, run_no, max_epochs=100, optimizer=torch.optim.Adam, scheduler=None, clipping=None,
              validation_loader=None, test_loader=None):

        time_per_epoch = []

        test_scores = []

        # Mitigate bad random initializations
        max_fold_val_acc_idx = 0
        max_fold_acc = 0
        fold_test_acc = []

        for epoch in range(1, max_epochs+1):

            start = time.time()
            train_acc, train_loss, dig_loss = self._train(train_loader, optimizer, clipping)
            end = time.time() - start
            time_per_epoch.append(end)

            if scheduler is not None:
                scheduler.step(epoch)

            test_acc, test_loss = self._eval(test_loader)

            val_acc, val_loss = self._eval(validation_loader)


            fold_test_acc.append(test_acc)
            if max_fold_acc < val_acc: #model selection based on val acc
                max_fold_acc = val_acc
                max_fold_val_acc_idx = epoch-1

            msg = f'Fold: {fold_no}, run {run_no}, Epoch: {epoch}, Train loss: {train_loss} DIG loss: {dig_loss}  Train acc: {train_acc}, Val loss: {val_loss} Val acc: {val_acc} Test acc: {test_acc}'
            print(msg)

        test_acc_for_fold = fold_test_acc[max_fold_val_acc_idx]
        print('Test accuracy: {:.4f} using Best Val Epoch: {}\n'.format( test_acc_for_fold, max_fold_val_acc_idx+1))

        #test_scores.append(test_acc_for_fold)

        #test_score = sum(test_scores) / 3

        return test_acc_for_fold, max_fold_val_acc_idx+1
