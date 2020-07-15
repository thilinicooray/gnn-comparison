import time
from datetime import timedelta
import torch
from torch import optim


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
        for data in train_loader:

            data = data.to(self.device)
            optimizer.zero_grad()
            output = model(data)

            if not isinstance(output, tuple):
                output = (output,)

            loss, acc = self.loss_fun(data.y, *output)

            loss.backward()

            loss_all += loss.item()
            acc_all += acc.item()

            if clipping is not None:  # Clip gradient before updating weights
                torch.nn.utils.clip_grad_norm_(model.parameters(), clipping)
            optimizer.step()
            optimizer.zero_grad()

        return acc_all / len(train_loader.dataset), loss_all / len(train_loader.dataset)


    def _eval(self, loader):
        model = self.model.to(self.device)
        model.eval()

        loss_all = 0
        acc_all = 0
        for data in loader:
            data = data.to(self.device)
            output = model(data)

            if not isinstance(output, tuple):
                output = (output,)

            loss, acc = self.loss_fun(data.y, *output)

            loss_all += loss.item()
            acc_all += acc.item()

        return acc_all / len(loader.dataset), loss_all / len(loader.dataset)


    def train(self, train_loader, max_epochs=100, optimizer=torch.optim.Adam, scheduler=None, clipping=None,
              validation_loader=None, test_loader=None):

        time_per_epoch = []
        max_fold_val_acc_idx = 0
        max_fold_acc = 0
        fold_test_acc = []

        for epoch in range(1, max_epochs+1):

            start = time.time()
            train_acc, train_loss = self._train(train_loader, optimizer, clipping)
            end = time.time() - start
            time_per_epoch.append(end)

            if scheduler is not None:
                scheduler.step(epoch)

            test_acc, test_loss = self._eval(test_loader)

            val_acc, val_loss = self._eval(validation_loader)


            fold_test_acc.append(test_acc)
            if val_acc > max_fold_acc:
                max_fold_acc = val_acc
                max_fold_val_acc_idx = epoch-1

                msg = f'Epoch: {epoch}, Train loss: {train_loss} Train acc: {train_acc}, VL loss: {val_loss} VL acc: {val_acc} '
                print(msg)

        test_acc_for_fold = fold_test_acc[max_fold_val_acc_idx]
        print('Test_accuracy: {:.4f} Best Epoch: {}\n'.format(test_acc_for_fold, max_fold_val_acc_idx))

        return test_acc_for_fold
