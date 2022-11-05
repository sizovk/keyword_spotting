import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, train_melspec, val_melspec, criterion, metrics, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None):
        super().__init__(model, criterion, metrics, optimizer, config)
        self.config = config
        self.device = device
        self.train_melspec = train_melspec
        self.val_melspec = val_melspec
        self.data_loader = data_loader
        if self.len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler

        self.train_metrics = MetricTracker('loss', 'grad_norm')
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metrics])

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)
            data = self.train_melspec(data)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.train_metrics.update('loss', loss.item())
            self.train_metrics.update("grad_norm", self.get_grad_norm())

            if batch_idx % self.log_step == 0:
                if self.writer is not None:
                    self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx, mode="train")
                    if self.lr_scheduler is not None:
                        self.writer.add_scalar(
                            "learning_rate", self.lr_scheduler.get_last_lr()[0]
                        )
                    for metric_name in self.train_metrics.keys():
                        self.writer.add_scalar(f"{metric_name}", self.train_metrics.avg(metric_name))
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        outputs = []
        labels = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                data = self.val_melspec(data)

                output = self.model(data)
                outputs.append(output.cpu())
                labels.append(target.cpu())
                loss = self.criterion(output, target)

                self.valid_metrics.update('loss', loss.item())
        outputs = torch.cat(outputs, dim=0)
        labels = torch.cat(labels, dim=0)
        for met in self.metrics:
            self.valid_metrics.update(met.__name__, met(outputs, labels))
        if self.writer is not None:
            self.writer.set_step(epoch * self.len_epoch, mode="val")
            for metric_name in self.valid_metrics.keys():
                self.writer.add_scalar(f"{metric_name}", self.valid_metrics.avg(metric_name))

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} steps ({:.0f}%)]'
        current = batch_idx
        total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
    
    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()
