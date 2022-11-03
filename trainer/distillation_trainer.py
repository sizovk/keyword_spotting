from .trainer import Trainer

class GenericDistillationTrainer(Trainer):

    def __init__(self, teacher, teacher_loss, **kwargs):
        self.teacher = teacher
        self.teacher_loss = teacher_loss
        super().__init__(**kwargs)

    def _train_epoch(self, epoch):
        self.model.train()
        self.teacher.eval()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)
            data = self.train_melspec(data)

            self.optimizer.zero_grad()
            output = self.model(data)
            teacher_output = self.teacher(data)
            loss = self.teacher_loss(output, teacher_output)
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
