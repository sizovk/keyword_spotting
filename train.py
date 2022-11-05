import argparse
import collections
import copy
import torch
import numpy as np
import dataloader as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import model.log_melspec as module_melspec
from trainer import Trainer
from utils import prepare_device, perfomance_estimate
from utils.parse_config import ConfigParser


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    train_dataset = config.init_obj('train_dataset', module_data)
    val_dataset = config.init_obj('val_dataset', module_data)
    data_loader = config.init_obj('dataloader', module_data, dataset=train_dataset, is_train=True)
    valid_data_loader = config.init_obj('dataloader', module_data, dataset=val_dataset, is_train=False)

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    train_melspec = config.init_obj('melspec', module_melspec, is_train=True, device=device)
    val_melspec = config.init_obj('melspec', module_melspec, is_train=False, device=device)
    
    # log compression & speed up rate
    logger.info(perfomance_estimate(
        copy.deepcopy(model),  # to avoid extra keys in state_dict
        val_melspec(torch.randn(config["dataloader"]["args"]["batch_size"], 2 * config["melspec"]["args"]["sample_rate"]).to(device)).cpu(),  # 2 sec audio for MACs estimate
        config.get("baseline_macs"),
        config.get("baseline_mb")
    ))

    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = config.init_obj('loss', module_loss)
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer) if config.get("lr_scheduler") else None

    trainer = Trainer(model, train_melspec, val_melspec, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
