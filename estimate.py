import argparse
import collections
import copy
import torch
import numpy as np
import dataloader as module_data
from model.metric import au_fa_fr
import model.model as module_arch
import model.log_melspec as module_melspec
from utils import prepare_device
from utils.parse_config import ConfigParser
from tqdm import tqdm
import tempfile
from thop import profile
import torch

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def get_size_in_megabytes(model):
    # https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html#look-at-model-size
    with tempfile.TemporaryFile() as f:
        torch.save(model.state_dict(), f)
        size = f.tell() / 2**20
    return size

def speed_up_rate(model, sample_batch, baseline_macs):
    macs = profile(model, (sample_batch,))[0]
    return f"Speed up rate: {baseline_macs / macs}"

def compression_rate(model, baseline_mb):
    mb = get_size_in_megabytes(model)
    return f"Compression rate: {baseline_mb / mb}"

def main(config):
    logger = config.get_logger('test')

    device, device_ids = prepare_device(config['n_gpu'])

    # setup data_loader instances
    dataset = config.init_obj('val_dataset', module_data)

    data_loader = module_data.SpeechCommandDataloader(
        dataset,
        is_train=False,
        batch_size=128,
        num_workers=4
    )

    val_melspec = config.init_obj('melspec', module_melspec, is_train=False, device=device)

    # build model architecture
    model = config.init_obj('arch', module_arch)
    state_dict = torch.load("best.pth")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info(model)

    logger.info(speed_up_rate(
        copy.deepcopy(model), 
        val_melspec(torch.randn(config["dataloader"]["args"]["batch_size"], 2 * config["melspec"]["args"]["sample_rate"]).to(device)),  # 2 sec audio for MACs estimate
        config.get("baseline_macs"),
    ))

    model = model.half()

    logger.info(compression_rate(
        model,
        config.get("baseline_mb")
    ))

    outputs = []
    labels = []
    with torch.no_grad():
        for (data, target) in tqdm(data_loader):
            data = data.to(device)
            target = target.to(device)
            data = val_melspec(data)
            output = model(data.half())
            outputs.append(output.cpu())
            labels.append(target.cput())
    
    outputs = torch.cat(outputs, dim=0)
    labels = torch.cat(labels, dim=0)
    result = au_fa_fr(outputs, labels)
    logger.info(f"The quality: {result}")
    


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
