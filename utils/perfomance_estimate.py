import tempfile
from thop import profile
import torch

def get_size_in_megabytes(model):
    # https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html#look-at-model-size
    with tempfile.TemporaryFile() as f:
        torch.save(model.state_dict(), f)
        size = f.tell() / 2**20
    return size


def perfomance_estimate(model, sample_batch, baseline_macs=None, baseline_mb=None):
    macs = profile(model, (sample_batch,))[0]
    mb = get_size_in_megabytes(model)
    result = f"Model performance: {macs} MACs, {mb} MB"
    if baseline_macs:
        result += f"\nSpeed up rate: {baseline_macs / macs}"
    if baseline_mb:
        result += f"\nCompression rate: {baseline_mb / mb}"
    return result