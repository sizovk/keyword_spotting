import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, WeightedRandomSampler

class Collator:
    
    def __call__(self, data):
        wavs = []
        labels = []    

        for el in data:
            wavs.append(el['wav'])
            labels.append(el['label'])

        # torch.nn.utils.rnn.pad_sequence takes list(Tensors) and returns padded (with 0.0) Tensor
        wavs = pad_sequence(wavs, batch_first=True)    
        labels = torch.Tensor(labels).long()
        return wavs, labels

# We should provide to WeightedRandomSampler _weight for every sample_; by default it is 1/len(target)

def get_sampler(target):
    class_sample_count = np.array(
        [len(np.where(target == t)[0]) for t in np.unique(target)])   # for every class count it's number of occ.
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in target])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.float()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler

class SpeechCommandDataloader(DataLoader):
    def __init__(
        self,
        dataset,
        is_train,
        batch_size,
        num_workers
    ):
        if is_train:
            train_sampler = get_sampler(dataset.df['label'].values)
        else:
            train_sampler = None
        super().__init__(
            dataset, 
            collate_fn=Collator(),
            sampler=train_sampler,
            batch_size=batch_size, 
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True 
        )
