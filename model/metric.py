import numpy as np
import torch
import torch.nn.functional as F

# FA - true: 0, model: 1
# FR - true: 1, model: 0
def count_FA_FR(preds, labels):
    FA = torch.sum(preds[labels == 0])
    FR = torch.sum(labels[preds == 0])
    
    # torch.numel - returns total number of elements in tensor
    return FA.item() / torch.numel(preds), FR.item() / torch.numel(preds)

def au_fa_fr(output, labels):
    probs = F.softmax(output, dim=-1)[:,1]
    sorted_probs, _ = torch.sort(probs)
    sorted_probs = torch.cat((torch.Tensor([0]), sorted_probs, torch.Tensor([1])))
        
    FAs, FRs = [], []
    for prob in sorted_probs:
        preds = (probs >= prob) * 1
        FA, FR = count_FA_FR(preds, labels)        
        FAs.append(FA)
        FRs.append(FR)

    # ~ area under curve using trapezoidal rule
    return -np.trapz(FRs, x=FAs)
