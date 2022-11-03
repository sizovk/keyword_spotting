import pandas as pd
from torch.utils.data import Dataset
import torchaudio

from .augmentations import AugsCreation


class SpeechCommandDataset(Dataset):

    def __init__(
        self,
        is_train=True,
        df_path=None
    ):        
        self.transform = AugsCreation() if is_train else None
        self.df = pd.read_csv(df_path)
        
    
    def __getitem__(self, index: int):
        instance = self.df.iloc[index]

        path2wav = instance['path']
        wav, sr = torchaudio.load(path2wav)
        wav = wav.sum(dim=0)
        
        if self.transform:
            wav = self.transform(wav)

        return {
            'wav': wav,
            'keywords': instance['keyword'],
            'label': instance['label']
        }

    def __len__(self):
        return len(self.df)