import pandas as pd
import pathlib
import torch

KEYWORD = 'sheila'
SEED = 123
torch.manual_seed(SEED)
path2dir = pathlib.Path('speech_commands')

all_keywords = sorted([
    p.stem for p in path2dir.glob('*')
    if p.is_dir() and not p.stem.startswith('_')
])

triplets = []
for keyword in all_keywords:
    paths = (path2dir / keyword).rglob('*.wav')
    if keyword == KEYWORD:
        for path2wav in paths:
            triplets.append((path2wav.as_posix(), keyword, 1))
    else:
        for path2wav in paths:
            triplets.append((path2wav.as_posix(), keyword, 0))

df = pd.DataFrame(
    triplets,
    columns=['path', 'keyword', 'label']
)

indexes = torch.randperm(len(df))
train_indexes = indexes[:int(len(df) * 0.8)]
val_indexes = indexes[int(len(df) * 0.8):]

train_df = df.iloc[train_indexes].reset_index(drop=True)
val_df = df.iloc[val_indexes].reset_index(drop=True)
train_df.to_csv(path2dir / "train_df.csv", index=False)
val_df.to_csv(path2dir / "val_df.csv", index=False)