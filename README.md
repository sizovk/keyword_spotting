# Keyword spotting

This repository contains code for building an efficient KWS model. The task was to optimize in time and memory the base model, which is a CRNN architecture with an attention layer. For this purpose, Dark Knowledge Distillation was used, described in https://arxiv.org/pdf/1503.02531.pdf

## [Experiments report](https://wandb.ai/k_sizov/keyword_spotting/reports/KWS-report--VmlldzoyOTI1OTAw?accessToken=rcz73x9n6mak2vno0ksh9ucdm1yf0kju5obi1pgnhzjjb7uxedkgxvnsal9e9di2)


## Reproduce results
### Setup data
```bash
pip install -r requirements.txt
wget http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz -O speech_commands_v0.01.tar.gz
mkdir speech_commands && tar -C speech_commands -xvzf speech_commands_v0.01.tar.gz 1> log
python utils/setup_dataset.py
```

### Train baseline
```bash
python train.py -c configs/baseline.json 
```

### Train distillation
Before running scripts, replace teacher checkpoint path in configs with the one that was received at the previous stage.
```bash
python distillation_train.py -c configs/night.json
python distillation_train.py -c configs/morning.json
python distillation_train.py -c configs/day.json 
python distillation_train.py -c configs/evening.json
```

### Streaming
To run pretrained model `stream_model.pth` (in TorchScript), use
```bash
python stream.py
```