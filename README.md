# keyword_spotting


### Setup
```bash
pip install -r requirements.txt
wget http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz -O speech_commands_v0.01.tar.gz
mkdir speech_commands && tar -C speech_commands -xvzf speech_commands_v0.01.tar.gz 1> log
python utils/setup_dataset.py
python train.py -c configs/baseline.json 
```