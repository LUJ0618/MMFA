# MMFA : Masked Multi-layer Feature Aggregation for Speaker Verification using WavLM
***
## Installation
1. Install pyenv and poetry
2. Clone this repository
3. Setup virtual environment and install python requirements.
```sh
pyenv install 3.8.10
pyenv virtualenv 3.8.10 env_name
pyenv local env_name
poetry env use python
poetry install
```
4. Download WavLM model to local pretrained model path (Base+) [official](https://github.com/microsoft/unilm/tree/master/wavlm)
5. Download Voxceleb1 and Voxceleb2 datasets
6. Update your local dataset path and save directory path for trained model and evaluation results. 
7. Train
```sh
poetry python run train.py
```
8. Evaluation
```sh
poetry run python train.py --eval
