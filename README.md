# MMFA : Masked Multi-layer Feature Aggregation for Speaker Verification using WavLM
***
### Uijong Lee and Seok-Pil Lee
- [paper link](https://www.mdpi.com/2079-9292/14/19/3857)
## Abstract
peaker verification (SV) is a core technology for security and personalized services, and its importance has been growing with the spread of wearables such as smartwatches, earbuds, and AR/VR headsets, where privacy-preserving on-device operation under limited compute and power budgets is required. Recently, self-supervised learning (SSL) models such as WavLM and wav2vec 2.0 have been widely adopted as front ends that provide multi-layer speech representations without labeled data. Lower layers contain fine-grained acoustic information, whereas higher layers capture phonetic and contextual features. However, conventional SV systems typically use only the final layer or a single-step temporal attention over a simple weighted sum of layers, implicitly assuming that frame importance is shared across layers and thus failing to fully exploit the hierarchical diversity of SSL embeddings. We argue that frame relevance is layer dependent, as the frames most critical for speaker identity differ across layers. To address this, we propose Masked Multi-layer Feature Aggregation (MMFA), which first applies independent frame-wise attention within each layer, then performs learnable layer-wise weighting to suppress irrelevant frames such as silence and noise while effectively combining complementary information across layers. On VoxCeleb1, MMFA achieves consistent improvements over strong baselines in both EER and minDCF, and attention-map analysis confirms distinct selection patterns across layers, validating MMFA as a robust SV approach even in short-utterance and noisy conditions.

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
```
***
## Reference
This code is based on
- [Vox-Trainer](https://github.com/clovaai/voxceleb_trainer)
- [SLT22_MultiHead-Factorized-Attentive-Pooling](https://github.com/JunyiPeng00/SLT22_MultiHead-Factorized-Attentive-Pooling?tab=readme-ov-file)
