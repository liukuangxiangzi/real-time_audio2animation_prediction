# Real-Time Audio to Animation Prediction
## Overview

This project aims to develop a model that can predict animations in real-time from audio inputs.
It contains two methods for predicting animation parameters, one by viseme IDs and the other by Wav2Vec 2.0 feature. The pipeline includes prediction of viseme IDs / Wav2Vec 2.0 feature from audio, prediction of animation parameters from viseme IDs / Wav2Vec 2.0 feature, and generation of animations using an animation model based on the parameters.



## Installation

Use environment.yml to install all the dependencies in Conda.

## Data
1. The data for this project includes Log-Mel spectrogram features, Wav2Vec 2.0 features and animation parameters.  They can be found at [link to data source].
2. Run the data processing script: 'python realtime_data_process.py'

## Training
### viseme IDs
1. Run the model training script: 'python train_visemeID_to_AniPara_MLP.py' to train an MLP model. 
2. Run the model training script: 'python train_visemeID_to_AniPara_MLP.py' to train a CNN model.
### Wav2Vec 2.0 feature
1. Run the model training script: 'python train_w2v_to_AniPara.py' to train a CNN model.
2. Run the model training script: 'python train_w2v_to_AniPara_preAniPara.py' to train a CNN model. The input takes additional previously predicted animation parameters.
3. Run the model training script: 'python train_w2v_to_AniPara_selfattention.py' to train a CNN model with additional self-attention layers.

##Usage
Connect a microphone to your computer.
### viseme IDs
Run the real-time audio to animation prediction script: 'python realtime_threading_v2ap_render_8frames.py'. The required trained logmel_cnn_ep50_bs32_nocenter.h5 model can be found at [link to data source].
### Wav2Vec 2.0 feature
Run the real-time audio to animation prediction script: 'python realtime_w2v_animation_render.py'.
