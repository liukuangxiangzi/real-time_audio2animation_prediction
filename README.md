# Real-Time Audio to Animation Prediction
## Overview

This project aims to develop a model that can predict animations in real-time from audio inputs.
It contains two methods for predicting animation parameters, one by viseme IDs and the other by Wav2Vec 2.0 feature. The pipeline includes prediction of viseme IDs / Wav2Vec 2.0 feature from audio, prediction of animation parameters from viseme IDs / Wav2Vec 2.0 feature, and generation of animations using an animation model based on the parameters.



## Environment

- Python 3.9.16
- Cuda 11.6.1
- Cudnn 8.4.0
## Set-up
The requirements (including tensorflow) can be installed using:<br>
``` pip install -r requirements.txt```<br>
Install PyTorch3d:<br>
```conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia```<br>
```conda install -c fvcore -c iopath -c conda-forge fvcore iopath```<br>
```pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"``` <br>
Download FFmpeg packages & executable files from
[ffmpeg-release-essentials](https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.7z)
and add the FFmpeg binary directory to path.<br>

## At test time:
### Wave2Vec2.0 Method
#### 1. Create and install required envs and packages according to environment and set-up sections.
#### 2. Download this repository to your local machine <br>
```git clone https://vigitlab.fe.hhi.de/liu/cvgrealtimeaudiovisemeprediction.git ```<br>
Note: Make sure FaceAnimationRenderer.py is in the root directory.
#### 3. Prepare data and model:<br>
- download the animation model from [HERE](https://vigitlab.fe.hhi.de/liu/cvgrealtimeaudiovisemeprediction.git), put the folder to the root directory.
- download the animation parameters prediction models from [HERE](https://vigitlab.fe.hhi.de/liu/cvgrealtimeaudiovisemeprediction.git), put all 4 files to w2v/model/.
- download the animation parameters of the eyes and mouth [HERE](https://vigitlab.fe.hhi.de/liu/cvgrealtimeaudiovisemeprediction.git), put all 2 files to data/. They are used for the unnormalization.
#### 4. Run command line:<br>
```python realtime_w2v_animation_render.py w2v/model/model_name.h5``` <br>

### viseme IDs Method
#### 1.  Create and install required envs and packages according to environment and set-up sections.
#### 2. Download this repository to your local machine <br>
```git clone https://vigitlab.fe.hhi.de/liu/cvgrealtimeaudiovisemeprediction.git ```<br>
#### 3. Prepare data and model:<br>
- download the viseme IDs prediction model from [HERE](https://vigitlab.fe.hhi.de/liu/cvgrealtimeaudiovisemeprediction.git), put the file to data/. They are used for the unnormalization.
- download the animation parameters prediction models from [HERE](https://vigitlab.fe.hhi.de/liu/cvgrealtimeaudiovisemeprediction.git), put all 4 files to w2v/model/.
  


## Train
1. The data for this project includes Log-Mel spectrogram features, Wav2Vec 2.0 features and animation parameters.  They can be found at [link to data source].
2. Run the data processing script: 'python realtime_data_process.py'
### Wave2Vec2.0 Method
1. Run the model training script: 'python train_w2v_to_AniPara.py' to train a CNN model.
2. Run the model training script: 'python train_w2v_to_AniPara_preAniPara.py' to train a CNN model. The input takes additional previously predicted animation parameters.
3. Run the model training script: 'python train_w2v_to_AniPara_selfattention.py' to train a CNN model with additional self-attention layers.
### viseme IDs Method
1. Run the model training script: 'python train_visemeID_to_AniPara_MLP.py' to train an MLP model. 
2. Run the model training script: 'python train_visemeID_to_AniPara_MLP.py' to train a CNN model.


