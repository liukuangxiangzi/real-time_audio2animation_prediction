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
Install graphviz:<br>
```conda install graphviz```<br>
Download FFmpeg packages & executable files from
[ffmpeg-release-essentials](https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.7z)
and add the FFmpeg binary directory to path.<br>

## At test time:

#### 1. Create and install required envs and packages according to environment and set-up sections.
#### 2. Download this repository to your local machine <br>
```git clone https://vigitlab.fe.hhi.de/liu/cvgrealtimeaudiovisemeprediction.git ```<br>
Note: Make sure FaceAnimationRenderer.py is in the root directory.
#### 3. Download the animation model folder from `\\hhi.de\abteilung\VIT\VIT-CVG-Ablage\personal\Wolfgang\CVGRealtimeAudio2AnimationPrediction\FaceModel`, put the folder to the root directory.
#### 4. Prepare data and trained model:<br>
#### Wav2Vec2.0 Method
- download the animation parameters prediction models from `\\hhi.de\abteilung\VIT\VIT-CVG-Ablage\personal\Wolfgang\CVGRealtimeAudio2AnimationPrediction\w2v\model`, put all 4 files to `w2v/model/`.
- download the animation parameters of the eyes `\\hhi.de\abteilung\VIT\VIT-CVG-Ablage\personal\Wolfgang\CVGRealtimeAudio2AnimationPrediction\data\eye_31_32_1195.npy` and mouth `\\hhi.de\abteilung\VIT\VIT-CVG-Ablage\personal\Wolfgang\CVGRealtimeAudio2AnimationPrediction\data\mouth_31_32_1195.npy`, put all 2 files to `data/`. They are used for the unnormalization.
#### Viseme IDs Method
- download the viseme IDs prediction model from `\\hhi.de\abteilung\VIT\VIT-CVG-Ablage\personal\Wolfgang\CVGRealtimeAudio2AnimationPrediction\visemeID\model\logmel_viseme_cnn_ep50_bs32_nocenter.h5` and the animation parameters prediction model from `\\hhi.de\abteilung\VIT\VIT-CVG-Ablage\personal\Wolfgang\CVGRealtimeAudio2AnimationPrediction\visemeID\model\viseme_params_cnn_ep400_31_32_1188.h5`, put the file to `visemeID/model/`.
#### 5. Run command line:<br>
```python realtime_w2v_animation_render.py w2v/model/model_name.h5``` <br>
or <br>
```python realtime_visemeID_animation_render.py visemeID/model/audio-visemeID-model_name.h5 visemeID/model/visemeID-param-model_name.h5``` <br>


## Train
### Data used to train
- Download the phoneme from `\\hhi.de\abteilung\VIT\VIT-CVG-Ablage\personal\Wolfgang\CVGRealtimeAudio2AnimationPrediction\data\phones`, Wav2Vec 2.0 features from `\\hhi.de\abteilung\VIT\VIT-CVG-Ablage\personal\Wolfgang\CVGRealtimeAudio2AnimationPrediction\data\wav2vec`.
- Download animation parameters from `\\hhi.de\abteilung\VIT\VIT-CVG-Ablage\personal\Wolfgang\CVGRealtimeAudio2AnimationPrediction\data\animation-params`.
- Put 3 downloaded folders to `data/`.
### Data processing
#### Viseme ID
Get viseme ID from phoneme data <br>
```python realtime_data_process.py make_viseme_ID_dataset -d data/phones/```<br>
Generate CNN viseme ID input <br>
```python realtime_data_process.py generate_cnn_input -d data/phones/viseme_ID.npy -t 8```<br>
Divide train and test datase for viseme ID (MLP) <br>
```python realtime_data_process.py make_train_test_dataset -d data/phones/viseme_ID.npy```<br>
Divide train and test dataset for viseme ID (CNN) <br>
```python realtime_data_process.py make_train_test_dataset -d data/phones/viseme_ID_8timesteps.npy```<br>
#### Wav2vec features
Load and concatenate all Wav2vec features numpy files into a single numpy file <br>
```python realtime_data_process.py append_np_data -d data/wav2vec```<br>
Calculate the softmax of Wav2vec data <br>
```python realtime_data_process.py softmax_data -d data/wav2vec/wav2vec.npy```<br>
Divide train and test dataset for Wav2vec feature <br>
```python realtime_data_process.py make_train_test_dataset -d data/wav2vec/wav2vec_softmax.npy```<br>
#### Animation params
Load and concatenate all eyes/mouth animation params numpy files into a single numpy file <br>
```python realtime_data_process.py append_np_data -d data/animation-params/eyes```<br>
```python realtime_data_process.py append_np_data -d data/animation-params/mouth```<br>
Normalize animation params for eyes and mouth <br>
```python realtime_data_process.py data_Normalization -d data/animation-params/eyes/eyes.npy```<br>
```python realtime_data_process.py data_Normalization -d data/animation-params/mouth/mouth.npy```<br>
Generate CNN eyes/mouth animation params lable <br>
```python realtime_data_process.py generate_cnn_lable -d data/animation-params/eyes/scaled_eye.npy -t 8```
```python realtime_data_process.py generate_cnn_lable -d data/animation-params/mouth/scaled_mouth.npy -t 8```
Divide train and test dataset for normalized eyes/mouth animation params <br>
```python realtime_data_process.py make_train_test_dataset -d data/animation-params/eyes/scaled_eyes.npy```<br>
```python realtime_data_process.py make_train_test_dataset -d data/animation-params/mouth/scaled_mouth.npy```<br>
Divide train and test dataset for CNN eyes/mouth animation params <br>
```python realtime_data_process.py make_train_test_dataset -d data/animation-params/eyes/scaled_eyes_8timesteps.npy```<br>
```python realtime_data_process.py make_train_test_dataset -d data/animation-params/mouth/scaled_mouth_8timesteps.npy```<br>

### Training
#### viseme IDs Method
Train MLP model with default arguments<br>
```python visemeID/train_visemeID_params_mlp.py --epochs 300 --batch-size 32```<br>
Train CNN model with default arguments<br>
```python visemeID/train_visemeID_params_cnn.py --timestep 8 --epochs 300 --batch-size 32```<br>
Train CNN model with your own arguments<br>
```python visemeID/train_visemeID_params_cnn.py --train-input-data-dir path/to/train_visemeID --test-input-data-dir path/to/test_viseme_ID --train-eyes-data-dir path/to/train_eyes_param --test-eyes-data-dir path/to/test_eyes_param --train-mouth-data-dir path/to/train_mouth_param --test-mouth-data-dir path/to/test_mouth_param --timestep 8 --epochs 300 --batch-size 32 --resume-training False --save-model-dir path/to/model/ --load-model-dir path/to/model/```<br>

#### Wav2Vec2.0 Method
Train CNN model with default argument<br>
```python w2v/train_w2v_params_cnn.py --epochs 300 --batch-size 16```<br>
Train CNN model, in which the input takes additional previously predicted animation params, with default argument<br>
```python w2v/train_w2v_params_cnn_previousParams.py --epochs 300 --batch-size 32```<br>
Train CNN model with self-attention layers<br>
```python w2v/train_w2v_params_cnn_selfattention.py --epochs 300 --batch-size 16```<br>

