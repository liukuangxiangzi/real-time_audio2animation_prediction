import pyaudio
import wave
import librosa
import numpy as np
import time
import collections
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import threading
import os
import shutil
import torch
import cv2
from FaceAnimationRenderer import FaceAnimationRenderer
import argparse

#tf.config.set_visible_devices([], 'GPU')

def melSpectra(y, sr, wsize):
    cnst = 1 + (int(sr * wsize) / 2)
    y_stft_abs = np.abs(librosa.stft(y,
                                     win_length=int(sr * wsize),
                                     hop_length=int(sr * wsize),
                                     center=False,
                                     n_fft=int(sr * wsize))) / cnst
    melspec = np.log(1e-16 + librosa.feature.melspectrogram(sr=sr, S=y_stft_abs ** 2, n_mels=64))
    return melspec

parser = argparse.ArgumentParser(description='Process input models.')
parser.add_argument("model1_path", help="Path to the audio-visemeID-model file")
parser.add_argument("model2_path", help="Path to the visemeID-param-model file")
args = parser.parse_args()

class RecordThread2(threading.Thread):
    def __init__(self, strm):
        threading.Thread.__init__(self)
        self.running = False
        self.stream = strm
        self.chunk = 2000
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 50000
        self.chunks = collections.deque()
        self.lock = threading.Lock()
        self.count = 0

    def run(self):
        self.running = True

        print("* recording")
        start_time = time.time()

        while self.running:
            data = stream.read(self.chunk, exception_on_overflow=False)
            with self.lock:
                self.chunks.append(data)
                self.count += 1

        print('end recording')
        end = time.time()
        print("Total Recording Time: {} secs".format(end-start_time))

    def get_next_data(self):
        data = None
        with self.lock:
            if len(self.chunks) > 0:
                data = self.chunks.popleft()
        return data

    def get_chunk_count(self):
        return self.count

    def stop(self):
        self.running = False


class PredictThread2(threading.Thread):
    def __init__(self, recorder: RecordThread2):
        threading.Thread.__init__(self)
        self.phoneme_dict = {
            'ER0': 9, 'AA1': 9, 'AE1': 9, 'AE2': 9, 'AH0': 9, 'AH1': 9, 'AO1': 11, 'AW1': 9, 'AY1': 9, 'B': 0, 'SH': 6, 'CH': 6, 'D': 5, 'DH': 3,
            'EH1': 9, 'EY1': 9, 'IY2': 9,'F': 2, 'G': 7, 'HH': 8, 'IH0': 9, 'IH1': 9, 'IY0': 9,'IY1': 9, 'NG':9, 'JH': 6, 'K': 7, 'L': 4,
            'M': 0, 'N': 5, 'OW0': 11, 'OW1': 11, 'OW2': 11, 'P': 0, 'R': 5, 'S': 6, 'T': 5, 'TH': 3, 'UW1': 10, 'V': 2, 'W': 1,
            'Y': 7, 'Z': 5, 'sil': 12, 'sp': 12
        }
        self.running = False
        self.recorder = recorder
        self.model_audio2viseme = load_model(args.model1_path)
        self.model_viseme2animation = load_model(args.model2_path)
        self.wsize=0.04
        self.hsize =0.04
        self.ctxWin = 5
        self.count = 1
        self.processed_frames = []
        self.detected_visemes = []

        self.CHUNK = 2000
        self.FORMAT = pyaudio.paInt16 #pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 50000
        #self.RECORD_SECONDS = 3

        self.melFrames_3 = collections.deque(maxlen=3)
        self.time_step_features = collections.deque(maxlen=6)

        self.viseme_8 = collections.deque(maxlen=8)
        self.mouth_animation_param_list = []
        self.eye_animation_param_list = []


    def run(self):
        self.running = True
        print("* predicting")
        start_time = time.time()

        zeroVecD = np.zeros((1, 64))
        zeroVecDD = np.zeros((2, 64))

        init_que_zeros = np.zeros((1, 128))
        for i in range(6):
            self.time_step_features.append(init_que_zeros)
        count = 1
        avg_processing_time = 0
        processing_count = 0
        finished_processing = False
        while not finished_processing:
            data = self.recorder.get_next_data()

            if data is not None:
                ss = time.time()
                self.processed_frames.append(data)

                data_sample = np.frombuffer(data, dtype=np.int16) #np.float32
                data_sample = np.ndarray.astype(data_sample, float) / 32768  #2000
                melFrames = np.transpose(melSpectra(data_sample, self.RATE, self.wsize))[:1,:] #(2, 64)
                self.melFrames_3.append(melFrames)
                if len(self.melFrames_3) == 3:
                    if count == 1:
                        count += 1
                        fir_melDelta = np.insert(np.squeeze(np.diff(self.melFrames_3, n=1, axis=0)), 0, zeroVecD, axis=0) #(2, 2, 64)
                        fir_melDDelta = np.insert(np.squeeze(np.diff(self.melFrames_3, n=2, axis=0), axis=0), 0, zeroVecDD, axis=0)
                        fir_features = np.concatenate((fir_melDelta, fir_melDDelta), axis=1) #76,128
                        fir_features = fir_features[1:, :]
                        self.melFrames_3.popleft()

                        self.time_step_features.appendleft(np.expand_dims(fir_features[0,:], axis=0))
                        self.time_step_features.appendleft(np.expand_dims(fir_features[1,:], axis=0))
                    else:
                        melDelta = np.insert(np.squeeze(np.diff(self.melFrames_3, n=1, axis=0)), 0, zeroVecD, axis=0) #(2, 2, 64)
                        melDDelta = np.insert(np.squeeze(np.diff(self.melFrames_3, n=2, axis=0), axis=0), 0, zeroVecDD, axis=0)
                        features = np.concatenate((melDelta, melDDelta), axis=1) #76,128
                        features = features[2:,:]
                        self.melFrames_3.popleft()
                        self.time_step_features.appendleft(features)

                if len(self.time_step_features) == 6:
                    model_in = np.array(self.time_step_features)
                    model_in = np.squeeze(model_in, axis=1)
                    model_in = np.expand_dims(model_in, axis=0)
                    y = self.model_audio2viseme.predict(model_in)
                    self.detected_visemes += [list(self.phoneme_dict.keys())[list(self.phoneme_dict.values()).index(np.argmax(y))]]
                    #viseme to animation parameter
                    self.viseme_8 .append(to_categorical(np.argmax(y),num_classes=13))
                    if len(self.viseme_8) == 8:
                        animation_model_in = np.array(self.viseme_8)
                        #animation_model_in = animation_model_in.reshape(13*8)
                        animation_model_in = np.expand_dims(animation_model_in, axis=0)
                        mouth_animation_param, eye_animation_param= self.model_viseme2animation.predict(animation_model_in)
                        self.mouth_animation_param_list += [mouth_animation_param]
                        self.eye_animation_param_list += [eye_animation_param]
                        self.viseme_8.popleft()
                    self.time_step_features.pop()
                tt = time.time()
                avg_processing_time += (tt-ss)
                processing_count += 1
            else:
                time.sleep(0)  # nothing to do, let other threads work
                if not self.running:
                    finished_processing = True

        print('end predicting')
        end = time.time()
        print("Total Prediction Time: {} sec".format(end-start_time))
        print("Average Prediction Time: {} msec".format((avg_processing_time / processing_count) * 1000))

    def stop(self):
        self.running = False
        np.savetxt('mouth_animation_param.txt', np.squeeze(np.array(self.mouth_animation_param_list)))
        np.savetxt('eye_animation_param.txt', np.squeeze(np.array(self.eye_animation_param_list)))

    def get_processed_frames(self):
        return self.processed_frames

    def get_detected_visemes(self):
        return self.detected_visemes

    def get_mouth_animation_params(self):
        return self.mouth_animation_param_list

    def get_eye_animation_params(self):
        return self.eye_animation_param_list


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

CHUNK = 2000 #1024 2000
FORMAT = pyaudio.paInt16 #pyaudio.paFloat32
CHANNELS = 1
RATE = 50000 #44100 50000
RECORD_SECONDS = 3
p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

stream_recording = RecordThread2(stream)#, 'recorded_result.wav')
stream_prediciting = PredictThread2(stream_recording)
stream_recording.start()
stream_prediciting.start()

a = input("Press ENTER to exit...")

# tell threads to stop
stream_recording.stop()
stream_prediciting.stop()

# wait for recording thread to finish
stream_recording.join()

# wait for until all audio samples have been processed
stream_prediciting.join()

print(" ".join(stream_prediciting.get_detected_visemes()) + "\n")
print("Number of Recorded Audio Chunks: {}".format(stream_recording.get_chunk_count()))
print("Number of Processed Audio Chunks: {}".format(len(stream_prediciting.get_processed_frames())))
print("Number of Visemes: {}".format(len(stream_prediciting.get_detected_visemes())))

# wait for threads to finish
# stream_prediciting.join()
# stream_recording.stoprecord()
# stream_recording.join()
# time.sleep(3)
# stream_recording.stoprecord()

waveFile = wave.open("recorded.wav", 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(p.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(stream_prediciting.get_processed_frames()))
waveFile.close()

'''
########## Render Animated Face
'''

render_device = 'cuda:0'
mouth_code_dim_vae = 256
eyes_code_dim_vae = 256
mouth_model_epoch = 100
eyes_model_epoch = 75
mouth_model_dir = './FaceModel/sarah/2022-03-15_12-19-35_neural-mouth-model256-snglconv-no_gan/'
eyes_model_dir = './FaceModel/sarah/2022-03-16_12-34-21_neural-eyes-model256-snglconv-no_gan/'
mouth_gen = torch.jit.load(mouth_model_dir + 'trace/epoch' + str(mouth_model_epoch) + '/decoder_full.pth').to(device=render_device).eval()
gen_eyes = torch.jit.load(eyes_model_dir + 'trace/epoch' + str(eyes_model_epoch) + '/decoder_full.pth').to(device=render_device).eval()

face_renderer = FaceAnimationRenderer('./FaceModel/sarah',
                                      render_device,
                                      mouth_code_dim_vae, eyes_code_dim_vae, 6,
                                      w=512, h=512, fov=20,
                                      mouth_gen=mouth_gen, eye_gen=gen_eyes,
                                      post_translation=torch.FloatTensor([[0, 0, -1.5]])
                                      )

if os.path.exists("animated_frames") and os.path.isdir("animated_frames"):
    shutil.rmtree("animated_frames")
    time.sleep(1)
os.makedirs("animated_frames", exist_ok=True)

for i in range(len(stream_prediciting.get_mouth_animation_params())):
    mouth_params = torch.from_numpy(stream_prediciting.get_mouth_animation_params()[i]).to(dtype=torch.float32)
    eye_params = torch.from_numpy(stream_prediciting.get_eye_animation_params()[i]).to(dtype=torch.float32)
    pose_params = torch.zeros(1, 6).to(dtype=torch.float32)
    full_params = torch.cat([mouth_params, eye_params, pose_params], dim=1).to(dtype=torch.float32).to(device=render_device)
    face_renderer.update_expression(full_params)
    img = (face_renderer.render().squeeze(0).permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    img_bgr = cv2.cvtColor(src=img, code=cv2.COLOR_RGB2BGR)
    cv2.imwrite("animated_frames/frame{:04}.png".format(i), img_bgr)
    cv2.namedWindow('test')
    cv2.imshow('test', img_bgr)
    cv2.waitKey(1)
os.system("ffmpeg -y -i animated_frames/frame%04d.png -i recorded.wav animation.mp4")
