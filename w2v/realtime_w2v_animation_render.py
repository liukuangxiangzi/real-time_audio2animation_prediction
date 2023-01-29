from pydub import AudioSegment
import pyaudio
import wave
import librosa
import numpy as np
import time
import collections
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import tensorflow as tf
from tensorflow.keras.models import load_model
import threading
from keras.utils import to_categorical
# from scipy.special import softmax
import os
import shutil
import torch
import cv2
from FaceAnimationRenderer import FaceAnimationRenderer
from sklearn.preprocessing import MinMaxScaler

#tf.config.set_visible_devices([], 'GPU')
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def denormalize_mouth(y_mouth_norm):
    y_mouth = y_mouth_norm*9.932146 - 5.5546255
    return y_mouth
def denormalize_eyes(y_eyes_norm):
    y_eyes = y_eyes_norm*12.494246 - 5.6491327
    return y_eyes


scaler_mouth = MinMaxScaler()
scaler_eye = MinMaxScaler()
scaled_y_mouth = scaler_mouth.fit_transform(np.load('data/mouth_31_32_1195.npy'))
scaled_y_eye = scaler_eye.fit_transform(np.load('data/eye_31_32_1195.npy'))

class RecordThread2(threading.Thread):
  def __init__(self):
    threading.Thread.__init__(self)
    CHUNK = 640
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    p = pyaudio.PyAudio()

    self.stream = p.open(format=FORMAT,
                         channels=CHANNELS,
                         rate=RATE,
                         input=True,
                         frames_per_buffer=CHUNK)

    self.running = False
    self.chunk = CHUNK
    self.format = FORMAT
    self.channels = CHANNELS
    self.rate = RATE
    self.chunks = collections.deque()
    self.lock = threading.Lock()
    self.count = 0

  def run(self):
    self.running = True

    print("* recording")
    start_time = time.time()

    while self.running:
      data = self.stream.read(self.chunk, exception_on_overflow=False)
      with self.lock:
        self.chunks.append(data)
        self.count += 1

    print('end recording')
    end = time.time()
    print("Total Recording Time: {} secs".format(end - start_time))

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


class ProcessingThread2(threading.Thread):
  def __init__(self, recorder: RecordThread2, model, tokenizer):
    threading.Thread.__init__(self)

    self.running = False
    self.recorder = recorder
    self.model_w2v2animation = load_model('model/model_w2v2AniPara/softmax_noappend/selfattention_unsoftmax.h5')
    self.count = 1
    self.model = model
    self.tokenizer = tokenizer
    self.processed_frame_count = 0
    self.processed_frames = []
    self.audio_buffer = collections.deque(maxlen=1000)
    self.lock = threading.Lock()
    #self.viseme_8 = collections.deque(maxlen=8)
    self.mouth_animation_param_list = []
    self.eye_animation_param_list = []



  def run(self):
    self.running = True
    print("* processing")
    # vad = webrtcvad.Vad(2)
    start_time = time.time()
    avg_processing_time = 0
    processing_count = 0
    finished_processing = False
    while not finished_processing:

      data = self.recorder.get_next_data()

      if data is not None:
        ss = time.time()
        self.processed_frame_count += 1
        self.processed_frames.append(data)

        data_sample = np.frombuffer(data, dtype=np.int16)
        data_sample = np.ndarray.astype(data_sample, float) / 32768.0

        self.audio_buffer.append(data_sample)

        if len(self.audio_buffer) == 48:  # give 48 frames of speech

          with torch.no_grad():
            audio_window = np.concatenate(list(self.audio_buffer))
            input_values = tokenizer(audio_window, return_tensors="pt").input_values.to(device=device)
            logits = model(input_values).logits
            logits = logits[:, -32:, :].detach().cpu().numpy()
            #for i in range(logits.shape[1]):
            #   logits[0][i] = softmax(logits[0][i])
            animation_model_in = np.array(logits)

            #32d to 33d
            #animation_model_in = np.append(np.expand_dims(animation_model_in[:,-1,:], axis=0), animation_model_in, axis=1)


            # the prediction of current animation parameters
            mouth_animation_param, eye_animation_param = self.model_w2v2animation.predict(animation_model_in)
            #norm1
            #mouth_animation_param = denormalize_mouth(mouth_animation_param)
            #eye_animation_param = denormalize_eyes(eye_animation_param)
            #norm2
            mouth_animation_param = scaler_mouth.inverse_transform(mouth_animation_param)
            eye_animation_param = scaler_eye.inverse_transform(eye_animation_param)

            # print(mouth_animation_param.shape, eye_animation_param.shape)
            self.mouth_animation_param_list += [mouth_animation_param]
            self.eye_animation_param_list += [eye_animation_param]

          self.audio_buffer.popleft()

        tt = time.time()
        avg_processing_time += (tt - ss)
        processing_count += 1
      else:
        time.sleep(0)  # nothing to do, let other threads work
        if not self.running:
          finished_processing = True

    print('end processing')
    end = time.time()
    print("Total Processing Time: {} sec".format(end - start_time))
    print("Average Processing Time: {} msec".format((avg_processing_time / processing_count) * 1000))

  def stop(self):
    self.running = False

  def get_processed_frame_count(self):
    return self.processed_frame_count

  def get_processed_frames(self):
      return self.processed_frames

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

device = 'cuda:0'
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device=device)
CHANNELS = 1
CHUNK = 640  #640
FORMAT = pyaudio.paInt16
RATE = 16000 # 16000
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

stream_recording = RecordThread2()
stream_processing = ProcessingThread2(stream_recording, model=model, tokenizer=tokenizer)
stream_recording.start()
stream_processing.start()
time.sleep(1)
a = input("Press ENTER to exit...")

# tell threads to stop
stream_recording.stop()
stream_processing.stop()

# wait for recording thread to finish
stream_recording.join()

# wait for until all audio samples have been processed
stream_processing.join()

print("Number of Recorded Audio Chunks: {}".format(stream_recording.get_chunk_count()))
print("Number of Processed Audio Chunks: {}".format(stream_processing.get_processed_frame_count()))


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
waveFile.writeframes(b''.join(stream_processing.get_processed_frames()))
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

for i in range(len(stream_processing.get_mouth_animation_params())):
    mouth_params = torch.from_numpy(stream_processing.get_mouth_animation_params()[i]).to(dtype=torch.float32)
    eye_params = torch.from_numpy(stream_processing.get_eye_animation_params()[i]).to(dtype=torch.float32)
    pose_params = torch.zeros(1, 6).to(dtype=torch.float32)
    full_params = torch.cat([mouth_params, eye_params, pose_params], dim=1).to(dtype=torch.float32).to(device=render_device)
    face_renderer.update_expression(full_params)
    img = (face_renderer.render().squeeze(0).permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    img_bgr = cv2.cvtColor(src=img, code=cv2.COLOR_RGB2BGR)
    cv2.imwrite("animated_frames/frame{:04}.png".format(i), img_bgr)
    cv2.namedWindow('test')
    cv2.imshow('test', img_bgr)
    cv2.waitKey(1)


sound = AudioSegment.from_wav("recorded.wav")
#Selecting Portion we want to cut

StrtSec = 1.8
# Time to milliseconds conversion
StrtTime = StrtSec*1000
# Opening file and extracting portion of it
extract = sound[StrtTime:]
# Saving file in required location
extract.export("recorded2.wav", format="wav")
os.system("ffmpeg -y -i animated_frames/frame%04d.png -i recorded2.wav animation.mp4")
