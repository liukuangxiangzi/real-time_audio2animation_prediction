import os
import logging
import argparse
import numpy as np
from datetime import datetime
from keras.utils.vis_utils import plot_model
from keras.layers import Input, LSTM, LeakyReLU
from keras.models import Model
from keras.models import load_model
from keras.callbacks import TensorBoard
from keras.layers import Dropout
from keras.optimizers import adam_v2
from keras.utils.vis_utils import plot_model
from keras.layers import Dense, Flatten, Conv1D

#tensorboard
#tensorboard --logdir /Users/liukuangxiangzi/PycharmProjects/PhonemeNet/logs/fit/ --host=127.0.0.1
log_dir = os.path.join(
    "logs",
    "fit",
    datetime.now().strftime("%Y%m%d-%H%M%S"),
)
tbCallBack = TensorBoard(log_dir= log_dir, histogram_freq=0, write_graph=True, write_images=True)
# configure logging
logging.basicConfig(level=logging.INFO, filename='myapp.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument('--train-input-data-dir', type=str, default='data/phones/train_viseme_ID.npy', help='path to training input data')
parser.add_argument('--test-input-data-dir', type=str, default='data/phones/test_viseme_ID.npy', help='path to test input data')
parser.add_argument('--train-eyes-data-dir', type=str, default='data/animation-params/eyes/train_scaled_eyes.npy', help='path to eye animation params training data')
parser.add_argument('--test-eyes-data-dir', type=str, default='data/animation-params/eyes/test_scaled_eyes.npy', help='path to eye animation params test data')
parser.add_argument('--train-mouth-data-dir', type=str, default='data/animation-params/mouth/train_scaled_mouth.npy', help='path to mouth animation params training data')
parser.add_argument('--test-mouth-data-dir', type=str, default='data/animation-params/mouth/test_scaled_mouth.npy', help='path to mouth animation params test data')
parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--batch-size', type=int, default=32, help='batch size for training')
parser.add_argument('--resume-training', type=bool, default=False, help='whether to resume training from a saved model')
parser.add_argument('--save-model-dir', type=str, default='visemeID/model/', help='directory where trained model will be saved')
parser.add_argument('--load-model-dir', type=str, default=None, help='path to trained model to resume training from')
args = parser.parse_args()

#load data
x_train = np.load(args.train_input_data_dir)
x_test = np.load(args.test_input_data_dir)
y_eye_train = np.load(args.train_eyes_data_dir)
y_eye_test = np.load(args.test_eyes_data_dir)
y_mouth_train = np.load(args.train_mouth_data_dir)
y_mouth_test = np.load(args.test_mouth_data_dir)

#network
if not args.resume_training:
    n_features = x_train.shape[1]
    lr = 1e-4
    initializer = 'glorot_uniform'

    # define trained_model
    net_in = Input(shape=(n_features))
    l1 = Dense(128,
               kernel_initializer=initializer, name='l1')(net_in)
    lrelu1 = LeakyReLU(0.2)(l1)
    l2 = Dense(256,
               kernel_initializer=initializer, name='l2')(net_in)
    lrelu2 = LeakyReLU(0.2)(l2)
    l3 = Dense(512,
               kernel_initializer=initializer, name='l3')(lrelu2)
    lrelu2_2 = LeakyReLU(0.2)(l3)
    #flatten = Flatten()(lrelu2_2)
    d1 = Dense(512,
               kernel_initializer=initializer, name='m1')(lrelu2_2)
    lrelu3 = LeakyReLU(0.2)(d1)
    d2 = Dense(256,
               kernel_initializer=initializer, name='m2')(lrelu3)
    lrelu4 = LeakyReLU(0.2)(d2)
    d3 = Dense(256,
               kernel_initializer=initializer, name='m_out',activation='linear')(lrelu4)
    e1 = Dense(512,
               kernel_initializer=initializer, name='e1')(lrelu2_2)
    elrelu3 = LeakyReLU(0.2)(e1)
    e2 = Dense(256,
               kernel_initializer=initializer, name='e2')(elrelu3)
    elrelu4 = LeakyReLU(0.2)(e2)
    e3 = Dense(256,
               kernel_initializer=initializer, name='e_out',activation='linear')(elrelu4)#
    model = Model(inputs=net_in, outputs=[d3,e3])
    model.summary()
    opt = adam_v2.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss={'m_out':'mse', 'e_out':'mse'}, loss_weights={'m_out':1.0, 'e_out':1.0},metrics=['accuracy'])
else:
    model = load_model(args.load_model_dir)
    logging.info('Trained_model loaded')



history = model.fit(x = x_train,
          y = [y_mouth_train, y_eye_train],
          validation_data = (x_test,[y_mouth_test, y_eye_test]),
          epochs = args.epochs,
          batch_size=args.batch_size,
          callbacks=[tbCallBack],
          shuffle=True)

plot_model(model, to_file='model_visemeID_params_plot.png', show_shapes=True, show_layer_names=True)
logging.info('Starting training')
logging.info('Epoch: %s, Batch: %s', args.epochs, args.batch_size)
training_loss = {output_name: history.history[f"{output_name}_loss"] for output_name in model.output_names}
validation_loss = {output_name: history.history[f"val_{output_name}_loss"] for output_name in model.output_names}
loss = {"training": training_loss, "validation": validation_loss}
logging.info(f"Loss: {loss}")
model.save(args.save_model_dir + 'mlp_epoch' + str(args.epochs) + '_bs' + str(args.batch_size) +'.h5')
logging.info('Saved to: ' + args.save_model_dir)