"""
State-Value Function
"""
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

import numpy as np


class NNValueFunction(object):
    """ NN-based state-value function """
    def __init__(self, obs_dim, hid1):
        """
        Args:
            obs_dim: number of dimensions in observation vector (int)
            hid1: size of first hidden layer
        """
        self.replay_buffer_x = None
        self.replay_buffer_y = None
        self.obs_dim = obs_dim
        self.hid1 = hid1
        self.epochs = 10
        self.lr = None  # learning rate set in _build_model()
        self.model = self._build_model()

    def _build_model(self):
        """ Construct TensorFlow graph, including loss function, init op and train op """
        obs = Input(shape=(self.obs_dim,), dtype='float32')
        # hid1 layer size is 10x obs_dim, hid3 size is 10, and hid2 is geometric mean
        hid1_units = self.obs_dim * self.hid1
        hid3_units = 5 
        hid2_units = int(np.sqrt(hid1_units * hid3_units))
        # heuristic to set learning rate based on NN size (tuned on 'Hopper-v1')
        self.lr = 1e-2 / np.sqrt(hid2_units)  # 1e-2 empirically determined
        print('Value Params -- h1: {}, h2: {}, h3: {}, lr: {:.3g}'
              .format(hid1_units, hid2_units, hid3_units, self.lr))
        y = Dense(hid1_units, activation='tanh')(obs)
        y = Dense(hid2_units, activation='tanh')(y)
        y = Dense(hid3_units, activation='tanh')(y)
        y = Dense(1)(y)
        model = Model(inputs=obs, outputs=y)
        optimizer = Adam(self.lr)
        model.compile(optimizer=optimizer, loss='mse')

        return model

    def fit(self, x, y):
        """ Fit model to current data batch + previous data batch

        Args:
            x: features
            y: target
        """

        if self.replay_buffer_x is None:
            x_train, y_train = x, y
        else:
            x_train = np.concatenate([x, self.replay_buffer_x])
            y_train = np.concatenate([y, self.replay_buffer_y])
        self.replay_buffer_x = x
        self.replay_buffer_y = y
        self.model.fit(x_train, y_train, epochs=self.epochs,
                       shuffle=True, verbose=0)

        
    def predict(self, x):
        """ Predict method """
        return self.model.predict(x)
