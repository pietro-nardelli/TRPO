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
        """ Construct NN, including loss function and optimizer"""
        obs = Input(shape=(self.obs_dim,), dtype='float32')
        hid1_units = self.obs_dim * self.hid1
        hid2_units = hid1_units/4 
        hid3_units = 5
        self.lr = 1e-2 / np.sqrt(hid2_units)

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
        #Concatenate previous X with new X to obtain x_train. Same with Y
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
