"""
NN Policy with KL Divergence Constraint
"""
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras.optimizers import Adam
import numpy as np


class Policy(object):
    def __init__(self, obs_dim, act_dim, kl_targ, hid1_size, init_logvar):
        """
        Args:
            obs_dim: num observation dimensions (int)
            act_dim: num action dimensions (int)
            kl_targ: target KL divergence between pi_old and pi_new
            hid1_size: size of first hidden layer
            init_logvar: natural log of initial policy variance
        """
        self.kl_targ = kl_targ
        self.epochs = 20
        self.trpo = TRPO(obs_dim, act_dim, hid1_size, kl_targ, init_logvar)
        self.policy = self.trpo.get_layer('policy_nn')
        self.lr = 0.000225
        self.trpo.compile(optimizer=Adam(self.lr))

    def LogProb(self, inputs):
        """Calculates log probabilities of a batch of actions."""
        actions, act_means, act_logvars = inputs
        logp = -0.5 * K.sum(act_logvars, axis=-1, keepdims=True)
        logp += -0.5 * K.sum(K.cast(K.square(actions - act_means), dtype='float32') / K.exp(act_logvars),
                            axis=-1, keepdims=True)
        return logp


    def sample(self, obs):
        """Draw sample from policy."""
        act_means, act_logvars = self.policy(obs)
        # logvar = log(sigma^2) => sigma^2 = e^logvar => sigma = sqrt(e^logvar)
        act_stddevs = np.exp(act_logvars / 2)

        return np.random.normal(act_means, act_stddevs).astype(np.float64)

    def update(self, observes, actions, advantages):
        """ Update policy based on observations, actions and advantages

        Args:
            observes: observations, shape = (N, obs_dim)
            actions: actions, shape = (N, act_dim)
            advantages: advantages, shape = (N,)
        """
        K.set_value(self.trpo.optimizer.lr, self.lr)
        old_means, old_logvars = self.policy(observes)
        old_means = old_means.numpy()
        old_logvars = old_logvars.numpy()

        old_logp = self.LogProb([actions, old_means, old_logvars])

        old_logp = old_logp.numpy()
        loss, kl = 0, 0
        
        
        filepath = "keras-weights.h5"
        for e in range(self.epochs):
            loss = self.trpo.train_on_batch([observes, actions, advantages,
                                             old_means, old_logvars, old_logp])

            kl = self.trpo.predict_on_batch([observes, actions, advantages,
                                                      old_means, old_logvars, old_logp])

            kl = np.mean(kl)
            if kl > self.kl_targ:  # early stopping if D_KL diverges badly
                self.trpo.load_weights(filepath)
                break
            else:
                self.trpo.save_weights(filepath)
                kl = self.trpo.predict_on_batch([observes, actions, advantages,
                                                      old_means, old_logvars, old_logp])
        

class PolicyNN(Layer):
    """ Neural net for policy approximation function.

    Policy parameterized by Gaussian means and variances. NN outputs mean
     action based on observation. Trainable variables hold log-variances
     for each action dimension (i.e. variances not determined by NN).
    """
    def __init__(self, obs_dim, act_dim, hid1_size, init_logvar, **kwargs):
        super(PolicyNN, self).__init__(**kwargs)
        self.batch_sz = None
        self.init_logvar = init_logvar
        hid1_units = hid1_size
        hid2_units = hid1_size/2  
        hid3_units = act_dim
        self.lr = 0.000225  
        
        self.dense1 = Dense(hid1_units, activation='tanh', input_shape=(obs_dim,))
        self.dense2 = Dense(hid2_units, activation='tanh', input_shape=(hid1_units,))
        self.dense3 = Dense(hid3_units, activation='tanh', input_shape=(hid2_units,))
        self.dense4 = Dense(act_dim, input_shape=(hid3_units,))

        self.logvars = self.add_weight(shape=(1,act_dim),
                                       trainable=True, initializer='zeros')
        print('Policy Params -- h1: {}, h2: {}, h3: {}, lr: {:.3g}'
              .format(hid1_units, hid2_units, hid3_units, self.lr))
        

    def build(self, input_shape):
        self.batch_sz = input_shape[0]

    def call(self, inputs, **kwargs):
        y = self.dense1(inputs)
        y = self.dense2(y)
        y = self.dense3(y)
        means = self.dense4(y)
        logvars = self.dense4(y)
        logvars = K.sum(self.logvars, axis=0, keepdims=True) + self.init_logvar
        logvars = K.tile(logvars, (self.batch_sz, 1))

        return [means, logvars]


class TRPO(Model):
    def __init__(self, obs_dim, act_dim, hid1_size, kl_targ, init_logvar, **kwargs):
        super(TRPO, self).__init__(**kwargs)
        self.kl_targ = kl_targ
        self.policy = PolicyNN(obs_dim, act_dim, hid1_size, init_logvar)
        self.act_dim = act_dim

    def call(self, inputs):
        obs, act, adv, old_means, old_logvars, old_logp = inputs
        new_means, new_logvars = self.policy(obs)

        def LogProb(inputs):
            """Calculates log probabilities of a batch of actions."""
            actions, act_means, act_logvars = inputs
            logp = -0.5 * K.sum(act_logvars, axis=-1, keepdims=True)
            logp += -0.5 * K.sum(K.cast(K.square(actions - act_means), dtype='float32') / K.exp(act_logvars),
                                axis=-1, keepdims=True)
            return logp

        def KLDiv(inputs):
            """
            Layer calculates:
                1. KL divergence between old and new distributions
                2. Entropy of present policy

            https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback.E2.80.93Leibler_divergence
            https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Entropy
            """


            old_means, old_logvars, new_means, new_logvars = inputs
            log_det_cov_old = K.sum(old_logvars, axis=-1, keepdims=True)
            log_det_cov_new = K.sum(new_logvars, axis=-1, keepdims=True)
            trace_old_new = K.sum(K.exp(old_logvars - new_logvars), axis=-1, keepdims=True)
            kl = 0.5 * (log_det_cov_new - log_det_cov_old + trace_old_new +
                        K.sum(K.square(new_means - old_means) /
                            K.exp(new_logvars), axis=-1, keepdims=True) -
                        np.float64(self.act_dim))

            return kl




        new_logp = LogProb([act, new_means, new_logvars])
        kl = KLDiv([old_means, old_logvars,
                                        new_means, new_logvars])
        
        loss1 = -K.mean(adv * K.exp(new_logp - old_logp))
        self.add_loss(loss1)

        return kl
    

        

