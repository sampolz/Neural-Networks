'''multiHiddenExtension.py
Constructs a mlp with two hidden layers, using leaky ReLU and sigmoid as the hidden layer activations and softmax as the output layer activation.
Sam Polyakov and Teagan Turner
CS 343: Neural Networks
Fall 2025
Project 2 Extension: Multilayer Perceptrons
'''

import numpy as np
from mlp import MLP

class multiHiddenMLP(MLP):
    def __init__(self, num_input_units, num_hidden1_units, num_hidden2_units, num_output_units):
        self.num_input_units = num_input_units
        self.num_hidden1_units = num_hidden1_units
        self.num_hidden2_units = num_hidden2_units
        self.num_output_units = num_output_units

        self.initialize_wts(num_input_units, num_hidden1_units, num_hidden2_units, num_output_units)

    def get_x_wts(self):
        '''Returns a copy of the first hidden layer wts'''
        return self.x_wts.copy()

    def get_y_wts(self):
        '''Returns a copy of the second hidden layer wts'''
        return self.y_wts.copy()


    def initialize_wts(self, M, H1, H2, C, std=0.1, r_seed=None):
        '''Initialize weights and biases for a two-hidden-layer MLP.
        Parameters:
        -----------
        M: int. Num input features
        H1: int. Number of first hidden unit layer
        H2: int. Num hidden units
        C: int. Num output units. Equal to # data classes.
        std: float. Standard deviation of the normal distribution of weights
        r_seed: None or int. Random seed for weight initialization.
        '''

        rng = np.random.default_rng(r_seed)

        self.x_wts = rng.normal(0.0, std, size=(M, H1))
        self.y_wts = rng.normal(0.0, std, size=(H1, H2))
        self.z_wts = rng.normal(0.0, std, size=(H2, C))

        self.x_b = np.zeros(H1)
        self.y_b = np.zeros(H2)
        self.z_b = np.zeros(C)
        
    def predict(self, features):
        '''Predicts the int-coded class value for network inputs ('features').

        NOTE: Loops of any kind are NOT ALLOWED in this method!

        Parameters:
        -----------
        features: ndarray. shape=(mini-batch size, num features)

        Returns:
        -----------
        y_pred: ndarray. shape=(mini-batch size,).
            This is the int-coded predicted class values for the inputs passed in.
            NOTE: You can figure out the predicted class assignments without applying the
            softmax net activation function — it will not affect the most active neuron.
        '''

        #First hidden layer with leaky ReLU activation
        h1_net_in = features @ self.x_wts + self.x_b
        h1_net_act = np.where(h1_net_in > 0, h1_net_in, 0.01 * h1_net_in)

        #Second hidden layer with sigmoid activation
        h2_net_in = h1_net_act @ self.y_wts + self.y_b
        h2_net_act = 1 / (1 + np.exp(-h2_net_in))

        z_net_in = h2_net_act @ self.z_wts + self.z_b

        return np.argmax(z_net_in, axis=1)

    def forward(self, features, y, reg=0):
        '''Same forward function but changed to two hidden layers with leaky ReLU and sigmoid activations.
        '''
        h1_net_in = features @ self.x_wts + self.x_b
        h1_net_act = np.where(h1_net_in > 0, h1_net_in, 0.01 * h1_net_in)

        h2_net_in = h1_net_act @ self.y_wts + self.y_b
        h2_net_act = 1 / (1 + np.exp(-h2_net_in))

        z_net_in = h2_net_act @ self.z_wts + self.z_b
        z_shift = z_net_in - np.max(z_net_in, axis=1, keepdims=True)
        exp_scores = np.exp(z_shift)
        z_net_act = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        N = features.shape[0]
        y = np.asarray(y, dtype=int).ravel()

        tiny = 1e-12
        probs = z_net_act[np.arange(N), y]
        correct_logprobs = -np.log(np.clip(probs, tiny, 1.0))
        data_loss = np.mean(correct_logprobs)

        reg_loss = 0.5 * reg * (np.sum(self.x_wts * self.x_wts) + np.sum(self.y_wts * self.y_wts) + np.sum(self.z_wts * self.z_wts))
        loss = data_loss + reg_loss
        return h1_net_in, h1_net_act, h2_net_in, h2_net_act, z_net_in, z_net_act, loss

    def backward(self, features, y, h1_net_in, h1_net_act, h2_net_in, h2_net_act, z_net_in, z_net_act, reg=0):
        N = features.shape[0]
        y_one_hot = self.one_hot(y, self.num_output_units)
        
        
        dz_net_in = (z_net_act - y_one_hot) / N
        dz_wts = h2_net_act.T @ dz_net_in
        dz_b = np.sum(dz_net_in, axis=0)
        dz_wts += reg * self.z_wts


        dh2_net_act = dz_net_in @ self.z_wts.T
        dh2_net_in = dh2_net_act * (h2_net_act * (1 - h2_net_act))
        dy_wts = h1_net_act.T @ dh2_net_in
        dy_b = np.sum(dh2_net_in, axis=0)
        dy_wts += reg * self.y_wts


        leakyRelu_derivative = np.where(h1_net_in > 0, 1, 0.01)

        dh1_net_act = dh2_net_in @ self.y_wts.T
        dh1_net_in = dh1_net_act * leakyRelu_derivative

        dx_wts = features.T @ dh1_net_in
        dx_b = np.sum(dh1_net_in, axis=0)
        dx_wts += reg * self.x_wts

        return dx_wts, dx_b, dy_wts, dy_b, dz_wts, dz_b


    def fit(self, features, y, x_validation, y_validation, n_epochs=500, lr=0.0001, mini_batch_sz=256, reg=0,
        r_seed=None, verbose=2, print_every=100):
        '''Trains the MLP using mini-batch stochastic gradient descent. Upgraded the fit function to handle two hidden layers.
        '''
        N = features.shape[0]
        rng = np.random.default_rng(r_seed)
        if r_seed is not None:
            self.initialize_wts(self.num_input_units, self.num_hidden1_units, self.num_hidden2_units, self.num_output_units, r_seed=r_seed)

        loss_history = []
        train_acc_history = []
        validation_acc_history = []

        iters_per_epoch = max(1, (N + mini_batch_sz - 1) // mini_batch_sz)

        if verbose > 0:
            print(f"Starting to train network...There will be {n_epochs} epochs and "
                f"{n_epochs * iters_per_epoch} iterations total, {iters_per_epoch} iter/epoch.")

        for epoch in range(n_epochs):
            indices = rng.permutation(N)

            for start in range(0, N, mini_batch_sz):
                end = min(start + mini_batch_sz, N)
                batch_idx = indices[start:end]

                Xb = features[batch_idx]
                yb = y[batch_idx]

                h1_net_in, h1_net_act, h2_net_in, h2_net_act, z_net_in, z_net_act, loss = self.forward(Xb, yb, reg=reg)
                loss_history.append(loss)

                dx_wts, dx_b, dy_wts, dy_b, dz_wts, dz_b = self.backward(Xb, yb, h1_net_in, h1_net_act, h2_net_in, h2_net_act, z_net_in, z_net_act, reg=reg)

                self.x_wts -= lr * dx_wts
                self.x_b   -= lr * dx_b
                self.y_wts -= lr * dy_wts
                self.y_b   -= lr * dy_b
                self.z_wts -= lr * dz_wts
                self.z_b   -= lr * dz_b

            train_pred = self.predict(features)
            val_pred   = self.predict(x_validation)
            train_acc  = self.accuracy(y, train_pred)
            val_acc    = self.accuracy(y_validation, val_pred)

            train_acc_history.append(train_acc)
            validation_acc_history.append(val_acc)

            if verbose > 0 and (epoch % print_every == 0):
                print(f"Completed Epoch {epoch}/{n_epochs-1}. "
                    f"Training loss: {loss_history[-1]:.2f}. "
                    f"Training acc: {train_acc*100:.2f}%. "
                    f"Validation acc: {val_acc*100:.2f}%.")

        if verbose > 0:
            print("Finished training!")

        return loss_history, train_acc_history, validation_acc_history