'''leakyReLUextension.py
Builds upon the mlp class to implement leaky ReLU as the hidden layer activaiton function instead of 
Sam Polyakov and Teagan Turner
CS 343: Neural Networks
Fall 2025
Project 2 Extension: Multilayer Perceptrons
'''

import numpy as np
from mlp import MLP

class leakyReLUextension(MLP):
    def predict(self, features):
        '''Changing the activation function to leaky ReLU. Instead of zeroing out negative values, we scale them by 0.01.
        '''
        y_net_in = features @ self.y_wts + self.y_b  
        y_net_act = np.where(y_net_in > 0, y_net_in, 0.01 * y_net_in)        
        z_net_in = y_net_act @ self.z_wts + self.z_b

        return np.argmax(z_net_in, axis=1)
    

    def forward(self, features, y, reg=0):
        '''Same forward function but changed to leaky ReLU activation function.
        '''
        y_net_in = features @ self.y_wts + self.y_b            
        y_net_act = np.where(y_net_in > 0, y_net_in, 0.01 * y_net_in)         

        z_net_in = y_net_act @ self.z_wts + self.z_b           
        z_shift = z_net_in - np.max(z_net_in, axis=1, keepdims=True)
        exp_scores = np.exp(z_shift)
        z_net_act = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        N = features.shape[0]
        y = np.asarray(y, dtype=int).ravel()

        tiny = 1e-12
        probs = z_net_act[np.arange(N), y]
        correct_logprobs = -np.log(np.clip(probs, tiny, 1.0))
        data_loss = np.mean(correct_logprobs)

        reg_loss = 0.5 * reg * (np.sum(self.y_wts * self.y_wts) + np.sum(self.z_wts * self.z_wts))
        loss = data_loss + reg_loss
        return y_net_in, y_net_act, z_net_in, z_net_act, loss
    
    def backward(self, features, y, y_net_in, y_net_act, z_net_in, z_net_act, reg=0):
        '''Same backwards pass function but change ReLU derivative to leaky ReLU derivative
        '''
        N = features.shape[0]
        y_one_hot = self.one_hot(y, self.num_output_units)
        dz_net_in = (z_net_act - y_one_hot) / N

        dz_wts = y_net_act.T @ dz_net_in
        dz_b = np.sum(dz_net_in, axis=0)

        dz_wts += reg * self.z_wts

        dy_net_act = dz_net_in @ self.z_wts.T

        leakyRelu_derivative = np.where(y_net_in > 0, 1, 0.01)

        dy_net_in = dy_net_act * leakyRelu_derivative

        dy_wts = features.T @ dy_net_in
        dy_b = np.sum(dy_net_in, axis=0)

        dy_wts += reg * self.y_wts

        return dy_wts, dy_b, dz_wts, dz_b
    
