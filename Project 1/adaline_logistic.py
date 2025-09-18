'''adaline_logistic.py
Implementing ADALINE for classification using the logistic activation function
Sam Polyakov and Teagan Turner
CS343: Neural Networks
Project 1: Single Layer Networks'''

import numpy as np
from adaline import Adaline

class AdalineLogistic(Adaline):

    def activation(self, net_in):
        '''Applies the activation function to the net input and returns the output neuron's activation.
        It is simply the identify function for vanilla ADALINE: f(x) = x

        Parameters:
        ----------
        net_in: ndarray. Shape = [Num samples N,]

        Returns:
        ----------
        net_act. ndarray. Shape = [Num samples N,]

        We changed the activation function to the sigmoid function.
        '''
        return 1/(1 + np.exp(-net_in))
    
    def predict(self, features):
        ''' Predict class labels after training the network
        
        Parameters:
        ----------
        features: ndarray. Shape = [Num samples N, Num features M]
            Collection of input vectors.

        Returns:
        ----------
        Predicted class labels. Shape = [Num samples N,]

        We changed the classification rule to 0.5 and recoded classes to 0 and 1.

        '''
        net_in = self.net_input(features)
        net_act = self.activation(net_in)
        return np.where(net_act >= 0.5, 1, 0)
    
    def loss(self, y, net_act):
        ''' Computes the loss for all samples

        Parameters:
        ----------
        y: ndarray. Shape = [Num samples N,]
            True class labels.

        net_act: ndarray. Shape = [Num samples N,]
            Output neuron activations for each input sample.

        Returns:
        ----------
        The loss from our logistic function.
        '''
        tiny = 1e-15
        net_act = np.clip(net_act, tiny, 1 - tiny)
        cross_entropy_loss = np.sum((-y * np.log(net_act)) - ((1 - y) * np.log(1 - net_act)))
        return cross_entropy_loss
            
