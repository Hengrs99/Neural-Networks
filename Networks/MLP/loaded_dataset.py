import numpy as np
from neuralnet2 import MLP_NeuralNet

nn = MLP_NeuralNet(n_hidden=100,
                   l2=0.01,
                   n_epochs=2,
                   eta=0.0005,
                   minibatch_size=100,
                   shuffle=True, seed=1)

dataset = np.load('dataset/mnist_scaled.npz')
X_train, y_train, X_test, y_test = [dataset[f] for f in ['X_train', 'y_train', 'X_test', 'y_test']]

nn.fit(X_train=X_train[:55000], 
       y_train=y_train[:55000],
       X_valid=X_train[55000:],
       y_valid=y_train[55000:])
