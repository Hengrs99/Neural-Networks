import numpy as np

class MLP_NeuralNet:
    def __init__(self, 
                 n_hidden=16, 
                 l2=0., 
                 n_epochs=200, 
                 eta=0.001, 
                 shuffle=True, 
                 minibatch_size=1, 
                 seed=None):
        
        self.randomState = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.l2_reg = l2
        self.n_epochs = n_epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size

    def _to_onehot(self, y, n_classes):
        onehot = np.zeros((n_classes, y.shape[0]))
        for index, value in enumerate(y):
            onehot[value, index] = 1.
        return onehot.T

    def _apply_sigmoid(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def _forward_propagate(self, X):
        z_hid = np.dot(X, self.w_hid) + self.bias_hid
        a_hid = self._apply_sigmoid(z_hid)
        
        z_out = np.dot(a_hid, self.w_out) + self.bias_out
        a_out = self._sigmoid(z_out)

        return z_hid, a_hid, z_out, a_out

    def _compute_cost(self, y_onehot, out):
        L2_reg = (self.l2 * (np.sum(self.w_hid ** 2.) + np.sum(self.w_out ** 2.)))
        term1 = -y_onehot * (np.log(out))
        term2 = (1. - y_onehot) * np.log(1. - out)
        cost = np.sum(term1 - term2) + L2_reg

    def predict(self, X):
        z_hid, a_hid, z_out, a_out = self._forward_propagate(X)
        y_pred = np.argmax(z_out, axis=1)

    def fit(self, X_train, y_train, X_valid, y_valid):
        n_out = np.unique(y_train).shape[0]
        n_features = X_train.shape[1]

        # Weight initialization

        self.bias_hid = np.zeros(self.n_hidden)
        self.w_hid = self.randomState.normal(loc=0.0,
                                        scale=0.1,
                                        size=(n_features, self.n_hidden))

        self.bias_out = np.zeros(n_out)
        self.w_out = self.randomState.normal(loc=0.0,
                                        scale = 0.1,
                                        size=(self.n_hidden, n_out))

        epoch_strlen = len(str(self.n_epochs))
        self.evaluation = {'cost': [], 'train_acc': [], 'valid_acc': []}

        y_train_onehot = self._to_onehot(y_train, n_out)


        for i in range(self.n_epochs):

            indicies = np.arange(X_train.shape[0])
            print(indicies)
            if self.shuffle:
                self.randomState.shuffle(indicies)
