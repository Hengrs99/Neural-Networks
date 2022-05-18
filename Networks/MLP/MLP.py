import numpy as np
import sys


class MLP_NeuralNetwork():
    """ 
    třída dopředné MLP sítě pro multinomiální klasifikaci

    Atributy
    --------
    :eval_ : dictionary
        Slovník pro shromažďování dat k zhodnocení z každé 
        epochy (ztrátové skóre, přesnost predikcí trénovacích a testovacích dat)
    :random : RandomState
        Generátor náhodných čísel
    """
    def __init__(self, n_hidden=30,
                 l2=0., epochs=100, eta=0.001,
                 shuffle=True, minibatch_size=1, seed=None):
        """
        konstrukotr MLP

        :param n_hidden : {int} počet neuronů ve skryté vrstvě  
        :param l2 : {float} hondonta lambda pro L2 regularizaci (0 = bez L2)
        :param epochs : {int} počet epoch
        :param eta : {float} rychlost učení
        :param shuffle : {bool} (true) pro každou epochu zamíchá pořadí dat
        :param minibatch_size : {int} počet trénovacích příkladů v dávce
        :param seed : {int} seed generátoru náhodných čísel
        """
        self.random = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size

    def _onehot(self, y, n_classes):
        """
        funkce pro převod cílových tříd do one-hot reprezentace
        
        :param y: {1d-array} vektor cílových tříd
        
        :return: {2d-array} matice cílových tříd ve one-hot reprezentaci
        """
        onehot = np.zeros((n_classes, y.shape[0]))
        for idx, val in enumerate(y):
            onehot[val, idx] = 1.

        return onehot.T

    def _sigmoid(self, z):
        """
        funkce pro výpočet logistické (sigmoid) funkce
        
        :param z: {int} lineární vstup

        :return: {float} hodnota logistické (sigmoid) funkce
        """
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def _forward(self, X):
        """
        funkce pro výpočty dopředného šíření

        :param X: {2d-array} matice trénovacích příkladů

        :return z_h: {2d-array} lineární vstupy neuronů skryté vrsty
        :return a_h: {2d-array} hodnoty aktivovaných neuronů skryté vsrtvy
        :return z_out: {2d-array} lineární vstupy neuronů výstupní vrsty
        :return a_out: {2d-array} hodnoty aktivovaných neuronů výstupní vsrtvy
        """

        z_h = np.dot(X, self.w_h) + self.b_h
        a_h = self._sigmoid(z_h)
        z_out = np.dot(a_h, self.w_out) + self.b_out
        a_out = self._sigmoid(z_out)

        return z_h, a_h, z_out, a_out

    def _compute_cost(self, y_enc, output):
        """
        funkce pro výpočet ztrátové funkce

        :param y_enc: {2d-array} matice cílových tříd ve one-hot reprezentaci
        :param output: {2d-array} matice aktivací neuronů výstupní vrstvy
    
        :return: {float} ztrátové skóre (včetně L2 regularizace)
        """
        L2_term = (self.l2 *(np.sum(self.w_h ** 2.) +np.sum(self.w_out ** 2.)))
        term1 = -y_enc * (np.log(output + 1e-5))
        term2 = (1. - y_enc) * np.log(1. - output + 1e-5)
        cost = np.sum(term1 - term2) + L2_term

        # konstanty 1e-5 jsou ve výpočtu pouze kvůli jisté
        # nedokonalosti Pythonu, která by při extrémních hodnotách 
        # mohla zapříčinit zhroucení programu

        return cost

    def predict(self, X):
        """
        funkce pro predikci tříd
        
        :param X: {2d-array} soubor dat k predikci

        :return: {1d-array} predikované třídy
        """
        z_h, a_h, z_out, a_out = self._forward(X)
        y_pred = np.argmax(z_out, axis=1)

        return y_pred

    def fit(self, X_train, y_train, X_valid, y_valid):
        """ 
        funkce pro trénování sítě

        :param X_train: {2d-array} trénovací datový soubor
        :param y_train: {1d-array} vektor cílových tříd trénovacího souboru
        :param X_valid: {2d-array} validační datový soubor
        :param y_valid: {1d-array} vektor cílových tříd validačního souboru 
        
        :return: self
        """
        n_output = np.unique(y_train).shape[0]  # počet tříd
        n_features = X_train.shape[1]

        # 1 => Náhodná inicializace váhových koeficientů a aktivce neuronů

        # váhy pro vrstvy: vstupní -> skrytá
        self.b_h = np.zeros(self.n_hidden)
        self.w_h = self.random.normal(loc=0.0, scale=0.1,
                                      size=(n_features, self.n_hidden))

        # váhy pro vrstvy: skrytá -> výstupní
        self.b_out = np.zeros(n_output)
        self.w_out = self.random.normal(loc=0.0, scale=0.1,
                                        size=(self.n_hidden, n_output))

        epoch_strlen = len(str(self.epochs))  # formátování pokroku
        self.eval_ = {'cost': [], 'train_acc': [], 'valid_acc': []}

        y_train_enc = self._onehot(y_train, n_output)

        # iterace přes epochy
        for i in range(self.epochs):

            # iterace přes dávky
            indices = np.arange(X_train.shape[0])

            if self.shuffle:
                self.random.shuffle(indices)

            for start_idx in range(0, indices.shape[0] - self.minibatch_size +
                                   1, self.minibatch_size):
                batch_idx = indices[start_idx:start_idx + self.minibatch_size]

                # výpočet dopředného šíření
                z_h, a_h, z_out, a_out = self._forward(X_train[batch_idx])

                # 2 => Zpětné šíření

                sigma_out = a_out - y_train_enc[batch_idx]
                sigmoid_derivative_h = a_h * (1. - a_h)

                sigma_h = (np.dot(sigma_out, self.w_out.T) *
                           sigmoid_derivative_h)

                grad_w_h = np.dot(X_train[batch_idx].T, sigma_h)
                grad_b_h = np.sum(sigma_h, axis=0)

                grad_w_out = np.dot(a_h.T, sigma_out)
                grad_b_out = np.sum(sigma_out, axis=0)

                delta_w_h = (grad_w_h + self.l2*self.w_h)
                delta_b_h = grad_b_h  # bias není regularizován
                self.w_h -= self.eta * delta_w_h
                self.b_h -= self.eta * delta_b_h

                delta_w_out = (grad_w_out + self.l2*self.w_out)
                delta_b_out = grad_b_out  # bias není regularizován
                self.w_out -= self.eta * delta_w_out
                self.b_out -= self.eta * delta_b_out

            # 3 => Evaluace

            z_h, a_h, z_out, a_out = self._forward(X_train)
            cost = self._compute_cost(y_enc=y_train_enc,
                                      output=a_out)

            y_train_pred = self.predict(X_train)
            y_valid_pred = self.predict(X_valid)

            train_acc = ((np.sum(y_train == y_train_pred)).astype(np.float) /
                         X_train.shape[0])
            valid_acc = ((np.sum(y_valid == y_valid_pred)).astype(np.float) /
                         X_valid.shape[0])

            sys.stderr.write('\r%0*d/%d | Cost: %.2f '
                             '| Train/Valid Acc.: %.2f%%/%.2f%% ' %
                             (epoch_strlen, i+1, self.epochs, cost,
                              train_acc*100, valid_acc*100))
            sys.stderr.flush()

            self.eval_['cost'].append(cost)
            self.eval_['train_acc'].append(train_acc)
            self.eval_['valid_acc'].append(valid_acc)

        return self
        