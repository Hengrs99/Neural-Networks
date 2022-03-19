import numpy as np

class ADALINE():
    """
    třída adaptviního lineárního neuronu (ADALINE) pro binární klasifikaci

    Atributy
    --------
    :w_ : 1d-array
    Vekotr váhových koeficientů

    cost_ : list
    Průměrná hodnota ztrázoté funkce SSE v každé epoše
    """    
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        """
        konstruktor binárního klasifikátoru

        :param eta: {float} rychlost učení
        :param epochs: {integer} počet epoch
        :param random_state: {integer} semínko generátoru náhodných 
                             čísel pro inicializaci váhových koeficientů
        """
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """
        funkce pro trénování klasifikátoru na poskytntých datech

        :param X: {2d-array} matice vlastností tréninkových příkladů
                             shape = [n_příklady, n_vlastnosti]
                             n_příklady: počet tréninkových příkladů
                             n_vlastnosti: počet vlastností
        :param y: {1d-array} vekotr skutečných tříd tréninkových příkladů
                             shape = [n_ppříklady]
        
        :return: self
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)

        return self


    def net_input(self, X):
        """
        funkce pro výpočet lineárního vstupu

        :return: {1d-array} vektor lineárních vstupů tréninkových příkladů
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """
        funkce pro aplikaci lineární aktivační funkce na lineární vstup

        :return: {1d-array} vektor modifikovaných lineárních vstupů 
                            tréninkových příkladů (v tomto případě beze změny)
        """
        return X

    def predict(self, X):
        """
        funkce pro predikci tříd vstupních dat

        :return: {1d-array} vektor predikovaných tříd vstupních dat
        """
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

