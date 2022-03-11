import numpy as np

class Perceptron():
    """
    třída binárního klasifikátoru

    Atributy
    --------
    :w_ : 1d-array
    Vekotr váhových koeficientů

    :errors_ : list
    Počet špatně klasifikovaných příkladů během každé epochy
    """
    def __init__(self, eta=0.01, epochs=50, random_state=1):
        """
        konstruktor binárního klasifikátoru

        :param eta: {float} rychlost učení
        :param epochs: {integer} počet epoch (trénovacích iterací)
        :param random_state: {integer} semínko generátoru náhodných 
                             čísel pro inicializaci váhových koeficientů
        """
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state

    def fit(self, X, y):
        """
        funkce pro trénování klasifikátoru na poskytntých datech

        :param X: {2d-array} matice vlastností trénovacích příkladů
                             shape = [n_příklady, n_vlastnosti]
                             n_příklady: počet trénovacích příkladů
                             n_vlastnosti: počet vlastností
        :param y: {1d-array} vekotr skutečných tříd trénovacích příkladů
                             shape = [n_ppříklady]
        
        :return: self objekt
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """
        funkce pro výpočet lineárního vstupu

        :return: {1d-array} vektor lineárních vstupů trénovacíh příkladů
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """
        funkce pro predikci tříd vstupních dat

        :return: {1d-array} vektor predikovaných tříd vstupních dat
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)
        
        