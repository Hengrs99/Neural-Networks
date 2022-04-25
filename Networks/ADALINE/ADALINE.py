import numpy as np

class ADALINE():
    """
    třída adaptviního lineárního neuronu (ADALINE) pro binární klasifikaci

    Atributy
    --------
    :w_ : 1d-array
    Vekotr váhových koeficientů

    cost_ : list
    Průměrná hodnota ztrátové funkce SSE v každé epoše
    """    
    def __init__(self, eta=0.01, epochs=50, random_state=1):
        """
        konstruktor binárního klasifikátoru

        :param eta: {float} rychlost učení
        :param epochs: {integer} počet epoch
        :param random_state: {integer} semínko generátoru náhodných čísel
        :param rgen: {RandomState} generátor náhodných čísel
        """
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state
        self.rgen = np.random.RandomState(self.random_state)

    def fit(self, X, y):
        """
        funkce pro trénování klasifikátoru na poskytntých datech

        :param X: {2d-array} matice příznaků trénovacích příkladů
                             shape = [n_příklady, n_příznaky]
                             n_příznaky: počet trénovacích příkladů
                             n_příznaky: počet příznaků
        :param y: {1d-array} vekotr skutečných tříd trénovacích příkladů
                             shape = [n_příklady]
        
        :return: self
        """
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.epochs):
            X, y = self._shuffle(X, y)
            cost = []
            for xi, gt in zip(X, y):
                cost.append(self._update_weights(xi, gt))
            avg_cost = sum(cost) / int(len(y))
            self.cost_.append(avg_cost)
   
        return self

    def _shuffle(self, X, y):
        """
        funkce pro určení náhodného pořadí dat
        
        :return: {1d-array} vekotr příznaků náhodného příkladu,
                 {integer} cílová třída
        """
        r = self.rgen.permutation(len(y))

        return X[r], y[r]

    def _update_weights(self, xi, gt):
        """
        funkce pro aktualizaci vah podle učebního pravidla
        
        :param xi: {1d-array} vektor příznaků
        :param gt: {integer} cílová třída

        :return: {integer} ztrátové skóre (hodnota ztrátové funkce)
        """
        output = self.activation(self.net_input(xi))
        error = (gt - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2

        return cost

    def net_input(self, X):
        """
        funkce pro výpočet lineárního vstupu

        :return: {1d-array} vektor lineárních vstupů trénovacích příkladů
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """
        funkce pro aplikaci lineární aktivační funkce na lineární vstup

        :return: {1d-array} vektor modifikovaných lineárních vstupů 
                            trénovacích příkladů (v tomto případě beze změny)
        """
        return X

    def predict(self, X):
        """
        funkce pro predikci tříd vstupních dat

        :return: {1d-array} vektor predikovaných tříd vstupních dat
        """
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

