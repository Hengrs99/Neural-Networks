from MLP import MLP_NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt

# inicializace MLP
nn = MLP_NeuralNetwork(n_hidden=150,
                   l2=0.01,
                   epochs=300,
                   eta=0.0005,
                   minibatch_size=100,
                   shuffle=True, seed=1)

# načtení datového souboru MNIST
dataset = np.load('dataset/mnist_scaled.npz')
X_train, y_train, X_test, y_test = [dataset[f] for f in ['X_train', 'y_train', 'X_test', 'y_test']] 

# trénování sítě
nn.fit(X_train=X_train[:55000], 
       y_train=y_train[:55000],
       X_valid=X_train[55000:],
       y_valid=y_train[55000:])


def visualize_cost(save):
    """
    funkce pro vizualizaci ztrátové funkce během učení
    :param save: {bool} True = uloží vizualizaci
    :return: None
    """
    plt.plot(range(nn.epochs), nn.eval_['cost'])
    plt.ylabel('Cost')
    plt.xlabel('Epochs')
    if save:
        plt.savefig('cost.png', dpi=300)
    plt.show()


def visualize_acc(save):
    """
    funkce pro vizualizaci přesnosti během učení
    :param save: {bool} True = uloží vizualizaci
    :return: None
    """
    plt.plot(range(nn.epochs), nn.eval_['train_acc'], label='training')
    plt.plot(range(nn.epochs), nn.eval_['valid_acc'], label='validation', linestyle='--')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    if save:
        plt.savefig('accuracy.png', dpi=300)
    plt.show()


visualize_cost(save=True)
visualize_acc(save=True)


def test_acc(X, y):
    """
    funkce pro určení přesnosti natrénované sítě

    :param X: {2d-array} matice příznaků příkladů k predikci
    :param y: {1d-array} vektor cílových tříd

    :return predictions: {1d-array} vekotr predikovaných tříd
    :return accuracy: {float} procentuální přesnost sítě
    :return problem_arrays: {dictionary} slovník špatně predikovaných vektorů
    :return classified_arrays: {dictionary} slovník správně predikovaných vektorů
    """
    predictions = nn.predict(X)
    n_examples = X.shape[0]
    ra = 0
    problem_arrays = {}
    classified_arrays = {}

    for i in range(n_examples):
        if predictions[i] == y[i]:
            classified_arrays[i] =X[i]
            ra += 1
        else:
            problem_arrays[i] = X[i]

    accuracy = ra / n_examples * 100
    return predictions, accuracy, problem_arrays, classified_arrays


def show_problems(problem_array):
    """
    funkce na zobrazení špatně určených obrázků
    :param problem_array: {dictionary} slovník problémových vektorů
    :return: None
    """
    for key in problem_array:
        print(y_test[key])
        plt.imshow(problem_array[key].reshape(28, 28), cmap='Greys')
        plt.show()


def show_classified(classified_array):
    """
    funkce na zobrazení správně určených obrázků
    :param classified_array: {dictionary} slovník správně přečtených vektorů
    :return: None
    """
    for key in classified_array:
        print(y_test[key])
        plt.imshow(classified_array[key].reshape(28, 28), cmap='Greys')
        plt.show()


predicitons, acc, problem_arr, classified_arr = test_acc(X_test, y_test)
print(f'\nAccuracy: {round(acc, 2)} %')

# show_classified(classified_arr)
# show_problems(problem_arr)
