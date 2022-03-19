from neuralnet import NeuralNetMLP
import numpy as np
import matplotlib.pyplot as plt

nn = NeuralNetMLP(n_hidden=100,
                   l2=0.01,
                   epochs=2,
                   eta=0.0005,
                   minibatch_size=100,
                   shuffle=True, seed=1)

dataset = np.load('dataset/mnist_scaled.npz')
X_train, y_train, X_test, y_test = [dataset[f] for f in ['X_train', 'y_train', 'X_test', 'y_test']]

nn.fit(X_train=X_train[:55000], 
       y_train=y_train[:55000],
       X_valid=X_train[55000:],
       y_valid=y_train[55000:])


def visualize_cost(save):
    plt.plot(range(nn.epochs), nn.eval_['cost'])
    plt.ylabel('Cost')
    plt.xlabel('Epochs')
    if save:
        plt.savefig('cost.png', dpi=300)
    plt.show()


def visualize_acc(save):
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
    predictions = nn.predict(X)
    n_examples = X.shape[0]
    ra = 0
    problem_arrays = {}

    for i in range(n_examples):
        if predictions[i] == y[i]:
            ra += 1
        else:
            problem_arrays[i] = X[i]

    accuracy = ra / n_examples * 100
    return predictions, accuracy, problem_arrays


predicitons, acc, problem_arr = test_acc(X_test, y_test)
print(f'\nAccuracy: {round(acc, 2)} %')

for key in problem_arr:
    print(y_test[key])
    plt.imshow(problem_arr[key].reshape(28, 28), cmap='Greys')
    plt.show()
