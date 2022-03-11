import struct
import numpy as np
import matplotlib.pyplot as plt


def load_mnist(kind):
    labels_path = f'{kind}-labels-idx1-ubyte'
    images_path = f'{kind}-images-idx3-ubyte'
        
    with open(labels_path, 'rb') as lbfile:
        magic, n = struct.unpack('>II', lbfile.read(8))
        labels = np.fromfile(lbfile, dtype=np.uint8)

    with open(images_path, 'rb') as imgfile:
        magic, num, rows, cols = struct.unpack(">IIII", imgfile.read(16))
        images = np.fromfile(imgfile, dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255.0) - 0.5) * 2
 
    return images, labels


def visualize_img(img_arr):
    image = img_arr.reshape(28, 28)
    plt.imshow(image, cmap='Greys')
    plt.show()


X_train, y_train = load_mnist('train')
# print(f'Rows: {X_train.shape[0]}, columns: {X_train.shape[1]}')

X_test, y_test = load_mnist('t10k')
# print(f'Rows: {X_test.shape[0]}, columns: {X_test.shape[1]}')

np.savez_compressed('mnist_scaled.npz', 
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test)

# mnist = np.load('mnist_scaled.npz')
# X_train, y_train, X_test, y_test = [mnist[f] for f in ['X_train', 'y_train', 'X_test', 'y_test']]
