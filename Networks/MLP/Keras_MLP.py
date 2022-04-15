from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# konfigurace
feature_vector_length = 784
num_classes = 10

# načtení MNIST databáze
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# převod dat z matic 28x28 na vektor 784x1
X_train = X_train.reshape(X_train.shape[0], feature_vector_length)
X_test = X_test.reshape(X_test.shape[0], feature_vector_length)

# převod dat na stupnici šedi
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# zakódování cílových tříd podle one-hot
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# nastavení tvaru vstupu
input_shape = (feature_vector_length,)

# poskládání modelu
model = Sequential()
model.add(Dense(150, input_shape=input_shape, activation='sigmoid'))
model.add(Dense(num_classes, activation='softmax'))

# konfigurace modelu a zahájení trénování
model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=300, batch_size=100, verbose=1, validation_split=0.2)

# otestování natrénovaného modelu
test_results = model.evaluate(X_test, y_test, verbose=1)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')
