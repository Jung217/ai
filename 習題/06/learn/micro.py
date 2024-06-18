import keras
import numpy as np
from keras.datasets import mnist
from sklearn.metrics import accuracy_score
from engine import Value
from nn import MLP

# From https://github.com/newcodevelop/micrograd/blob/master/mnist.ipynb
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.asarray(x_train, dtype = np.float32) / 255.0
x_test = np.asarray(x_test, dtype = np.float32) / 255.0
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
y_train = keras.utils.to_categorical(y_train)

epochs = 6
batch_size = 32
learning_rate = 0.3
model = MLP(784, [128, 10])

for epoch in range(1, epochs + 1):
    pre = 0
    cur = batch_size
    index = np.random.permutation(x_train.shape[0])

    while pre < cur:
        ri = index[pre:cur]
        Xb, yb = Value(x_train[ri]), Value(y_train[ri])

        y_predW = model(Xb)
        probs = y_predW.softmax()
        loss = probs.cross_entropy(yb)

        model.zero_grad()
        loss.backward()

        for p in model.parameters():
            if p.data.ndim > 1:
                p.data -= learning_rate * p.grad
            else:
                p.data -= learning_rate * p.grad.sum(axis = 0)

        pre = cur

        if cur <= x_train.shape[0] - batch_size:
            cur += batch_size
        else:
            cur = x_train.shape[0]

    print(f'epoch {epoch}, accuracy: {(accuracy_score(np.argmax(model(x_test).data, axis = 1), y_test) * 100):.2f}%')