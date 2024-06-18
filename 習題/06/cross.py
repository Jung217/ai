import keras
from nn import MLP
from engine import Value
import numpy as np
from keras.datasets import mnist
from sklearn.metrics import accuracy_score

# From https://github.com/newcodevelop/micrograd/blob/master/mnist.ipynb
(x_train,y_train),(x_test,y_test) = mnist.load_data()
train_images = np.asarray(x_train, dtype=np.float32) / 255.0
test_images = np.asarray(x_test, dtype=np.float32) / 255.0
train_images = train_images.reshape(60000,784)
test_images = test_images.reshape(10000,784)
y_train = keras.utils.to_categorical(y_train)

batch_size = 32
steps = 20000
Wb = Value(np.random.randn(784,10))# new initialized weights for gradient descent

for step in range(steps):
  ri = np.random.permutation(train_images.shape[0])[:batch_size]
  Xb, yb = Value(train_images[ri]), Value(y_train[ri])
  y_predW = Xb.matmul(Wb)
  probs = y_predW.softmax()
  log_probs = probs.log()

  zb = yb*log_probs

  outb = zb.reduce_sum(axis = 1)
  finb = -outb.reduce_sum()  #cross entropy loss
  finb.backward()
  if step%1000==0:
    loss = calculate_loss(train_images,y_train,Wb.data)
    print(f'loss in step {step} is {loss}')
  Wb.data = Wb.data- 0.01*Wb.grad
  Wb.grad = 0
loss = calculate_loss(train_images,y_train,Wb.data)
print(f'loss in final step {step+1} is {loss}')