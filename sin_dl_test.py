# -*- coding: utf-8 -*-

import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.core import Dense, Activation
from keras.optimizers import adam, rmsprop, adadelta
def generate_data(seq):
    X = []
    # 序列的第i项和后面的TIMESTEPS-1项合在一起作为输入；第i + TIMESTEPS项作为输
    # 出。即用sin函数前面的TIMESTEPS个点的信息，预测第i + TIMESTEPS个点的函数值。
    for i in range(len(seq)):
        X.append([seq[i]])

    return np.array(X, dtype=np.float32)

train_X=np.linspace(0,7,1000, dtype=np.float32)
train_y = np.sin(np.linspace(0,7,1000, dtype=np.float32))
#plt.plot(train_X,train_y)
#plt.show()
train_X=generate_data(train_X)
train_y=generate_data(train_y)
print(type(train_X))
model=Sequential()
model.add(Dense(500,input_shape=(1,)))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('sigmoid'))
model.add(Dense(1,activation='tanh'))

model.summary()

adamoptimizer = adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.00001)
model.compile(optimizer='adam', loss='mse',metrics=["accuracy"] )
history=model.fit(train_X,train_y,
                  batch_size=1000,epochs=10000,shuffle=True,verbose=1)

score=model.evaluate(train_X,train_y,verbose=1)
print('Test score:',score[0])
print('Test accuracy:',score[1])

predictY = model.predict(train_X, batch_size=1)
plt.plot(train_X,predictY)
plt.plot(train_X,train_y)
plt.show()