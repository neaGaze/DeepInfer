import numpy
from numpy import loadtxt, math
from keras.models import Sequential
from keras.layers import Dense
import math
from tensorflow import keras
import pandas as pd
import numpy as np
# load the dataset
from scipy.stats import nbinom

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras import regularizers
from keras import optimizers
from keras.layers import Dense, Dropout, Activation
import inferDataPrecondition

df = loadtxt('Dataset/mobile_classification_train.csv', delimiter=',', skiprows=1)
X = df[:,0:20]
y = df[:,20]
print(X.shape)
print(y.shape)

#train-test split
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=10)

model = Sequential()
model.add(Dense(12, input_dim=20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10, verbose=0, validation_split=0.2)
# make class predictions with the model
# access validation accuracy for each epoch
# acc = model.history.history['accuracy']
# print(acc)

# COMMENT THE NEXT LINE IF YOU ALREADY HAVE SAVED THE FOLLOWING FILE
model.save('MD1.h5')
score = model.evaluate(X, y, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

predictions = (model.predict(X) > 0.5).astype(int)
# summarize the first 5 cases
for i in range(5):
    #X[i]
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
