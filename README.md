# neural_nets

import numpy as np
import pandas as pd
import tensorflow as tf

tf.__version__

dataset = pd.read_csv('BankNote_Authentication.csv')
dataset


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from keras.models import Sequential
from keras.layers import Dense

# Custom activation function
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects


def func(x, k0=2, k1=2):
    return x * k0 + k1


from keras.utils.generic_utils import get_custom_objects

get_custom_objects().update({'func_act': Activation(func)})


ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='func_act'))
ann.add(tf.keras.layers.Dense(units=6, activation='func_act'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


history = ann.fit(X_train, y_train,validation_split = 0.1, epochs=50, batch_size=4)


y_pred = ann.predict(X_test)
y_pred


y_pred = (y_pred > 0.5)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
    
    
