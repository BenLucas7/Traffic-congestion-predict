import gc
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
from sklearn.preprocessing import StandardScaler


import keras
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout

with open('lstm_x.json','r') as f:
    x = np.array(json.load(f))

with open('lstm_y.json','r') as f:
    y = np.array(json.load(f))

scaler = StandardScaler()
x = scaler.fit_transform(x)

print (x)

x_train = x[:-10]
x_test = x[-10:]
y_train = y[:-10][:,:1]
y_test = y[-10:][:,:1]

print (x_train.shape)
print (x_test.shape)
print (y_train.shape)
print (y_test.shape)

# model = Sequential()
#
# model.add(LSTM(units = 64, input_shape = (100,3), return_sequences = True))
# model.add(LSTM( 64, return_sequences=False)) # SET HERE
# # regressor.add(Dropout(0.2))
# model.add(Dense(1))
# model.compile(optimizer = 'adam', loss = 'mean_squared_error')
# model.summary()
#
# model.fit(x_train, y_train, epochs = 1000, batch_size = 32)
#
# with open('lstm.pickle','wb') as f:
#     pickle.dump(model,f)

with open('lstm.pickle','rb') as f:
    model = pickle.load(f)
res=[]
for i in x_test:
    vec = np.array([i.tolist()])
    speed = model.predict(vec)
    res.append(speed.tolist()[0])

x = list(range(len(y_test)))
y_true = y_test.ravel().tolist()

fig = plt.figure()
plt.plot(x,y_true,'r')
plt.plot(x,res,'b')
plt.show()
# model = Sequential()
# model.add(LSTM(1,input_shape=(100,3),return_sequences = True))
# model.add(Dense(8, input_dim=3, activation= 'linear' ))
# model.compile(loss='mse',optimizer='rmsprop',metrics=['accuracy'])
# model.summary()
# model.fit(x_train,y_train,epochs=100 ,batch_size=5,validation_split=0.05,verbose=0);
# scores2 = model2.evaluate(y_train,y_test,verbose=1,batch_size=5)
# print('Accurracy: {}'.format(scores2[1]))
