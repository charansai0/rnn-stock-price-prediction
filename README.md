# Stock Price Prediction

## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset
Make a Recurrent Neural Network model for predicting stock price using a training and testing dataset. The model will learn from the training dataset sample and then the testing data is pushed for testing the model accuracy for checking its accuracy. The Dataset has features of market open price, closing price, high and low price for each day.
## Neural Network Model
![WhatsApp Image 2022-10-13 at 18 11 56](https://github.com/charansai0/rnn-stock-price-prediction/assets/94296221/8cac7b95-6ec7-482f-bf96-acb12f87a12e)


## DESIGN STEPS

### STEP 1:
import the neccesary tensorflow modules

### STEP 2:
load the stock dataset

### STEP 3:
fit the model and then predict

## PROGRAM
```
 Developed by: v.charan sai
 Reg No: 212221240061
```
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from keras.models import Sequential

dataset_train = pd.read_csv('trainset.csv')

dataset_train.columns

dataset_train.head()

train_set = dataset_train.iloc[:,1:2].values

type(train_set)

train_set.shape

sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(train_set)

training_set_scaled.shape

X_train_array = []
y_train_array = []
for i in range(60, 1259):
  X_train_array.append(training_set_scaled[i-60:i,0])
  y_train_array.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train_array), np.array(y_train_array)
X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1],1))

X_train.shape
length = 60
n_features = 1

model = Sequential()
model.add(layers.SimpleRNN(50,input_shape=(length,n_features)))
model.add(layers.Dense(1))
model.compile(optimizer='adam', loss='mse')

model.summary()

model.fit(X_train1,y_train,epochs=100, batch_size=32)

dataset_test = pd.read_csv('testset.csv')

test_set = dataset_test.iloc[:,1:2].values

test_set.shape

dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)

inputs = dataset_total.values
inputs = inputs.reshape(-1,1)
inputs_scaled=sc.transform(inputs)
X_test = []
for i in range(60,1384):
  X_test.append(inputs_scaled[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))

X_test.shape

predicted_stock_price_scaled = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price_scaled)

plt.plot(np.arange(0,1384),inputs, color='red', label = 'Test(Real) Google stock price')
plt.plot(np.arange(60,1384),predicted_stock_price, color='blue', label = 'Predicted Google stock price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
```
### OUTPUT

### True Stock Price, Predicted Stock Price vs time
<img width="413" alt="Screenshot 2023-09-27 085331" src="https://github.com/charansai0/rnn-stock-price-prediction/assets/94296221/b1d211a6-8ade-4964-8d90-3af607473f5f">



### Mean Square Error
<img width="399" alt="Screenshot 2023-09-27 085418" src="https://github.com/charansai0/rnn-stock-price-prediction/assets/94296221/1ca7c903-bd50-407a-a8a2-0f494eba8938">



## RESULT
Thus a Recurrent Neural Network model for stock price prediction is created and executed successfully.
