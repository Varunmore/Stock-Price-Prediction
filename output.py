from re import X
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from matplotlib.pylab import rcParams
import nsepy
from datetime import date
import numpy as np
import os
from keras.models import load_model


comp = input("Please enter company symbol: ")
data = nsepy.get_history(symbol=comp, start=date(2005, 3, 1), end=date.today())
print("File Downloaded. Please wait for some time for output")
data.drop(["Symbol", "Series", "Prev Close", "VWAP", "Volume", "Deliverable Volume", "%Deliverble"], axis = 1, inplace=True)
data.to_csv(comp + '.csv')
df=pd.read_csv(comp + ".csv")
df["Date"]=pd.to_datetime(df.Date,format="%Y-%m-%d")
df.index=df['Date']
data=df.sort_index(ascending=True,axis=0)
new_dataset=pd.DataFrame(index=range(0,len(df)),columns=['Date','Close'])


for i in range(0,len(data)):
    new_dataset["Date"][i]=data['Date'][i]
    new_dataset["Close"][i]=data["Close"][i]
    

new_dataset.index=new_dataset.Date
new_dataset.drop("Date",axis=1,inplace=True)

final_dataset=new_dataset.values
# print(final_dataset)

arr = np.array([[192, 156, 123], [199, 150, 122], [111, 123, 122]])
model=load_model("saved_lstm_model.h5")
model.predict(arr)