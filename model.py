import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


data = pd.read_csv("cleanThy.csv")
data = data.drop('Date', axis=1)

print(data)



train_x = data.drop(' Highest Price', axis=1)
target_label = data[" Highest Price"]

x_train, x_test, y_train, y_test = train_test_split(train_x, target_label, test_size = 0.2)

linear = LinearRegression()
linear.fit(x_train,y_train)

pickle.dump(linear, open("linear.pkl", "wb"))

