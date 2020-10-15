import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

#-->LOAD DATA<--#
data = pd.read_csv('coronovirus.csv',sep=',')
data = data[['id','cases']]
print("-"*20);print("head");print("-"*20)
print(data.head())

#--> PREPARE DATA <--#
print("-"*20);print("prepare data");print("-"*20)
x = np.array(data['id']).reshape(-1,1)
y = np.array(data['cases']).reshape(-1,1)
plt.plot(y,'-m')
# plt.show()
polyFeat = PolynomialFeatures(degree=3)
x = polyFeat.fit_transform(x)
# print(x)

#--> TRAINING DATA <--#
print("-"*20);print("TRAINING data");print("-"*20)
model = linear_model.LinearRegression()
model.fit(x,y)
accuracy = model.score(x,y)
print(f'Accuracy:{round(accuracy*100,3)}%')
y0 = model.predict(x)
# plt.plot(y0,'--b')
# plt.show()


#--> PREDICTION<--#
days = 30 #iam predicting for after 30 days
print("-"*20);print("Predictions");print("-"*20)
print(f'Prediction - Cases after {days} days:',end='')
print(round(int(model.predict(polyFeat.fit_transform([[287+days]])))/1000000,2),'Million')

x1 = np.array(list(range(1,289+days))).reshape(-1,1)
y1 = model.predict(polyFeat.fit_transform(x1))
plt.plot(y1,'--r')
plt.plot(y0,'--b')
plt.show()
