#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

names = ['sepal-length', 'sepal-widht', 'petal-length', 'petal-width', 'Class']

irisdata = pd.read_csv(url, names=names)


#spliting dataset for atrybutes and labels

x = irisdata.iloc[:, 0:4]
y = irisdata.select_dtypes(include=[object])

y.Class.unique()

#encoding labels to numbers

labelencoder = preprocessing.LabelEncoder()

y = y.apply(labelencoder.fit_transform)

y.Class.unique()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

#feature scaling x_set(values)

scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#training 

mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
mlp.fit(x_train, y_train.values.ravel())

#prediction

predictions = mlp.predict(x_test)

#evaluating algorithm
#cofusion matrix
print(confusion_matrix(y_test, predictions))
#classification report
print(classification_report(y_test,predictions))



# In[ ]:




