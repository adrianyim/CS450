import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from random import randint as rand

def handleNonNum(data):
    stringMap = {}
    it = 0
    for (x, y), value in np.ndenumerate(data):
        if not value.isdigit():
            if not value in stringMap:
                stringMap[data[x][y]] = it
                it += 1
            data[x][y] = stringMap[value]

def handleMissingData(data):
    for (x, y), value in np.ndenumerate(data):
       if value == "?" or value == "NA":
           while data[x][y] == "?" or data[x][y] == "NA":
               data[x][y] = data[rand(0, len(data) - 1)][y]

carData = np.array(pd.read_csv("car.csv", sep=",", header=None))

data = carData[:,0:-1]
target = carData[:,-1]

handleMissingData(data)
handleNonNum(data)

print("\n")
print("Car dataset:")
print(carData)

data_train, data_test, targets_train, targets_test = train_test_split(data, target, test_size = 0.3)

clf = KNeighborsClassifier(n_neighbors = 3)
kf = KFold(n_splits = 10)

for train_index, test_index in kf.split(data):
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = target[train_index], target[test_index]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(str(round(accuracy_score(y_test, y_pred) * 100, 1)) + "% correct")

# aadData = np.array(pd.read_csv("autism_adult_data.csv", sep=", ", header=None))

# data = aadData[:,0:-1]
# target = aadData[:,-1]

# handleMissingData(data)
# print(aadData)
# handleNonNum(data)

# print("\n")
# print("Auto-mpg dataset:")
# print(mpgData)

# data_train, data_test, targets_train, targets_test = train_test_split(data, target, test_size = 0.3)

# clf = KNeighborsClassifier(n_neighbors = 3)
# kf = KFold(n_splits = 10)

# for train_index, test_index in kf.split(data):
#     X_train, X_test = data[train_index], data[test_index]
#     y_train, y_test = target[train_index], target[test_index]
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     print(str(round(accuracy_score(y_test, y_pred) * 100, 1)) + "% correct")