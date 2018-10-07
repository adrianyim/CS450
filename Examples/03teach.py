import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

def handleMissingData(data):
    for i in range(len(data[0])):
        maxValue = max(data[:,i])
        for j in range(len(data)):
            if data[j][i] == "?,":
                data[j][i] = maxValue

def quantifyData(data):
    stringMap = {}
    it = 0
    for (x, y), value in np.ndenumerate(data):
        if not value.isdigit():
            if not value in stringMap:
                stringMap[data[x][y]] = it
                it += 1
            data[x][y] = stringMap[value]

data = np.array(pd.read_csv("census-data.txt", sep=" ", header=None))

target = data[:,-1]
data = data[:,0:-1]

handleMissingData(data)
quantifyData(data)

data_train, data_test, targets_train, targets_test = train_test_split(data, target, test_size = 0.3)

classifier = KNeighborsClassifier(n_neighbors = 3)
kf = KFold(n_splits = 10)

ac = []

for train_index, test_index in kf.split(data):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = target[train_index], target[test_index]
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    ac.append(accuracy_score(y_test, y_pred))

for i in range(len(ac)):
    print(str(i + 1) + ")", str(round(ac[i] * 100, 1)) + "%")