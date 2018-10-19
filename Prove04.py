import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from random import randint as rand
from sklearn import tree
import graphviz

# Iris Decision Tree function
def getIrisTree():
    iris = load_iris()

    # Splitting the dataset into the Training set and Test set
    data_train, data_test, target_train, target_test = train_test_split(iris.data, iris.target, train_size = 0.7, test_size = 0.3)

    skTree = tree.DecisionTreeClassifier()
    skTree = skTree.fit(data_train, target_train)

    dot_data = tree.export_graphviz(skTree, out_file = None, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True, special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("iris")
    print(graph)

# Lenses Decision Tree function
def getLensesTree():
    lenses = np.array(pd.read_csv("lenses.csv", header=None))
    data = lenses[:,0:-1]
    target = lenses[:,-1]
    print(lenses)


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

#getIrisTree()
getLensesTree()