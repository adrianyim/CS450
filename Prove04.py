import pandas as pd
import numpy as np
import graphviz
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from random import randint as rand
from sklearn import tree
from sklearn.metrics import accuracy_score
import random
from sklearn.preprocessing import LabelEncoder

#-- Iris Decision Tree function --
def getIrisTree():
    iris = load_iris()

    # Splitting the dataset into the Training set and Test set
    data_train, data_test, target_train, target_test = train_test_split(iris.data, iris.target, test_size = 0.3)

    # Calling the decision tree classifier from the sklearn
    skTree = tree.DecisionTreeClassifier(max_depth=4)
    skTree = skTree.fit(data_train, target_train)

    # Exporting a tree by using graphviz
    dot_data = tree.export_graphviz(skTree, out_file=None, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True, special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("iris")

    # Predicting and testing the accuracy, and then print it out
    targets_predicted = skTree.predict(data_test)
    print("The iris accuracy is", round(accuracy_score(target_test, targets_predicted)*100, 1), "%")

#-- Lenses Decision Tree function --
def getLensesTree():
    lenses = pd.read_fwf("lenses.txt", sep=" ", header=None)
    lenses.columns = ["patient-num", "age", "perscription", "astilmatic", "tear-rate", "classif"]

    # Assigning the dataset into data and target
    target = lenses.iloc[:,5].values
    data = lenses.iloc[:,1:].values

    # Splitting the dataset into the Training set and Test set
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size = 0.3)

    # Calling the decision tree classifier from the sklearn
    skTree = tree.DecisionTreeClassifier(max_depth=4)
    skTree = skTree.fit(data_train, target_train)

    # Exporting a tree by using graphviz
    dot_data = tree.export_graphviz(skTree, out_file=None, filled=True, rounded=True, special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("lenses")
    
    # Predicting and testing the accuracy, and then print it out
    targets_predicted = skTree.predict(data_test)
    print("The lenses accuracy is", round(accuracy_score(target_test, targets_predicted)*100, 1), "%")

#-- Vote Decision Tree function --
def getVoteTree():
    vote = pd.read_csv("vote.csv", sep=",", header=None)
    #vote.columns = ["Class Name", "Handicapped-infants", "Water-project-cost-sharing", "Adoption-of-the-budget-resolution", "Physician-fee-freeze", "El-salvador-aid", "Religious-groups-in-schools", "Anti-satellite-test-ban", "Aid-to-nicaraguan-contras", "Mx-missile", "immigration", "Synfuels-corporation-cutback", "Education-spending", "Superfund-right-to-sue", "Crime", "Duty-free-exports", "Export-administration-act-south-africa"]

    # Assiging a missing data in "y" and "n" randomly
    vote = vote.replace("?", np.random.choice(["y","n"]))

    # Assigning the dataset into data and target
    data = vote.iloc[:,1:].values
    target = vote.iloc[:,0].values

    # Convert data into a 1-D array
    target = target.ravel()

    # Normalizing class data with a labelEncoder()
    LEncoder = LabelEncoder()
    target = LEncoder.fit_transform(target)

    # Normalizing attributes using same process
    for i in range(data.shape[1]):
        data[:,i] = LEncoder.fit_transform(data[:,i])

    # Splitting the dataset into the Training set and Test set
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size = 0.3)

    # Calling the decision tree classifier from the sklearn
    skTree = tree.DecisionTreeClassifier(max_depth=4)
    skTree = skTree.fit(data_train, target_train)

    # Exporting a tree by using graphviz
    dot_data = tree.export_graphviz(skTree, out_file=None, filled=True, rounded=True, special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("vote")

    # Predicting and testing the accuracy, and then print it out
    targets_predicted = skTree.predict(data_test)
    print("The voteing accuracy is", round(accuracy_score(target_test, targets_predicted)*100, 1), "%")
               
getIrisTree()
getLensesTree()
getVoteTree()