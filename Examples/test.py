import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Importing the dataset
# dataset = pd.read_csv('../CS450/iris.csv')
iris = datasets.load_iris()

# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 0, shuffle = True)

print(iris)