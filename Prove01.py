from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

class HardCodedClassifier:
    def fit(self, data, target):
        return HardCodedModel(data, target)

class HardCodedModel:
    def __init__(self, data, target):
        self.data = data
        self.target = target
    def predict(self, data):
        templist = []
        for i in data:
            templist.append(0)
        return templist

# Loading a dataset
iris = datasets.load_iris()

data = iris.data # X
target = iris.target # y

# Splitting the dataset into the Training set and Test set
data_train, data_test, target_train, target_test = train_test_split(data, target, train_size = 0.7, test_size = 0.3)

# Creating a model
classifier = GaussianNB()
model = classifier.fit(data_train, target_train)

# Making predictions
targets_predicted = model.predict(data_test)

# Caluating the accuracy from the sklearn
targetPred = classifier.predict(data_test)
print(round(accuracy_score(target_test, targetPred) * 100, 1), '%')

# A classifier
classifier = HardCodedClassifier()
model = classifier.fit(data_train, target_train)
targets_predicted = model.predict(data_test)

# Caluating the accuracy from the customer
score = 0
for i in range(len(targets_predicted)):
    if targets_predicted[i] == target_test[i]:
        score += 1
print(round(score / len(targets_predicted) * 100, 1), '%')
