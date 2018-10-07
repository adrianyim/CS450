from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from operator import itemgetter

#-- Manhattan function --
def manhattan(m1, m2):
    total = 0
    for i in range(len(m2)):
        if (not isinstance(m1[i], str) and not isinstance(m2[i], str)):
            total += abs(m2[i] - m1[i])

    return total

#-- storedList class --
class storedList:
    def __init__(self, listData, target):
        self.listData = listData
        self.target = target

#-- HardCodedClassifier class --
class HardCodedClassifier:
    def __init__(self, k):
        self.k = k
    def fit(self, data, target):
        matchedList = [] # Create a list

        # Loading all the data into matchList
        for i in range(len(data)):
            matchedList.append(storedList(data[i], target[i]))

        return HardCodedModel(matchedList, self.k)

#-- HardCodedModel class --
class HardCodedModel:
    def __init__(self, matchedList, k):
        self.matchedList = matchedList
        self.k = k
    def predict(self, tempData):
        templist1 = [] # Create a templist1

        for i in tempData:
            templist2 = [] # Create a templist2

            for j in self.matchedList:
                templist2.append((manhattan(i, j.listData), j.target))
            
            # Sorting function
            sortedList = sorted(templist2, key = itemgetter(0))[0:self.k]

            # Push the data into templist1
            templist1.append((max(set(sortedList), key = sortedList.count)[1]))

        # Return the result
        return templist1

#-- ComparedCustom function --
def comparedCustom(data, target):
    # Splitting the dataset into the Training set and Test set
    data_train, data_test, target_train, target_test = train_test_split(data, target, train_size = 0.7, test_size = 0.3)

    # Own algorithm classifier
    classifier = HardCodedClassifier(3)
    model = classifier.fit(data_train, target_train)
    targets_predicted = model.predict(data_test)

    # Print the result
    display(targets_predicted, target_test)

#-- ComparedOriginal function --
def comparedOriginal(data, target):
    # Splitting the dataset into the Training set and Test set
    data_train, data_test, target_train, target_test = train_test_split(data, target, train_size = 0.7, test_size = 0.3)

    # Sklearn algorithm classifier
    classifier = KNeighborsClassifier(n_neighbors = 3)
    model = classifier.fit(data_train, target_train)
    targets_predicted = model.predict(data_test)

    # Print the result
    display(targets_predicted, target_test)

#-- Display function --
def display(targets_predicted, target_test):
    # Caluating the accuracy from the customer
    score = 0

    for i in range(len(targets_predicted)):
        if targets_predicted[i] == target_test[i]:
            score += 1

    # Print the result
    print(round(score / len(targets_predicted) * 100, 1), '%')

# Loading a dataset
iris = datasets.load_iris()

# Customer
print('Custom:')
comparedCustom(iris.data, iris.target)

# Original
print('Original:')
comparedOriginal(iris.data, iris.target)