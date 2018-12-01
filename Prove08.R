#install.packages('e1071', dependencies = TRUE);
#setwd('C:/CS450')
# Include the LIBSVM package
#library(e1071)

# Load vowel.csv
dataset <- read.csv("vowel.csv", head = TRUE, sep=",")

# Partition the data into training and test sets
index <- 1:nrow(dataset)

testIndex <- sample(index, trunc(length(index) * 0.3))

# The test set contains all the test rows
testset <- dataset[testIndex,]

# The training set contains all the other rows
trainset <- dataset[-testIndex,]

# Train an SVM model
model <- svm(Sex~., data = trainset, kernel = "radial", gamma = 0.001, cost = 10)
summary(model)

# Use the model to make a prediction on the test set
prediction <- predict(model, testset[,-2])

# Produce a confusion matrix
confusionMatrix <- table(pred = prediction, true = testset[,2])

# Calculate the accuracy, by checking the cases that the targets agreed
agreement <- prediction == testset$Sex
accuracy <- prop.table(table(agreement))

# Print our results to the screen
print(accuracy)

# svm(Sex~., data = trainset, kernel = "radial", gamma = 0.001, cost = 0.5) >> 87.8%
# svm(Sex~., data = trainset, kernel = "radial", gamma = 0.001, cost = 1) >> 87.8%
# svm(Sex~., data = trainset, kernel = "radial", gamma = 0.001, cost = 10) >> 96.9%

# svm(Sex~., data = trainset, kernel = "radial", gamma = 0.0002, cost = 0.5) >> 53.1%
# svm(Sex~., data = trainset, kernel = "radial", gamma = 0.0002, cost = 1) >> 65.6%
# svm(Sex~., data = trainset, kernel = "radial", gamma = 0.0002, cost = 10) >> 92.5%

# svm(Sex~., data = trainset, kernel = "radial", gamma = 0.005, cost = 0.5) >> 89.2%
# svm(Sex~., data = trainset, kernel = "radial", gamma = 0.005, cost = 1) >> 91.2%
# svm(Sex~., data = trainset, kernel = "radial", gamma = 0.005, cost = 10) >> 100%

# svm(Sex~., data = trainset, kernel = "radial", gamma = 0.0006, cost = 0.5) >> 76.7%
# svm(Sex~., data = trainset, kernel = "radial", gamma = 0.0006, cost = 1) >> 87.2%
# svm(Sex~., data = trainset, kernel = "radial", gamma = 0.0006, cost = 10) >> 97.3%

# ----------------------------------------------------------------------------------------

# Load letters.csv
dataset <- read.csv("letters.csv", head = TRUE, sep=",", nrows=2500)

# Partition the data into training and test sets
index <- 1:nrow(dataset)

testIndex <- sample(index, trunc(length(index) * 0.3))

# The test set contains all the test rows
testset <- dataset[testIndex,]

# The training set contains all the other rows
trainset <- dataset[-testIndex,]

# Train an SVM model
model <- svm(letter~., data = trainset, kernel = "radial", gamma = 0.5, cost = 0.5)
summary(model)

# Use the model to make a prediction on the test set
prediction <- predict(model, testset[-1])

# Produce a confusion matrix
confusionMatrix <- table(pred = prediction, true = testset[,1])

# Calculate the accuracy, by checking the cases that the targets agreed
agreement <- prediction == testset$letter
accuracy <- prop.table(table(agreement))

# Print our results to the screen
print(accuracy)

# Only 2500 rows
# svm(letter~., data = trainset, kernel = "radial", gamma = 0.001, cost = 0.5) >> 16.1%
# svm(letter~., data = trainset, kernel = "radial", gamma = 0.001, cost = 1) >> 25%
# svm(letter~., data = trainset, kernel = "radial", gamma = 0.001, cost = 10) >> 67.4%

# svm(letter~., data = trainset, kernel = "radial", gamma = 0.0002, cost = 0.5) >> 2.5%
# svm(letter~., data = trainset, kernel = "radial", gamma = 0.0002, cost = 1) >> 3.6%
# svm(letter~., data = trainset, kernel = "radial", gamma = 0.0002, cost = 10) >> 42.9%

# svm(letter~., data = trainset, kernel = "radial", gamma = 0.005, cost = 0.5) >> 44.8%
# svm(letter~., data = trainset, kernel = "radial", gamma = 0.005, cost = 1) >> 61.3%
# svm(letter~., data = trainset, kernel = "radial", gamma = 0.005, cost = 10) >> 79.3%

# svm(letter~., data = trainset, kernel = "radial", gamma = 0.5, cost = 0.5) >> 66%
# svm(letter~., data = trainset, kernel = "radial", gamma = 0.5, cost = 1) >> 82.1%
# svm(letter~., data = trainset, kernel = "radial", gamma = 0.5, cost = 10) >> 86.1%

