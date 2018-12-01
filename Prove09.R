#install.packages('arules');
#library(arules);

# Loading a dataset from the library
data(Groceries)

# Putting Groceries dataset into an apriori function
rules <- apriori(Groceries, parameter = list(supp = 0.001, conf = 0.08))
summary(rules)


