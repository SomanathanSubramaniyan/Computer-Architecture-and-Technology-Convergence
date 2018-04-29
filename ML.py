
#http://docs.python-guide.org/en/latest/scenarios/ml/
#http://scikit-learn.org/stable/user_guide.html
#https://www.python-course.eu/machine_learning_with_scikit.php
#http://scikit-learn.org/stable/tutorial/basic/tutorial.html
#http://www.pythonforbeginners.com/os/pythons-os-module
import os
#https://docs.python.org/3/library/pickle.html
import pickle
#http://scikit-learn.org/stable/tutorial/basic/tutorial.html#machine-learning-the-problem-setting
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.metrics import accuracy_score
import numpy as np
#http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
from sklearn.tree import DecisionTreeRegressor
#loading the iris dataset
iris = load_iris()
x = iris.data #array of the data
#print("Full Record in array",x)
y = iris.target #array of labels of each data entry as flower name
#print("labels of each data entry",y)
#getting label names the three flower species
y_names = iris.target_names
#print("Flower name",y_names)
#taking random indices to split the dataset into train and test
test_ids = np.random.permutation(len(x))
#test_ids=np.permutation(len(x))
#print(test_ids)
#splitting data and labels into train and test
#keeping last 10 entries for testing, rest for training
TrainData = x[test_ids[:-60]]
#print(TrainData)
#InputTrainData = x[test_ids[-10:]]
#print(InputTrainData)
TrainFlower = y[test_ids[:-60]]
#print(TrainFlower)
y_test = y[test_ids[-60:]]
#print(y_test)
#print(y_test)
#classifying using decision tree
clf = tree.DecisionTreeClassifier()
#training the classifier with the training set
clf.fit(TrainData, TrainFlower)
InputTrainData = x[test_ids[-60:]]
#InputTrainData = [[sepallength,sepalWidth,Petallength,PetallWidth]]
#predictions for test dataset
pred = clf.predict(InputTrainData)
#print ("pred", pred )#predicted labels i.e flower species
accuracy= (float(accuracy_score(pred, y_test))*100) #prediction accuracy
print("prediction percentage",accuracy)
# using DecisionTreeRegressor mim and max value


