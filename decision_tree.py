import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def train_using_gini(x_train, x_test, y_train):

	clf_gini = DecisionTreeClassifier(criterion = "gini",
            random_state = 100,max_depth=3, min_samples_leaf=5, class_weight = 'balanced')

	clf_gini.fit(x_train, y_train)
	return clf_gini

def train_using_entropy(x_train, x_test, y_train):

	clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
            max_depth = 3, min_samples_leaf = 5, class_weight = 'balanced')

	clf_entropy.fit(x_train, y_train)
	return clf_entropy

def prediction(x_test, clf_object):

	y_pred = clf_object.predict(x_test)
	print("predicted values: ")
	print(y_pred)
	return y_pred

def accuracy(y_test, y_pred):

    print("Confusion Matrix: ",
        confusion_matrix(y_test, y_pred))
      
    print ("Accuracy : ",
    accuracy_score(y_test,y_pred))

    print("Recall:",metrics.recall_score(y_test, y_pred))

    print("Precision:",metrics.precision_score(y_test, y_pred))
    
    print()

def main():
	data=pd.read_csv('train.csv')

	data.dropna(inplace = True)

	y_data = data['TenYearCHD']
	x_data = data.drop('TenYearCHD', axis = 1)

	sex_data = pd.get_dummies(x_data['sex'], drop_first = True)
	sex_data.rename(columns={'M':'is_male'}, inplace=True)

	smoking_data = pd.get_dummies(x_data['is_smoking'], drop_first = True)
	smoking_data.rename(columns={'YES':'is_smoking'}, inplace=True)

	x_data.drop(['sex', 'is_smoking'], axis = 1, inplace = True)
	x_data = pd.concat([x_data, sex_data, smoking_data], axis = 1)

	# x_dv = x_data.values
	# y_dv = y_data.values
	# x_dv = np.float32(x_dv)
	# y_dv = np.int32(y_dv)
	# print(x_dv.dtype)
	# print(y_dv.dtype)
	# x = [d for d in x_data]
	# y = [d for d in y_data]

	x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.3)
	clf_gini = train_using_gini(x_train, x_test, y_train)
	clf_entropy = train_using_entropy(x_train, x_test, y_train)

	print("Results using Gini Index:")
	y_pred_gini = prediction(x_test, clf_gini)
	accuracy(y_test, y_pred_gini)

	print("Results using Entropy:")
	y_pred_entropy = prediction(x_test, clf_entropy)
	accuracy(y_test, y_pred_entropy)

if __name__ == '__main__':
	main()