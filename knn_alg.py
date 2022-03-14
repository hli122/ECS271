import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LNG
from sklearn.model_selection import train_test_split
from collections import Counter

def kNN(x_train, y_train, x_test, y_test, k):

	correct_count = np.zeros(10)
	tests_count = np.zeros(10)

	for test_x, test_y in zip(x_test, y_test):
		distances = []

		for train_x, train_y in zip(x_train, y_train):
			eucli_dist = LNG.norm(train_x - test_x)
			distances.append([eucli_dist, train_y])

		distances = sorted(distances)
		knn_list = [i[1] for i in distances[:k]]
		predict_result = Counter(knn_list).most_common(1)[0][0]

		if predict_result == test_y:
			correct_count[test_y] += 1
		tests_count[test_y] += 1

	# acc = correct_count / tests_count
	acc_av = np.sum(correct_count) / np.sum(tests_count)
	
	return acc_av

def find_best_k(x_train, y_train, x_test, y_test):

	k_list = [1, 2, 3, 5, 10]
	acc_av = []

	for k in k_list:
		a_a = kNN(x_train, y_train, x_test, y_test, k)
		acc_av.append(a_a)
		print("Average accuracy in k="+str(k)+" is "+str(a_a))

	plt.plot(k_list, acc_av, '-o')
	plt.xlabel("k")
	plt.ylabel("accuracy")
	plt.title('Find best k value in (k = 1,2,3,5,10)')
	plt.show()

def main():
	data=pd.read_csv('train.csv')

	y_data = data['TenYearCHD']
	x_data = data.drop('TenYearCHD', axis = 1)

	sex_data = pd.get_dummies(x_data['sex'], drop_first = True)
	sex_data.rename(columns={'M':'is_male'}, inplace=True)

	smoking_data = pd.get_dummies(x_data['is_smoking'], drop_first = True)
	smoking_data.rename(columns={'YES':'is_smoking'}, inplace=True)

	x_data.drop(['sex', 'is_smoking'], axis = 1, inplace = True)
	x_data = pd.concat([x_data, sex_data, smoking_data], axis = 1)

	x_dv = x_data.values
	y_dv = y_data.values
	x = [d for d in x_dv]
	y = [d for d in y_dv]

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

	# accuracy = kNN(x_train, y_train, x_test, y_test, 3)
	# print("Average accuracy: ", accuracy)
	find_best_k(x_train, y_train, x_test, y_test)

if __name__ == '__main__':
	main()