import glob
import os, random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Dataset:
	trainset = []
	targetset = []
	testset = []
	def __init__(self, path):
		self.path = path

	def createDataset(self):
		i = 0; arr1 = []; arr2 = []
		for f1 in glob.glob(self.path):
			arr1.append(os.listdir(f1))
			arr2.append(i)
			i += 1
		self.dataset['data'] = np.array(arr1)
		self.dataset['target'] = np.array(arr2)
		self.setData()
		self.setTarget()

	def setData(self):
		self.data.extend(list(self.dataset['data']))

	def setTarget(self):
		self.target.extend(list(self.dataset['target']))

	def setTestset(self): #ham nay lay ra 5000 doc de test
		temp = set()
		for f1 in glob.glob(self.path):
			elements = random.randint(0, 1000)
			if len(temp) < 5000:
				x = set(random.sample(os.listdir(f1), elements))
				temp.update(x)
			else:
				break
		self.testset.extend(random.sample(temp, 5000))

	def setTrainset(self):
		temp = set()
		while len(temp) < 1000:
			pathchild = random.choice(os.listdir(path))
			f1 = self.path+"/"+pathchild
			elements = random.randint(0, 1000)
			x = set(random.sample(os.listdir(f1), elements)) - set(self.testset)
			self.targetset.extend([pathchild for i in range(len(x))])
			temp.update(x)
		self.trainset.extend(random.sample(temp, n))




d = Dataset('20_newsgroups/*')
d.createDataset()

# doc_X = d.data
# doc_Y = d.target

# sample = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]
# label = [0,0,1,2,2,2,3,3,3,3]

# X_train, X_test, Y_train, Y_test = train_test_split(sample, label, test_size=5)
# clf = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2)
# clf.fit(X_train, Y_train)#Fit the model using X as training data and y as target values
# Y_pred = clf.predict(X_test)
# print(Y_pred)
# print(list(Y_pred))
# print(Y_test)


# X_train, X_test, Y_train, Y_test = train_test_split(doc_X, doc_Y, test_size=10)

# # print(X_test)
# # print(Y_test)

# clf = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2)
# clf.fit(X_train, Y_train)#Fit the model using X as training data and y as target values
# Y_pred = clf.predict(X_test)
# print(Y_pred)
# # print(Y_test)