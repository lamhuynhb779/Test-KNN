import glob, os, random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Dataset:
	category = {}
	trainset = []
	targetset = []
	testset = []
	def __init__(self, path):
		self.path = path

	def createDataset(self):
		self.getCategory()
		self.setTestset()
		self.setTrainset()

	def getCategory(self):
		i = 0
		for f in glob.glob(self.path+"/*"):
			self.category[f[14:]] = i
			i += 1

	def setTestset(self): #ham nay lay ra 5000 doc de test
		temp = set()
		for f1 in glob.glob(self.path+"/*"):
			elements = random.randint(0, 5)
			if len(temp) < 20:
				x = set(random.sample(os.listdir(f1), elements))
				temp.update(x)
			else:
				break
		self.testset.extend(random.sample(temp, 20))

	def setTrainset(self):
		temp = []
		while len(temp) < 10:
			pathchild = random.choice(os.listdir(self.path))
			f1 = self.path+"/"+pathchild
			elements = random.randint(0, 2)
			if len(temp)+elements >= 10:
				x = set(random.sample(os.listdir(f1), 10 - len(temp))) - set(self.testset)
			else:
				x = set(random.sample(os.listdir(f1), elements)) - set(self.testset)
			y = [pathchild for i in range(len(x))]
			self.targetset.extend(y)
			temp.extend(list(x))		
		self.trainset.extend(list(temp))
		self.targetset = [self.targetset[i] for i in range(10)]

d = Dataset('20_newsgroups/')
d.createDataset()

# print(d.testset)
# results = list(map(int, d.testset))
# print(results)
# print(d.trainset)
# print(d.targetset)

# print(len(d.testset))
# print(len(d.trainset))
# print(len(d.targetset))

# doc_X = d.data
# doc_Y = d.target

# sample = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]
# label = [0,0,1,2,2,2,3,3,3,3]
# for i in d.category:
# 	print(i)

X_train, X_test, Y_train, Y_test = train_test_split(d.trainset, d.targetset, test_size=5)
print(X_train); print(Y_train)
print(X_test); print(Y_test)
X = np.reshape(X_train,(-1,1))
print(X)
Xt = np.reshape(X_test,(-1,1))
clf = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2)
clf.fit(X, Y_train)#Fit the model using X as training data and y as target values
Y_pred = clf.predict(Xt)
# print(Y_pred)
print(list(Y_pred))
print(Y_test)


# X_train, X_test, Y_train, Y_test = train_test_split(doc_X, doc_Y, test_size=10)

# # print(X_test)
# # print(Y_test)

# clf = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2)
# clf.fit(X_train, Y_train)#Fit the model using X as training data and y as target values
# Y_pred = clf.predict(X_test)
# print(Y_pred)
# # print(Y_test)