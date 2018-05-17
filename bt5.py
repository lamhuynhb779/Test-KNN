import glob, os, random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

class Dataset:
	trainset = []
	targetset = []
	testset = []
	targettestset = []
	def __init__(self, path):
		self.path = path

	def createDataset(self, n):
		self.setTestset()
		self.setTrainset(n)

	def setTestset(self): #ham nay lay ra 5000 doc de test
		temp = []; temp2 = []
		self.testset = []; self.targettestset = []
		for f1 in glob.glob(self.path+"/*"):
			elements = random.randint(0, 1000)
			if len(temp) < 5000:
				x = set(random.sample(os.listdir(f1), elements))
				temp2.extend([f1[14:] for i in range(len(x))])
				temp.extend(list(x))
			else:
				break
		# self.testset.extend(random.sample(temp, 5000))
		self.testset.extend([temp[i] for i in range(5000)])
		self.targettestset.extend([temp2[i] for i in range(5000)])

	def setTrainset(self, n):
		temp = []
		self.trainset = []; self.targetset = []
		while len(temp) < n:
			pathchild = random.choice(os.listdir(self.path))
			f1 = self.path+"/"+pathchild
			elements = random.randint(0, 1000)
			if len(temp)+elements >= n:
				x = set(random.sample(os.listdir(f1), n - len(temp))) - set(self.testset)
			else:
				x = set(random.sample(os.listdir(f1), elements)) - set(self.testset)
			y = [pathchild for i in range(len(x))]
			self.targetset.extend(y)
			temp.extend(list(x))		
		self.trainset.extend(list(temp))
		self.targetset = [self.targetset[i] for i in range(n)]

	def draw_pr_curve(self, pre, rec):
		plt.xlabel('Trainset')
		plt.ylabel('Accuracy of 100NN')
		plt.ylim([0.0, 100.0])
		plt.xlim([0.0, 15000.0])
		
		plt.step(rec,pre,where='pre', label='Accuracy (%)')
		plt.title('K-Nearest Neighbors')
		plt.legend()
		plt.show()



def main():
	rec = [1000, 3000, 5000, 8000, 10000, 12000, 15000]
	pre = []
	l = len(rec)
	i = 0
	while i < l:		
		d = Dataset('20_newsgroups')
		d.createDataset(rec[i])
		# print(len(d.trainset), len(d.targetset), len(d.testset), len(d.targettestset))
		# X_train, X_test, Y_train, Y_test = train_test_split(d.trainset, d.targetset, test_size=5000)
		X_train = np.reshape(d.trainset,(-1,1))
		X_test = np.reshape(d.testset,(-1,1))
		Y_train = d.targetset
		Y_test = d.targettestset
		clf = neighbors.KNeighborsClassifier(n_neighbors = 100, p = 2)
		clf.fit(X_train, Y_train)#Fit the model using X as training data and y as target values
		Y_pred = clf.predict(X_test)
		# print(list(Y_pred))
		# print(Y_test)
		pre.append(accuracy_score(Y_test, list(Y_pred))*100)
		print("Accuracy of 100NN: %.2f %%" %(pre[i]))
		i += 1
	d.draw_pr_curve(pre, rec)

if __name__ == "__main__":
	main()

# ## call MultinomialNB
# 		clf = MultinomialNB()
# 		# training 
# 		clf.fit(X_train, Y_train)
# 		# test
# 		print('Predicting class of d5:', str(clf.predict(X_test)[0]))
# 		print('Probability of d6 in each class:', clf.predict_proba(X_test))