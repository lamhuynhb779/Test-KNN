import glob, os, random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_curve

class Dataset:
	trainset = []
	targetset = []
	testset = []
	targettestset = []
	def __init__(self, path):
		self.path = path
		self.setTestset()

	def createDataset(self, n):
		self.setTrainset(n)

	def setTestset(self):
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
		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.ylim([0.0, 100.0])
		plt.xlim([0.0, 15000.0])
		
		plt.step(rec,pre,where='pre', label='Accuracy (%)')
		plt.title('K-Nearest Neighbors')
		plt.legend()
		plt.show()

def KNN():
	rec = [1000, 3000, 5000, 8000, 10000, 12000, 15000]
	pre = []
	l = len(rec)
	i = 0
	d = Dataset('20_newsgroups')
	X_test = np.reshape(d.testset,(-1,1))
	while i < l:		
		d.createDataset(rec[i])
		X_train = np.reshape(d.trainset,(-1,1))
		Y_train = d.targetset
		Y_test = d.targettestset
		clf = neighbors.KNeighborsClassifier(n_neighbors = 100, p = 2)
		clf.fit(X_train, Y_train)
		Y_pred = clf.predict(X_test)
		pre.append(accuracy_score(Y_test, list(Y_pred))*100)
		print("Accuracy of 100NN: %.2f %%" %(pre[i]))
		i += 1
	d.draw_pr_curve(pre, rec)

def NaiveBayes(): #Multinomial Naive Bayes
	rec = [1000, 3000, 5000, 8000, 10000, 12000, 15000]
	pre = []
	l = len(rec)
	i = 0
	d = Dataset('20_newsgroups')
	X_test = list(map(int, d.testset))
	X_test = np.reshape(X_test,(-1,1))
	while i < l:		
		d.createDataset(rec[i])
		X_train = list(map(int, d.trainset))
		X_train = np.reshape(X_train,(-1,1))
		Y_train = np.array(d.targetset)
		Y_test = d.targettestset
		clf = MultinomialNB()
		clf.fit(X_train, Y_train)
		Y_pred = clf.predict(X_test)[0]
		prob = clf.predict_proba(X_test)[0]
		pre.append(np.amax(prob)*100)
		print('Predicting of testset:', Y_pred)
		print('Probability of testset in each category:', prob)
		print('Probability of testset in %s:' %(Y_pred),":", pre[i])
		print("-"*20)
		i += 1
	# d.draw_pr_curve(pre, rec)
	
def main():
	KNN()
	NaiveBayes()

if __name__ == "__main__":
	main()