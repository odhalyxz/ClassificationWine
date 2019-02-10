#Import scikit-learn dataset library
from sklearn import datasets
def readDataSet():
	#Load dataset
	wine = datasets.load_wine()

	return wine