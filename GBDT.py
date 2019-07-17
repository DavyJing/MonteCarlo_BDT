from random import seed
from random import randrange
import pickle
import numpy as np
import treePlotter

def load_dataset(filename):
	fr = open(filename)
	datamat = []
	for line in fr.readlines():
		cutLine = line.strip().split()
		floatLine = map(float,cutLine)
		datamat += [floatLine]
	return datamat

def binarySplit(dataset,feature,val):
	matLeft = dataset[np.nonzero(dataset[:,feature] <= value)[0],:]
	matRight = dataset[np.nonzero(dataset[:,feature] > value)[0],:]
	return matLeft, matRight

def regressLeaf(dataset):
	return np.mean(dataset[:,-1])

def regressErr(dataset):
	return np.var(dataset[:-1]) * dataset.shape[0]

def regressData(filename):
	fr = open(filename)
	return pickle.load(fr)

def chooseBestSplit(dataset,leafType=regressLeaf,errType=regressErr,threshold=(1,4)):
	threshouldErr, threshouldSample = threshold[0], threshold[1]
	if len(set(dataset[:,-1].T.tolist()[0])) == 1:
		return None,leafType(dataset)
	m,n = dataset.shape
	Err = errType(dataset)
	bestErr, bestFeatureIndex, bestFeatureValue = np.inf, 0, 0
	for featureIndex in range(n-1):
		for featureValue in dataset[:,featureIndex]:
			matLeft, matRight = binarySplit(dataset,featureIndex,featureValue)
			if (np.shape(matLeft)[0] < thresholdSamples) or (np.shape(matRight)[0] < thresholdSamples):
				continue
			tmpErr = errType(matLeft) + errType(matRight)
			if tmpErr < bestErr:
				bestErr,bestFeatureIndex,bestFeatureValue = tmpErr,featureIndex,featureValue
	if (Err - bestErr) < threshouldErr:
		return None, leafType(dataset)
	matLeft,matRight = binarySplitDataSet(dataset,bestFeatureIndex,bestFeatureValue)
	if (np.shape(matLeft)[0] < thresholdSamples) or (np.shape(matRight)[0] < thresholdSamples):
		return None,leafType(dataset)
	return bestFeatureIndex,bestFeatureValue

def createCART(dataset,leafType=regressLeaf,errType=regressErr,threshold=(1,4),depth=depth):
	if depth==0: return leafType(dataset)
	feature,value = chooseBestSplit(dataset,leafType,errType,threshold)
	if feature == None: return value
	returnTree = {}
	returnTree['bestSplitFeature'] = feature
	returnTree['bestFeatureValue'] = value
	leftSet, rightSet = binarySplit(dataset,feature,value)
	returnTree['left'] = createCART(leftSet,leafType,errType,threshold,depth-1)
	returnTree['right'] = createCART(rightSet,leafType,errType,threshold,depth-1)
	return returnTree

def isTree(obj):
	return (type(obj).__name__=='dict')

def regressEvaluation(tree,inputData):
	return float(tree)

def treePredict(tree,inputData,modelEval = regressEvalutaion):
	if not isTree(tree): return modelEval(tree,inputData)
	if inputFata[tree['bestSplitFeature']] <= tree['bestFeatureValue']:
		if isTree(tree['left']):
			return treePredict(tree['left'],inputData,modelEval)
		else:
			return modelEval(tree['left'],inputData)
	else:
		if isTree(tree['right']):
			return treePredict(tree['right'],inputData,modelEval)
		else:
			return modelEval(tree['right'],inputData)	

def wrappedPredict(tree,testData,modelEval=regressEvaluation):
	m = len(testData)
	yHat = np.mat(np.zeros((m,1)))
	for i in range(m):
		yHat[i] = treePredict(tree,testData[i],modelEval)
	return yHat

if __name__ = '__main__':
	regressTree = createCART(trainDataset,threshold(1,4),5)
	
