from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt
import numpy as np

############################Use train.csv from https://www.kaggle.com/c/digit-recognizer/data###########################

my_data = genfromtxt('train.csv', delimiter=',')

##########################Untouched Images########################

data = my_data[1:,:]

trainData = data[0:10000, :]
testData  = data[10001:11000, :]

nrows = trainData.shape[0]
ncols = trainData.shape[1]

for i in range(nrows):
	for j in range(ncols):
		if(trainData[i,j] > 85):
			trainData[i, j] = 255

train_data = trainData[:,1:]
train_labels = trainData[:,0]
test_data = testData[:,1:]
test_labels = testData[:,0]

forest = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=0)
forest.fit(train_data, train_labels)


for i in range(test_data.shape[0]):
	for j in range(test_data.shape[1]):
		if(test_data[i,j] > 85):
			test_data[i, j] = 255

results = forest.predict(test_data)

correct_predictions = 0
total_examples = 0
for i in range(results.shape[0]):
	if(results[i] == test_labels[i]):
		correct_predictions += 1
	total_examples += 1

print("Accuracy: ")
print(correct_predictions)
print(total_examples)
print((correct_predictions/(total_examples*1.0))*100)


#######################Stretched Bounding Box##############################
from PIL import Image
from numpy import array
import PIL.Image

my_data_sbb = genfromtxt('train.csv', delimiter=',')
data_sbb = my_data_sbb[1:,:]

# np.floor(0.8*my_data.shape[0])
trainData = data_sbb[0:5000, :]
testData = data_sbb[5001:5300, :]

trainData_sbb = trainData[:,1:]
trainLabels_sbb = trainData[:,0]
testData_sbb = testData[:,1:]
testLabels_sbb = testData[:,0]

nrows = trainData_sbb.shape[0]
ncols = trainData_sbb.shape[1]
nrows2 = testData_sbb.shape[0]
ncols2 = testData_sbb.shape[1]

def resizeFunction(newMatrix):
	numRows = newMatrix.shape[0]
	numCols = newMatrix.shape[1]
	print("-----------------------")
	print(numRows)
	print(numCols)
	if(numRows < 20):
		a = np.zeros((20-numRows, numCols))
		newMatrix = np.append(newMatrix, a, axis=0)
	if(numRows > 20):
		newMatrix = newMatrix[1:21, :]
	if(numCols < 20):
		a = np.zeros((20, 20-numCols))
		print(a.shape)
		print(newMatrix.shape)
		newMatrix = np.append(newMatrix, a, axis=1)
	if(numCols > 20):
		newMatrix = newMatrix[:, 1:21]
	return newMatrix

processedTrainData = np.empty( shape=(400,) )
for i in range(nrows):
	print(i)
	converted2DMatrix = np.reshape(trainData_sbb[i,:], (28,28))
	train_nonzero_indices = np.argwhere(converted2DMatrix)
	row_min = np.min(train_nonzero_indices[:, 0])
	row_max = np.max(train_nonzero_indices[:, 0])
	col_min = np.min(train_nonzero_indices[:, 1])
	col_max = np.max(train_nonzero_indices[:, 1])

	newMatrix = converted2DMatrix[row_min:row_max+1, col_min:col_max+1]
	newMatrix = resizeFunction(newMatrix)	
	new1DArray = newMatrix.flatten()
	# new1DArray[new1DArray > 0] = 255
	# new1DArray[new1DArray <= 0] = 0
	processedTrainData = np.vstack([processedTrainData, new1DArray])


forest = RandomForestClassifier(n_estimators=30, max_depth=16, random_state=0)
forest.fit(processedTrainData[1:,:], trainLabels_sbb)



processedTestData = np.empty( shape=(400, ) )
for i in range(nrows2):
	print(i)
	convertedTestMatrix = np.reshape(testData_sbb[i,:], (28,28))

	test_nonzero_indices = np.argwhere(convertedTestMatrix)
	row_min = np.min(test_nonzero_indices[:, 0])
	row_max = np.max(test_nonzero_indices[:, 0])
	col_min = np.min(test_nonzero_indices[:, 1])
	col_max = np.max(test_nonzero_indices[:, 1])
	newCroppedMatrix = convertedTestMatrix[row_min:row_max+1, col_min:col_max+1]
	try:
		newCroppedMatrix = resizeFunction(newCroppedMatrix)
		newArray = newCroppedMatrix.flatten()
	except ValueError:
		print("value error-------------------")
		newArray = np.zeros(shape=(400,))
	processedTestData = np.vstack([processedTestData, newArray])

print(processedTestData.shape)
print(processedTestData[1:, :])
print(testLabels_sbb.shape)
results = forest.predict(processedTestData[1:, :])

print(results)
print(testLabels_sbb)

correct_predictions = 0
total_examples = 0
for i in range(results.shape[0]):
	if(results[i] == testLabels_sbb[i]):
		correct_predictions += 1
	total_examples += 1

print(correct_predictions)
print(total_examples)
