from numpy import genfromtxt
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB

import numpy as np

############################Use train.csv from https://www.kaggle.com/c/digit-recognizer/data###########################

##########################Untouched Images########################
np.set_printoptions(edgeitems=3,infstr='inf',linewidth=75, nanstr='nan', precision=8,suppress=False, threshold=1000, formatter=None)
my_data = genfromtxt('train.csv', delimiter=',')
data = my_data[1:,:]

trainData = data[0:300, :]
testData  = data[301:400, :]

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


model1 = GaussianNB()
model2 = BernoulliNB()
model1.fit(train_data, train_labels)
model2.fit(train_data, train_labels)


for i in range(test_data.shape[0]):
	for j in range(test_data.shape[1]):
		if(test_data[i,j] > 85):
			test_data[i, j] = 255

results1 = model1.predict(test_data)
results2 = model2.predict(test_data)


correct_predictions = 0
total_examples = 0
for i in range(results1.shape[0]):
	if(results1[i] == test_labels[i]):
		correct_predictions += 1
	total_examples += 1

print("Accuracy: ")
print(correct_predictions)
print(total_examples)
print((correct_predictions/(total_examples*1.0))*100)

correct_predictions = 0
total_examples = 0
for i in range(results2.shape[0]):
	if(results2[i] == test_labels[i]):
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

trainData = data_sbb[0:35000, :]
testData = data_sbb[35001:39999, :]

trainData_sbb = trainData[:,1:]
trainLabels_sbb = trainData[:,0]
testData_sbb = testData[:,1:]
testLabels_sbb = testData[:,0]

nrows = trainData_sbb.shape[0]
ncols = trainData_sbb.shape[1]
nrows2 = testData_sbb.shape[0]
ncols2 = testData_sbb.shape[1]


# for i in range(nrows):
# 	for j in range(ncols):
# 		if(trainData_sbb[i,j] > 20):
# 			trainData_sbb[i, j] = 255
# 		else:
# 			trainData_sbb[i, j] = 0

def resizeFunction(newMatrix):
	numRows = newMatrix.shape[0]
	numCols = newMatrix.shape[1]
	if(numRows < 20):
		a = np.zeros((20-numRows, numCols))
		newMatrix = np.append(newMatrix, a, axis=0)
	if(numRows > 20):
		newMatrix = newMatrix[1:21, :]
	if(numCols < 20):
		a = np.zeros((20, 20-numCols))
		newMatrix = np.append(newMatrix, a, axis=1)
	if(numCols > 20):
		newMatrix = newMatrix[:, 1:21]
	return newMatrix


processedTrainData = np.empty( shape=(400,) )
for i in range(nrows):
	print(i)
	converted2DMatrix = np.reshape(trainData_sbb[i,:], (28,28))
	# np.savetxt('oldmatrix.txt', converted2DMatrix.astype(int),'%5.2f')
	train_nonzero_indices = np.argwhere(converted2DMatrix)
	row_min = np.min(train_nonzero_indices[:, 0])
	row_max = np.max(train_nonzero_indices[:, 0])
	col_min = np.min(train_nonzero_indices[:, 1])
	col_max = np.max(train_nonzero_indices[:, 1])
	newMatrix = converted2DMatrix[row_min:row_max+1, col_min:col_max+1]
	# np.savetxt('croppedmatrix.txt', newMatrix.astype(int), '%5.2f')
	newMatrix = resizeFunction(newMatrix)
	# np.savetxt('resizedmatrix.txt', newMatrix.astype(int), '%5.2f')
	new1DArray = newMatrix.flatten()
	# new1DArray[new1DArray > 0] = 255
	# new1DArray[new1DArray <= 0] = 0
	processedTrainData = np.vstack([processedTrainData, new1DArray])


model1 = GaussianNB()
model2 = BernoulliNB()

model1.fit(processedTrainData[1:,:], trainLabels_sbb)
model2.fit(processedTrainData[1:,:], trainLabels_sbb)


# for i in range(nrows2):
# 	for j in range(ncols2):
# 		if(testData_sbb[i,j] > 20):
# 			testData_sbb[i, j] = 255
# 		else:
# 			testData_sbb[i, j] = 0

processedTestData = np.empty( shape=(400, ) )
for i in range(nrows2):
	print(i)
	convertedTestMatrix = np.reshape(testData_sbb[i,:], (28,28))
	# np.savetxt('t_oldmatrix.txt', convertedTestMatrix.astype(int), '%5.2f')

	test_nonzero_indices = np.argwhere(convertedTestMatrix)
	row_min = np.min(test_nonzero_indices[:, 0])
	row_max = np.max(test_nonzero_indices[:, 0])
	col_min = np.min(test_nonzero_indices[:, 1])
	col_max = np.max(test_nonzero_indices[:, 1])
	newCroppedMatrix = convertedTestMatrix[row_min:row_max+1, col_min:col_max+1]
	# np.savetxt('t_croppedmatrix.txt', newCroppedMatrix.astype(int), '%5.2f')
	try:
		newCroppedMatrix = resizeFunction(newCroppedMatrix)
		# np.savetxt('t_resizedmatrix.txt', newMatrix.astype(int), '%5.2f')
		newArray = newCroppedMatrix.flatten()
	except ValueError:
		print("value error-------------------")
		newArray = np.zeros(shape=(400,))
	processedTestData = np.vstack([processedTestData, newArray])


results1 = model1.predict(processedTestData[1:,:])
results2 = model2.predict(processedTestData[1:,:])

print(results1)
print(testLabels_sbb)

correct_predictions = 0
total_examples = 0
for i in range(results1.shape[0]):
	if(results1[i] == testLabels_sbb[i]):
		correct_predictions += 1
	total_examples += 1

print("Gaussian Results")
print(correct_predictions)
print(total_examples)

correct_predictions = 0
total_examples = 0
for i in range(results2.shape[0]):
	if(results2[i] == testLabels_sbb[i]):
		correct_predictions += 1
	total_examples += 1

print("Bernoulli Results")
print(correct_predictions)
print(total_examples)
