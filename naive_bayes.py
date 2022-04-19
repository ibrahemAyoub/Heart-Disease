#!/usr/bin/env python
# coding: utf-8

# In[1]:


from csv import reader
from math import sqrt
from math import exp
from math import pi
import csv
import math
import random
import pandas as pd


# In[2]:


df = pd.read_csv('heart_disease_male.csv', header=None)


# In[3]:


col_names = ['age', 'chest_pain_type', 'rest_blood_pressure', 'blood_sugar', 'rest_electro', 'max_heart_rate', 'exercice_angina','disease']
df.columns = col_names


# In[4]:


df = df[~df['rest_electro'].isin(['?'])]


# In[5]:


df.to_csv('heart_disease_male1.csv', index=False,header=None)


# In[6]:


df


# In[7]:


# Importing LabelEncoder from Sklearn
# library from preprocessing Module.
from sklearn.preprocessing import LabelEncoder
 
# Creating a instance of label Encoder.
le = LabelEncoder()
 
# Using .fit_transform function to fit label
# encoder and return encoded label
disease = le.fit_transform(df['disease'])
exercice_angina = le.fit_transform(df['exercice_angina'])
rest_electro = le.fit_transform(df['rest_electro'])
blood_sugar = le.fit_transform(df['blood_sugar'])
chest_pain_type = le.fit_transform(df['chest_pain_type'])


# In[8]:


# removing the column 'Purchased' from df
# as it is of no use now.
# Appending the array to our dataFrame
# with column name 'Purchased'
df["disease"] = disease
df["exercice_angina"] = exercice_angina
df["rest_electro"] = rest_electro
df["blood_sugar"] = blood_sugar
df["chest_pain_type"] = chest_pain_type
 
# printing Dataframe
df


# In[9]:


print(df["rest_electro"].values)


# In[10]:


# Importing library
import math
import random
import csv


# the categorical class names are changed to numberic data
# eg: yes and no encoded to 1 and 0
def encode_class(mydata):
	classes = []
	for i in range(len(mydata)):
		if mydata[i][-1] not in classes:
			classes.append(mydata[i][-1])
	for i in range(len(classes)):
		for j in range(len(mydata)):
			if mydata[j][-1] == classes[i]:
				mydata[j][-1] = i
	return mydata		
			

# Splitting the data
def splitting(mydata, ratio):
	train_num = int(len(mydata) * ratio)
	train = []
	# initially testset will have all the dataset
	test = list(mydata)
	while len(train) < train_num:
		# index generated randomly from range 0
		# to length of testset
		index = random.randrange(len(test))
		# from testset, pop data rows and put it in train
		train.append(test.pop(index))
	return train, test


# Group the data rows under each class
def groupUnderClass(mydata):
	dict = {}
	for i in range(len(mydata)):
		if (mydata[i][-1] not in dict):
			dict[mydata[i][-1]] = []
		dict[mydata[i][-1]].append(mydata[i])
	return dict


# Calculating Mean
def mean(numbers):
	return sum(numbers) / float(len(numbers))

# Calculating Standard Deviation
def std_dev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
	return math.sqrt(variance)

def MeanAndStdDev(mydata):
	info = [(mean(attribute), std_dev(attribute)) for attribute in zip(*mydata)]
	# here mean of 1st attribute =(a + m+x), mean of 2nd attribute = (b + n+y)/3
	# delete summaries of last class
	del info[-1]
	return info

# find Mean and Standard Deviation under each class
def MeanAndStdDevForClass(mydata):
	info = {}
	dict = groupUnderClass(mydata)
	for classValue, instances in dict.items():
		info[classValue] = MeanAndStdDev(instances)
	return info


# Calculate Gaussian Probability Density Function
def calculateGaussianProbability(x, mean, stdev):
	expo = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
	return (1 / (math.sqrt(2 * math.pi) * stdev)) * expo


# Calculate Class Probabilities
def calculateClassProbabilities(info, test):
	probabilities = {}
	for classValue, classSummaries in info.items():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, std_dev = classSummaries[i]
			x = test[i]
			probabilities[classValue] *= calculateGaussianProbability(x, mean, std_dev)
	return probabilities


# Make prediction - highest probability is the prediction
def predict(info, test):
	probabilities = calculateClassProbabilities(info, test)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel


# returns predictions for a set of examples
def getPredictions(info, test):
	predictions = []
	for i in range(len(test)):
		result = predict(info, test[i])
		predictions.append(result)
	return predictions

# Accuracy score
def accuracy_rate(test, predictions):
	correct = 0
	for i in range(len(test)):
		if test[i][-1] == predictions[i]:
			correct += 1
	return (correct / float(len(test))) * 100.0


# In[11]:


mydata = df[['age', 'chest_pain_type', 'rest_blood_pressure', 'blood_sugar', 'rest_electro','max_heart_rate','exercice_angina','disease']].values


# In[12]:


print(mydata)
pd.DataFrame(mydata).to_csv('mydata.csv',index=False)


# In[13]:


for i in range(len(mydata)):
    mydata[i] = [float(x) for x in mydata[i]]


# In[14]:


print(mydata)


# In[18]:


# split ratio = 0.7
# 70% of data is training data and 30% is test data used for testing
ratio = 0.7
train_data, test_data = splitting(mydata, ratio)
print('Total number of examples are: ', len(mydata))
print('Out of these, training examples are: ', len(train_data))
print("Test examples are: ", len(test_data))


# In[19]:


# prepare model
info = MeanAndStdDevForClass(train_data)


# In[20]:


# test model
predictions = getPredictions(info, test_data)
accuracy = accuracy_rate(test_data, predictions)
print("Accuracy of your model is: ", accuracy)


# In[110]:


predict(info,[43,0,140,0,2,135,1])

