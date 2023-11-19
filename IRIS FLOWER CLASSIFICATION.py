#!/usr/bin/env python
# coding: utf-8

# # BHARAT INTERN: MACHINE LEARNING INTERN

# # TASK 3: Iris Flower Prediction

# # DATASET INFORMATION

# The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. One class is linearly separable from the other 2; the latter are NOT linearly separable from each other.
# 
# ## Atribute Information:
#     (1) sepal length in cm
#     (3) petal length in cm
#     (5) Species: --Iris Setosa --Iris Versicolour --Iris Virginica

# # Import modules

# In[38]:


import pandas as pd #to read dataset
import numpy as np #to do some operaion with the dataset
import os #to add some files
import matplotlib.pyplot as plt #to visualize detailed data as graph model
import seaborn as sns #graph model(simple)
import warnings
warnings.filterwarnings('ignore') #to ignoer warnings


# # Loading dataset

# In[75]:


df = pd.read_csv('Iris.csv')
df.head()


# In[76]:


df = df.drop(columns = ['Id','SepalWidthCm', 'PetalWidthCm'])
df.head()


# In[77]:


# to display stateestics about data
df.describe()


# In[78]:


#to display basic info about datatype
df.info()


# In[79]:


#to display no. of samples of each class
df['Species'].value_counts()


# # Preprocessing the dataset

# In[80]:


#check for null values wheather there are
df.isnull().sum()


# # Exploratory data analysis

# In[44]:


#datas in form of graphs - histograms
df['SepalLengthCm'].hist()


# In[67]:


#df['SepalWidthCm'].hist()


# In[46]:


df['PetalLengthCm'].hist()


# In[68]:


#df['PetalWidthCm'].hist()


# In[81]:


#scatterplot 
colors = ['red', 'orange', 'blue']
species = ['Iris-virginica','Iris-versicolor','Iris-setosa']


# In[82]:


#for i in range(3):
  #  x = df[df['Species'] == species[i]]
  #  plt.scatter(x['SepalLengthCm'], x['SepalWidthCm'], c = colors[i], label=species[i])
#plt.xlabel("Sepal Length")
#plt.ylabel("Sepal Width")
#plt.legend()


# In[83]:


#for i in range(3):
 #   x = df[df['Species'] == species[i]]
 #   plt.scatter(x['PetalLengthCm'], x['PetalWidthCm'], c = colors[i], label=species[i])
#plt.xlabel("Petal Length")
#plt.ylabel("Petal Width")
#plt.legend()


# In[84]:


for i in range(3): #iterate 3 classes
    x = df[df['Species'] == species[i]] #filter the points
    plt.scatter(x['SepalLengthCm'], x['PetalLengthCm'], c = colors[i], label=species[i]) #plot 
plt.xlabel("Sepal Length") #x axis
plt.ylabel("Petal Length") #y axis
plt.legend()


# In[69]:


#for i in range(3):
 #   x = df[df['Species'] == species[i]]
 #   plt.scatter(x['SepalWidthCm'], x['PetalWidthCm'], c = colors[i], label=species[i])
#plt.xlabel("Sepal Width")
#plt.ylabel("Petal Width")
#plt.legend()


# # Coorelation Matrix

# A correlation matrix is a table showing correlation coefficients between variables. Each cell in the table shows the correlation between two variables. The value is in the range of -1 to 1. If two varibles have high correlation, we can neglect one variable from those two.

# In[89]:


df.corr()


# In[104]:


corr = df.corr()
fig, ax = plt.subplots(figsize=(5,4)) #set the size of the graph
sns.heatmap(corr, annot=True, ax=ax, cmap = 'coolwarm')


# # Label Encoder

# In machine learning, we usually deal with datasets which contains multiple labels in one or more than one columns. These labels can be in the form of words or numbers. Label Encoding refers to converting the labels into numeric form so as to convert it into the machine-readable form

# In[92]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[93]:


df['Species'] = le.fit_transform(df['Species']) #species numeric form
df.head()


# # Model Training

# In[105]:


from sklearn.model_selection import train_test_split
# train - 70
# test - 30
X = df.drop(columns=['Species'])
Y = df['Species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30) #text on the basis of input


# In[107]:


# logistic regression 
from sklearn.linear_model import LogisticRegression
model = LogisticRegression() #initialize model


# In[108]:


# model training
model.fit(x_train, y_train) # to train data


# In[109]:


# print metric to get performance
print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[110]:


# knn - k-nearest neighbours
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()


# In[111]:


model.fit(x_train, y_train)


# In[112]:


# print metric to get performance
print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[113]:


# decision tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()


# In[114]:


model.fit(x_train, y_train)


# In[115]:


# print metric to get performance
print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[ ]:




