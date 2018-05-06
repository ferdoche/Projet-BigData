# -*- coding: utf-8 -*-
"""
Created on Tue May  1 21:11:47 2018

@author: Ferdinand
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv('winequality-red.csv', sep=';')
data.describe()

index = ["fixed acidity","volatile acidity","citric acid","residual sugar",
         "chlorides","free sulfur dioxide","total sulfur dioxide","density",
         "pH","sulphates","alcohol","quality"]

pH = data['pH']
pH.describe()
plt.hist(pH, bins = 25)



# =============================================================================
# PCA
# =============================================================================
X = data.values
xtx = np.dot(np.transpose(X),X)

###Normalisation###
mean = sum(X[:])/len(X)
var = sum(X[:]**2/len(X))-mean**2
X = X[:]-mean
X = X[:]/(var**(1/2))

###PCA###
#U, D_vp , V = np.linalg.svd(X,full_matrices=False) 
D_vp,V = np.linalg.eig(np.dot(np.transpose(X),X)/(len(X)-1)) 
#V = np.transpose(V)
D = np.diag(D_vp)
axis_x = [i for i in range(1,13)]
axis_y = [i for i in D_vp]
plt.bar(axis_x,axis_y)

###Projections PCA###
PCA_x = np.zeros((len(X),12))
PCA_y = np.zeros((len(X),12))
for i in range(0,12) : PCA_x[:,i] = X[:,i]*V[i,0] 
for i in range(0,12) : PCA_y[:,i] = X[:,i]*V[i,1] 


for ii in range(0,12):
    axis_x = [i for i in PCA_x[:,ii]]
    axis_y = [i for i in PCA_y[:,ii]]
    plt.scatter(axis_x, axis_y,s=0.7, label=index[ii])
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    
 
plt.legend(loc=3, prop={'size': 6}) 
plt.show()

###Format vectoriel###
import matplotlib.colors as colors
colors_list = list(colors._colors_full_map.values())
origin = [0], [0] 
for ii in range(0,12):
    plt.quiver(*origin,V[ii,0],V[ii,1],scale=1.5,color=colors_list[-ii],label=index[ii])

plt.legend(loc=3, prop={'size': 6}) 
plt.show()

# =============================================================================
# Multivariate linear regression
# =============================================================================
quality = X[:,11]
quality_mean = sum(quality)/len(X)
quality_var = sum((quality - quality_mean)**2)

data_set = X[:,0:11]
mean_data_set = sum(data_set[:])/len(X)
data_set = data_set[:]- mean_data_set
beta = np.zeros(11)


xty = np.dot(np.transpose(data_set),quality)
xtx = np.dot(np.transpose(data_set),data_set)

beta = np.dot(np.linalg.inv(xtx),xty)
rss = sum((np.dot(data_set,beta) - quality_mean)**2)
R2 = rss/quality_var


data_set = X[:,0:11]
var_data_set = sum(data_set[:]**2/len(X))-mean_data_set**2
data_set_norm = data_set - mean_data_set
data_set_norm = data_set_norm[:]/(var_data_set**(1/2))

beta_norm = np.zeros(11)

xty = np.dot(np.transpose(data_set_norm),quality)
xtx = np.dot(np.transpose(data_set_norm),data_set_norm)

beta_norm = np.dot(np.linalg.inv(xtx),xty)
rss = sum((np.dot(data_set_norm,beta) - quality_mean)**2)
R2 = rss/quality_var


# =============================================================================
# Lasso Regression
# =============================================================================



"""
X_check = np.dot(U,np.dot(D,V))   #Vérification de la SVD

X_vp = np.dot(D,D)
"""


#D = np.diag(D) #Conversion de D en matrice diagonale








"""
data.values
print(data)

"""

"""
plt.bar(data[:,0], data[:,1], color='g')
plt.ylabel('Frequency')
plt.xlabel('Words')
plt.title('Title')

plt.show()
"""