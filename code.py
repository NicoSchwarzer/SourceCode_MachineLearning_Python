# -*- coding: utf-8 -*-
"""
Created on Fri May 14 17:55:57 2021

@author: Nico
"""

############
### Ex 6 ###
############

import numpy as np
import pandas as pd 
import os 
import scipy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns


#####
# a #
#####

## ordinary least squares

def cost_fn_2a(w, x, y):
    w2 = np.expand_dims(w, 1)
    return x @ (w2 - y)*(w2 - y).sum() 

def LeastSquares(X, Y):
    
    #X: deisgn matrix n x d
    #Y: true values n x 1
    
    #output: weight of linear regression d x 1
    
    w = minimize(cost_fn_2, np.zeros(X.shape[1]),
                 args=(X, Y, lmbd_reg)).x
    return np.expand_dims(w, 1)


## Ridge Regression 

def cost_fn_2b(w, x, y, lmbd):
    w2 = np.expand_dims(w, 1)
    return x @ (w2 - y)*(w2 - y).sum() +\
           lmbd * (w ** 2).sum()   # perhaps using w2 ??
           
           

def  RidgeRegression(X, Y, lmbd_reg=0.):
    ''' solves linear regression with
    L1 Loss + L2 regularization

    X: deisgn matrix n x d
    Y: true values n x 1
    lmbd_reg: weight regularization

    output: weight of linear regression d x 1
    '''
    w = minimize(cost_fn, np.zeros(X.shape[1]),
                 args=(X, Y, lmbd_reg)).x
    return np.expand_dims(w, 1)



#####
# b #
#####

# d= 1 

def Basis(X,k):
    # making sure that X has the correct dimension
    X = np.expand_dims(X, 1)
    
    dim_x = np.size(X, 1)
    dim2 = 2*k + 1
    phi_mat = np.zeros(dim_x, dim_2)
    # first column
    phi_mat[:,0] = 1
    
    # other columns 
        
    for i in range(k):
    # all even column numbers - use sinus basis function

        if i % 1 == 0:
            phi_mat[:,i] = np.sin(2*np.pi*i*X)
        else:
            phi_mat[:,i] = np.cos(2*np.pi*i*X)
            
    # finally:
    
    return phi_mat

    

#####
# c #
#####

# setting WD
os.chdir("C:/Users/Nico/Documents/Tübingen/2. Sem/Stat ML/EX/EX3")

# using: numpy.load(<file name>).item(),
a = np.load("onedim_data.npy", allow_pickle=True).item()

result = a.items()
data = list(result)
data2 = np.array(data, dtype=object)

#data2.size

# finally retrieving data: 
    
# Xtest
print("the values for " + str(data2[0,0]) + " are stored in data2[0,1].")
Xtest =  data2[0,1]

#Xtrain
print("the values for " + str(data2[1,0]) + " are stored in data2[1,1].")
Xtrain = data2[1,1]

#Ytrain
print("the values for " + str(data2[2,0]) + " are stored in data2[2,1].")
Ytrain = data2[2,1]

#Ytest
print("the values for " + str(data2[3,0]) + " are stored in data2[3,1].")
Ytrain = data2[3,1]


## plotting - using histograms

# since the function will only be trained
# on the training data, only considering Xtrain 

plt.figure()
plt.hist(Xtrain, bins = 80)

# rather uniform - L2 is plausible!




# Convert list to an array
AA = np.array(data)

np.cos(0)
np.sin(1)    
    
a = np.array([1,2,3])
b = np.array([3,3,3])

aa = pd.DataFrame({'a':a, 'b':b})
aa.size()    
aa.size

np.size(a,0)

a = np.array([[3,3],[1,3]])
a
a[0,:]  = 9

a = a[:,0]
a
np.sin(a)
