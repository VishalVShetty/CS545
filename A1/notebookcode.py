#!/usr/bin/env python
# coding: utf-8

# # A1: Three-Layer Neural Network

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Requirements" data-toc-modified-id="Requirements-1">Requirements</a></span></li><li><span><a href="#Example-Results" data-toc-modified-id="Example-Results-2">Example Results</a></span></li><li><span><a href="#Discussion" data-toc-modified-id="Discussion-3">Discussion</a></span></li></ul></div>

# ## Requirements

# In this assignment, you will start with code from lecture notes 04 and add code to do the following.
# 
# * Add another hidden layer, for a total of two hidden layers.  This layer will use a weight matrix named `U`.  Its outputs will be named `Zu` and the outputs of the second hidden layer will be changed to `Zv`.
# * Define function `forward` that returns the output of all of the layers in the neural network for all samples in `X`. `X` is assumed to be standardized and have the initial column of constant 1 values.
# 
#       def forward(X, U, V, W):
#           .
#           .
#           .
#           Y = . . . # output of neural network for all rows in X
#           return Zu, Zv, Y
#       
# * Define function `gradient` that returns the gradients of the mean squared error with respect to each of the three weight matrices. `X` and `T` are assumed to be standardized and `X` has the initial column of 1's.
# 
#       def gradient(X, T, Zu, Zv, Y, U, V, W):
#           .
#           .
#           .
#           return grad_wrt_U, grad_wrt_V, grad_wrt_W
#           
# * Define function `train` that returns the resulting values of `U`, `V`, and `W` and the standardization parameters.  Arguments are unstandardized `X` and `T`, the number of units in the two hidden layers, the number of epochs and the learning rate, which is the same value for all layers. This function standardizes `X` and `T`, initializes `U`, `V` and `W` to uniformly distributed random values between -1 and 1, and `U`, `V` and `W` for `n_epochs` times as shown in lecture notes 04.  This function must call `forward`, `gradient` and `addOnes`.
# 
#       def train(X, T, n_units_U, n_units_V, n_epochs, rho):
#           .
#           .
#           .
#           return U, V, W, X_means, X_stds, T_means, T_stds
#           
# * Define function `use` that accepts unstandardized `X`, standardization parameters, and weight matrices `U`, `V`, and `W` and returns the unstandardized output.
# 
#       def use(X, X_means, X_stds, T_means, T_stds, U, V, W):
#           .
#           .
#           .
#           Y = ....
#           return Y

# ## Example Results

# ## This is where the solution begins

# Initialization cell below

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import IPython.display as ipd  # for display and clear_output
import time  # for sleep
import pandas


# The below function Adds a coloumn of 1's

# In[3]:


def addOnes(X):
    return np.insert(X, 0, 1, axis=1)


# Add code cells here to define the functions above.  Once these are correctly defined, the following cells should run and produce similar results as those here.

# * Requirement 1 : Define function `forward` that returns the output of all of the layers in the neural network for all samples in `X`. `X` is assumed to be standardized and have the initial column of constant 1 values.
# 
#       def forward(X, U, V, W):
#           .
#           .
#           .
#           Y = . . . # output of neural network for all rows in X
#           return Zu, Zv, Y

# In[4]:


def forward(X, U, V, W):
    Zu = np.tanh(X @ U) #1st layer op
    Zu1 = addOnes(Zu)
    Zv = np.tanh(Zu1 @ V) #2nd layer op
    Zv1 = addOnes(Zv)
    Y = Zv1 @ W
    return Zu, Zv, Y


# * Requirement 2 : Define function `gradient` that returns the gradients of the mean squared error with respect to each of the three weight matrices. `X` and `T` are assumed to be standardized and `X` has the initial column of 1's.
# 
#       def gradient(X, T, Zu, Zv, Y, U, V, W):
#           .
#           .
#           .
#           return grad_wrt_U, grad_wrt_V, grad_wrt_W

# In[5]:


def gradient(X, T, Zu, Zv, Y, U, V, W):
    Dw = T - Y
    Dv = Dw @ W[1:, :].T * (1 - Zv**2)
    Du = Dv @ V[1:, :].T * (1 - Zu**2)
    Zv1 = addOnes(Zv)
    Zu1 = addOnes(Zu)
    grad_wrt_W = - Zv1.T @ Dw
    grad_wrt_V = - Zu1.T @ Dv
    grad_wrt_U = - X.T @ Du
    return grad_wrt_U, grad_wrt_V, grad_wrt_W


# * Requirement 3 : Define function `train` that returns the resulting values of `U`, `V`, and `W` and the standardization parameters.  Arguments are unstandardized `X` and `T`, the number of units in the two hidden layers, the number of epochs and the learning rate, which is the same value for all layers. This function standardizes `X` and `T`, initializes `U`, `V` and `W` to uniformly distributed random values between -1 and 1, and `U`, `V` and `W` for `n_epochs` times as shown in lecture notes 04.  This function must call `forward`, `gradient` and `addOnes`.
# 
#       def train(X, T, n_units_U, n_units_V, n_epochs, rho):
#           .
#           .
#           .
#           return U, V, W, X_means, X_stds, T_means, T_stds

# In[10]:


def train(X, T, nu, nv, n_epochs, rho):
    no = T.shape[1]
    ni = X.shape[1]
    rho = rho / (ni * no)
    # Standardizes X and T
    X_means = X.mean(axis=0)
    X_stds = X.std(axis=0)
    T_means = T.mean(axis=0)
    T_stds = T.std(axis=0)
    XS = (X - X_means) / X_stds
    TS = (T - T_means) / T_stds
    # prepend 1's column in XS
    XS1 = addOnes(XS)
    TS1 = addOnes(TS)
    #Weight matrix
    U = np.random.uniform(-1, 1, size=(1 + ni, nu)) / np.sqrt(XS1.shape[1])
    V = np.random.uniform(-1, 1, size=(1 + nu, nv)) / np.sqrt(nu+1)
    W = np.random.uniform(-1, 1, size=(1 + nv, no)) / np.sqrt(nv+1)
    for epochs in range(n_epochs):
        # Call function for gradient & required Forward function
        Zu, Zv, Y = forward(addOnes(XS), U, V, W)
        grad_wrt_U, grad_wrt_V, grad_wrt_W = gradient(XS1, TS, Zu, Zv, Y, U, V, W)
        # Take step down the gradient
        W = W - rho * grad_wrt_W
        V = V - rho * grad_wrt_V
        U = U - rho * grad_wrt_U
    return U, V, W, X_means, X_stds, T_means, T_stds


# * Reqirement 4 : Define function `use` that accepts unstandardized `X`, standardization parameters, and weight matrices `U`, `V`, and `W` and returns the unstandardized output.
# 
#       def use(X, X_means, X_stds, T_means, T_stds, U, V, W):
#           .
#           .
#           .
#           Y = ....
#           return Y

# In[7]:


def use(X, X_means, X_stds, T_means, T_stds, U, V, W):
    #standardisation of input data
    XS = (X - X_means) / X_stds
    X1 = addOnes(XS)
    Y1 = addOnes(np.tanh(X1 @ U))
    Y2 = addOnes(np.tanh(Y1 @ V))
    Y3 = Y2 @ W
    Y = Y3 * T_stds + T_means
    return Y


# In[12]:


X = (np.arange(40).reshape(-1, 4) - 10) * 0.1
T = np.hstack((X ** 3, np.sin(X)))
U, V, W, X_means, X_stds, T_means, T_stds = train(X, T, 100, 50, 10, 0.01)


# In[ ]:


Xtrain = np.arange(4).reshape(-1, 1)
Ttrain = Xtrain ** 2

Xtest = Xtrain + 0.5
Ttest = Xtest ** 2


# In[ ]:


U = np.array([[1, 2, 3], [4, 5, 6]])  # 2 x 3 matrix, for 2 inputs (include constant 1) and 3 units
V = np.array([[-1, 3], [1, 3], [-2, 1], [2, -4]]) # 2 x 3 matrix, for 3 inputs (include constant 1) and 2 units
W = np.array([[-1], [2], [3]])  # 3 x 1 matrix, for 3 inputs (include constant 1) and 1 ounit


# In[ ]:


X_means = np.mean(Xtrain, axis=0)
X_stds = np.std(Xtrain, axis=0)
Xtrain_st = (Xtrain - X_means) / X_stds


# In[ ]:


Zu, Zv, Y = forward(addOnes(Xtrain_st), U, V, W)
print('Zu = ', Zu)
print('Zv = ', Zv)
print('Y = ', Y)


# In[ ]:


T_means = np.mean(Ttrain, axis=0)
T_stds = np.std(Ttrain, axis=0)
Ttrain_st = (Ttrain - T_means) / T_stds
grad_wrt_U, grad_wrt_V, grad_wrt_W = gradient(Xtrain_st, Ttrain_st, Zu, Zv, Y, U, V, W)
print('grad_wrt_U = ', grad_wrt_U)
print('grad_wrt_V = ', grad_wrt_V)
print('grad_wrt_W = ', grad_wrt_W)


# In[ ]:


Y = use(Xtrain, X_means, X_stds, T_means, T_stds, U, V, W)
Y


# Here is another example that just shows the final results of training.

# In[ ]:


n = 30
Xtrain = np.linspace(0., 20.0, n).reshape((n, 1)) - 10
Ttrain = 0.2 + 0.05 * (Xtrain + 10) + 0.4 * np.sin(Xtrain + 10) + 0.2 * np.random.normal(size=(n, 1))

Xtest = Xtrain + 0.1 * np.random.normal(size=(n, 1))
Ttest = 0.2 + 0.05 * (Xtest + 10) + 0.4 * np.sin(Xtest + 10) + 0.2 * np.random.normal(size=(n, 1))


# In[ ]:


U, V, W, X_means, X_stds, T_means, T_stds = train(Xtrain, Ttrain, 5, 5, 1000, 0.01)


# In[ ]:


Y = use(Xtrain, X_means, X_stds, T_means, T_stds, U, V, W)


# In[ ]:


plt.plot(Xtrain, Ttrain)
plt.plot(Xtrain, Y);


# Let the above cell data be our base standard with values, 
# neurons in U layer = 5;
# neurons in V layer = 5;
# epochs = 1000;
# rho = 0.001;
# used for training, in the cells below I have used different values for the variable listed above through trial and error and found the best value possible.

# In[ ]:


U, V, W, X_means, X_stds, T_means, T_stds = train(Xtrain, Ttrain, 300, 100, 10000, 0.005)
Y = use(Xtrain, X_means, X_stds, T_means, T_stds, U, V, W)
plt.plot(Xtrain, Ttrain, label='Train')
plt.plot(Xtrain, Y, label='Test')
plt.legend();


# From the above cell we can say that increasing the number of neurons in the layers does not benefit us much.

# In[ ]:


U, V, W, X_means, X_stds, T_means, T_stds = train(Xtrain, Ttrain, 100, 50, 50000, 0.005)
Y = use(Xtrain, X_means, X_stds, T_means, T_stds, U, V, W)
plt.plot(Xtrain, Ttrain, label='Train')
plt.plot(Xtrain, Y, label='Test')
plt.legend();


# From the above cell we can say that increasing the number of iterations does 
# a better job, even with the decreased number of neurons.

# In[ ]:


U, V, W, X_means, X_stds, T_means, T_stds = train(Xtrain, Ttrain, 100, 50, 50000, 0.01)
Y = use(Xtrain, X_means, X_stds, T_means, T_stds, U, V, W)
plt.plot(Xtrain, Ttrain, label='Train')
plt.plot(Xtrain, Y, label='Test')
plt.legend();


# From the above cell we can say that with an increase in learning rate, 
# our data maps a bit better to the test data.

# In[ ]:


U, V, W, X_means, X_stds, T_means, T_stds = train(Xtrain, Ttrain, 75, 25, 75000, 0.01)
Y = use(Xtrain, X_means, X_stds, T_means, T_stds, U, V, W)
plt.plot(Xtrain, Ttrain, label='Train')
plt.plot(Xtrain, Y, label='Test')
plt.legend();


# From the above cell we can say that, number of iterations and 
# the learning rate have a much greater effect on our training model than the total number of neurons in our neural network.

# In[ ]:


U, V, W, X_means, X_stds, T_means, T_stds = train(Xtrain, Ttrain, 75, 25, 75000, 0.09)
Y = use(Xtrain, X_means, X_stds, T_means, T_stds, U, V, W)
plt.plot(Xtrain, Ttrain, label='Train')
plt.plot(Xtrain, Y, label='Test')
plt.legend();


# From the above cell we see that our Neural net model is overfitting, which is not good since 
# it is not possible to collect an unbiased sample of data, 
# but we can use this to our leverage by reducing 
# the number of neurons and the number of iterations and reduce overall computational load.

# In[ ]:


U, V, W, X_means, X_stds, T_means, T_stds = train(Xtrain, Ttrain, 50, 25, 50000, 0.1)
Y = use(Xtrain, X_means, X_stds, T_means, T_stds, U, V, W)
plt.plot(Xtrain, Ttrain, label='Train')
plt.plot(Xtrain, Y, label='Test')
plt.legend();


# From the above cell we can say that any significant 
# increase in the learning rate can be used as a 
# counter-weight for a decrease in the number of neurons
# and iterations without any significant loss in model accuracy

# In[ ]:


U, V, W, X_means, X_stds, T_means, T_stds = train(Xtrain, Ttrain, 30, 15, 40000, 0.6)
Y = use(Xtrain, X_means, X_stds, T_means, T_stds, U, V, W)
plt.plot(Xtrain, Ttrain, label='Train')
plt.plot(Xtrain, Y, label='Test')
plt.legend();


# Here I can conclude that the number of neurons in our neural 
# net model have a saturation point above which they do not significantly 
# contribute to increase model accuracy.

# ## Discussion

# In this markdown cell, describe what difficulties you encountered in completing this assignment. What parts were easy for you and what parts were hard?
# Answer :
# 1. A good learning rate value can help in reducing computational load.
# 2. Number of neurons in a neural net have a saturation point.
# 3. Overfitting is not good for a model, but can be used to our leverage.

# # Grading
# 
# **A1grader.tar is now available.**
# 
# Your notebook will be run and graded automatically. Test this grading process by first downloading [A1grader.tar](http://www.cs.colostate.edu/~anderson/cs545/notebooks/A1grader.tar) and extract `A1grader.py` from it. Run the code in the following cell to demonstrate an example grading session.  The remaining 10 points will be based on your discussion of this assignment.
# 
# A different, but similar, grading script will be used to grade your checked-in notebook. It will include additional tests. You should design and perform additional tests on all of your functions to be sure they run correctly before checking in your notebook.  
# 
# For the grading script to run correctly, you must first name this notebook as 'Lastname-A1.ipynb' with 'Lastname' being your last name, and then save this notebook.

# In[9]:


get_ipython().run_line_magic('run', '-i A1grader.py')


# # Extra Credit
# 
# Apply your multilayer neural network code to a regression problem using data that you choose 
# from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets.php). Pick a dataset that
# is listed as being appropriate for regression.

# In[ ]:


data = pandas.read_csv('AirQualityUCI.csv', delimiter=';', decimal=',', usecols=range(15), na_values=-200)
data = data.dropna(axis=0)
print(data.shape)
hour = [int(t[:2]) for t in data['Time']]
X = np.array(hour).reshape(-1, 1)
CO = data['CO(GT)']
T = np.array(CO).reshape(-1, 1)
print(X.shape,T.shape)


# In[ ]:


U, V, W, X_means, X_stds, T_means, T_stds = train(X, T, 30, 15, 40000, 0.6)
Y = use(X, X_means, X_stds, T_means, T_stds, U, V, W)
plt.plot(X, T,'.', label='Train')
plt.plot(X, Y,'o', label='Test')
plt.ylabel('CO')
plt.xlabel('Time in Hours')
plt.legend();


# In[ ]:




