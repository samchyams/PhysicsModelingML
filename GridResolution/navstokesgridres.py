import os
import xlrd
import numpy as np
import pandas as pd
from NavierStokesCFD import *

############################################### Getting Actual Data #########################################################
# Reading in Steady state data 
gridpath = "C:\\Users\\sam.hyams\\Documents\\Navier Stokes Results\\WaterResults\\PUV27500.txt"
udata = np.genfromtxt(gridpath, usecols=(1))
meshsize = 257
# Constructing data frame 
axes = np.arange(1,meshsize + 1,1)
exportframe = pd.DataFrame(index=[axes], columns=[axes])
row = 0
q = 0
while row <= (meshsize-1):
    col = 0
    while col <= (meshsize-1):
        exportframe.iat[row,col] = udata[q]
        q = q + 1
        col = col + 1
    row = row + 1


newpath = "C:\\Users\\sam.hyams\\Documents\\NavStokesGridExpansion.xlsx"
#exportframe.to_excel(newpath, sheet_name='257x257MeshGrid')

############################################### Setting Up Training Data #########################################################
traingridpath = "C:\\Users\\sam.hyams\\Documents\\Navier Stokes Results\\65x65griddata.txt"
trainudata = np.genfromtxt(traingridpath, usecols=(1))

axes = np.arange(1,meshsize+1,4)
trainingframe = pd.DataFrame(index=[axes], columns=[axes])
row = 0
q = 0
while row <= 64:
    col = 0
    while col <= 64:
        trainingframe.iat[row,col] = trainudata[q]
        q = q + 1
        col = col + 1
    row = row + 1

trainingpath = "C:\\Users\\sam.hyams\\Documents\\NavStokesGridExpansionTraining.xlsx"
#trainingframe.to_excel(trainingpath, sheet_name='TrainingGrid')

# Generating X and Y labels for Training data
trainxlist = []
trainylist = []
i = 1
j = 1
while i <= meshsize+1 :  
    for j in range(1,meshsize+1,4):
        trainxlist.append(i)
        trainylist.append(j)
    i =i+4

trainxframe = pd.DataFrame(trainxlist)
trainyframe = pd.DataFrame(trainylist)

trainxportion = trainxframe.assign(y = trainylist)
trainxportion.columns = ['x', 'y']

# Generating X and Y labels for Testing data
testxlist = []
testylist = []
i = 1
j = 1
while i <= meshsize :  
    for j in range(meshsize):
        testxlist.append(i)
        testylist.append(j+1)
    i =i+1
testxframe = pd.DataFrame(testxlist)
testyframe = pd.DataFrame(testylist)

testxportion = testxframe.assign(y = testylist)
testxportion.columns = ['x', 'y']

############################################### Implementing ML Alg and Getting Prediction Grid #########################################################
# Preparing Data
trainX = trainxportion.loc[:,["x", "y"]]
trainY = trainudata
testX = testxportion.loc[:,["x", "y"]]
testY = udata

# Building ML Model 
from sklearn.ensemble import ExtraTreesRegressor
forest = ExtraTreesRegressor(n_estimators=200, random_state=50)
forest.fit(trainX, trainY)
forestpredict = forest.predict(testX)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error as mae
final_mse = mean_squared_error(testY, forestpredict)
final_mae = mae(testY, forestpredict)
final_rmse = np.sqrt(final_mse)
print("Final RMSE: ", final_rmse)
print("Final MAE: ", final_mae)

############################################### Writing Out Prediction Grid Back Out to Excel #########################################################
axes = np.arange(1,meshsize+1,1)
predictionframe = pd.DataFrame(index=[axes], columns=[axes])
row = 0
q = 0
while row <= (meshsize-1):
    col = 0
    while col <= (meshsize-1):
        predictionframe.iat[row,col] = forestpredict[q]
        q = q + 1
        col = col + 1
    row = row + 1

predictionpath = "C:\\Users\\sam.hyams\\Documents\\NavStokesGridExpansionMLPredictions.xlsx"
#predictionframe.to_excel(predictionpath, sheet_name='ForestPredictionsfor257x257')

################# Regression Timers and Accuracy ###################### 
from lazypredict.Supervised import LazyRegressor
import seaborn as sns; sns.set()
from sklearn.utils import shuffle
from sklearn.model_selection import RandomizedSearchCV

# This function will take in data and run all kinds of ML techniques on them and output the time that it took to run and a basic evaluation metric
def MLtimers(X, y):
    X, y = shuffle(X, y, random_state=13)
    X = X.astype(np.float32)

    offset = int(X.shape[0] * 0.9)

    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]

    reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)

    print(models)

# Switch to use the MLtimers function
runMLtimers = True
if runMLtimers: 
    MLtimers(trainX, trainY)