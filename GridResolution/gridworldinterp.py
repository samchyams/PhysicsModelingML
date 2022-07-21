import os
import xlrd
import numpy as np
import pandas as pd

# Reading Data In
cwdir = os.getcwd()
filepath = "Documents\Gridworldnew.xlsx"
GridworldPath = os.path.join(cwdir, filepath)
trainingdata = pd.read_excel(GridworldPath, sheet_name='Training Data', header=1)
testingdata = pd.read_excel(GridworldPath, sheet_name='GroundTruth High Resolution', header= 0, index_col=0, skiprows=[0])

# Getting the Data from the high resolution grid  
xlist = []
ylist = []
zlist = []
i = 0.01
j = 0
while i <= 1.01 :   
    for j in range(100):
        zdata = testingdata[i].values[j]
        xdata = round(i, 2)
        ydata = format((j*0.01) +0.01, '.2f')
        xlist.append(xdata)
        ylist.append(ydata)
        zlist.append(zdata)
    i =i+0.01
xdf = pd.DataFrame(xlist)
ydf = pd.DataFrame(ylist)
zdf = pd.DataFrame(zlist)
dflist = [xdf, ydf, zdf]
testingframe = pd.concat(dflist, axis = 1)
testingframe.columns = ['x','y','z']


# Preparing Data
trainX = trainingdata.loc[:,["x", "y"]]
trainY = trainingdata.loc[:,"z"]
testX = testingframe.loc[:,["x", "y"]]
testY = testingframe.loc[:,"z"]

#Building ML Model
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

# Writing Data Back out to Excel File
predframe = pd.DataFrame(forestpredict, columns=['z'])
exportframe = [testX, predframe]
axes = np.arange(0.01,1.01,0.01)
exportframe = pd.DataFrame(index=[axes],columns=[axes])
row = 0
q = 0
while row <= 99:
    col = 0
    while col <= 99:
        exportframe.iat[row,col] = predframe.iat[q,0]
        q = q + 1
        col = col + 1
    row = row + 1

newpath = "C:\\Users\\sam.hyams\\Documents\\GridworldML.xlsx"
exportframe.to_excel(newpath, sheet_name='ForestPredictions')
