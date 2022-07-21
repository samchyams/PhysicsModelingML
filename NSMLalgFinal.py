from pstats import Stats
from NavierStokesCFD import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

################# Regression Timers and Accuracy ###################### 
from lazypredict.Supervised import LazyRegressor
import seaborn as sns; sns.set()
from sklearn.utils import shuffle

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

# Reading in Training/Testing Data - Made files manually from FinalData file
# change file names/paths accordingly 
import pandas as pd
trainpath = "c:/Users/sam.hyams/Documents/NavierStokesFinalDataTrain.csv"
trainX = pd.read_csv(trainpath, usecols=[0]) # getting Reynolds numbers only
dataframe = pd.read_csv(trainpath)
firstcol = dataframe.columns[0]              
trainY = dataframe.drop([firstcol], axis=1)  # getting the rest of the data 

testpath = "c:/Users/sam.hyams/Documents/NavierStokesFinalDataTest1.csv"
Re = 300
xtest = pd.read_csv(testpath, usecols=[0])
dataframe = pd.read_csv(testpath)
ytest = dataframe.drop([firstcol], axis=1)
ytest = ytest.to_numpy()       # converting data frame to np array
ytest = np.squeeze(ytest)      # stripping a dimension off of np array (has 2 needs 1)


# Building Random Forest Model (ExtraTrees - Random Forest that does a little more to prevent overfitting)
from sklearn.ensemble import ExtraTreesRegressor
forest = ExtraTreesRegressor(n_estimators=200, random_state=50)
forest.fit(trainX, trainY)
forestpredict = forest.predict(xtest)

# Preparing the Forest Data 
forestpredict = np.squeeze(forestpredict) # strip a dimension from the array 
vforestpredictions = forestpredict[:63]  # getting the x portion of the data (the first 63)
uforestpredictions = forestpredict[63:]  # getting the y portion of the data (the last 63)

samp = np.arange(start=4, stop=253, step=4) # getting our sampling of points 
scale = (length*samp)/257  # Need to scale to Get Graphs looking right


# Getting out of Results folder
resetpath = "C:\\Users\\sam.hyams"
os.chdir(resetpath)
# Plotting For Random Forest Prediction 
plt.figure(dpi=1200)
plt.plot(y,u_c[:,int(np.ceil(colpts/2))],"darkblue",label='Simulation data')
plt.scatter(scale,uforestpredictions, color = "lightcoral",label='Random Forest predictions', edgecolors='black')
plt.legend(loc="upper left")
plt.grid(True)
plt.xlabel("Vertical distance along center")
plt.ylabel("Horizontal velocity")
plt.title(f"Re {Re} Y-axis Data")
plt.savefig(f"Documents\StokesGraphs\Re{Re}Plot1.png", bbox_inches='tight')


plt.figure(dpi=1200)
plt.plot(x,v_c[int(np.ceil(rowpts/2)),:],"darkblue",label='Simulation data')
plt.scatter(scale,vforestpredictions, color = "lightcoral",label='Random Forest predictions', edgecolors='black')
plt.legend()
plt.xlabel("Horizontal distance along center")
plt.ylabel("Vertical velocity")
plt.title(f"Re {Re} X-axis Data")
plt.savefig(f"Documents\StokesGraphs\Re{Re}Plot2.png", bbox_inches='tight')

# Switch to use the MLtimers function
runMLtimers = True
if runMLtimers: 
    MLtimers(trainX, trainY)

# Evaluation of Random Forest
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error as mae
final_mse = mean_squared_error(ytest, forestpredict)
final_mae = mae(ytest, forestpredict)
final_rmse = np.sqrt(final_mse)
print("Final RMSE: ", final_rmse)
print("Final MAE: ", final_mae)