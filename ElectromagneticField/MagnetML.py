from MagnetDataGrab import *
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# Building Training Data
filepath = "C:\\Users\\sam.hyams\\Documents\\MagnetData.csv"
trainX = pd.read_csv(filepath, usecols=[0], skiprows=(2,8))
trainY = pd.read_csv(filepath, skiprows =(2,8))
trainY = trainY.drop(columns=trainY.columns[0], axis=1)
# Building Testing Data
scalar = 0.4
if scalar == 0.4:
    xtest = pd.read_csv(filepath, header=None, usecols=[0], skiprows=2, nrows=1)
    ytest = pd.read_csv(filepath, header=None, skiprows=2, nrows=1)
    ytest.drop(columns=ytest.columns[0], axis=1, inplace=True)
elif scalar == 1.6:
    xtest = pd.read_csv(filepath, header=None, usecols=[0], skiprows=8, nrows=1)
    ytest = pd.read_csv(filepath, header=None, skiprows=8, nrows=1)
    ytest.drop(columns=ytest.columns[0], axis=1, inplace=True)
else: 
    print("Rerun program with valid portion of the test set")
    quit()

#### Building the Random Forest ####
from sklearn.ensemble import ExtraTreesRegressor
forest = ExtraTreesRegressor(n_estimators=200, random_state=50)
forest.fit(trainX, trainY)
forestpredict = forest.predict(xtest)
forestpredict = np.squeeze(forestpredict) # Cleaning Data
simdata = ytest.squeeze()

# Plotting the sim vs predicted tree data 
plt.figure(figsize=[6.8,4.8],dpi=1200)
plt.plot(xposition,simdata,color='darkblue',label='Simulation data')
plt.scatter(xposition, forestpredict, color='lightcoral', label='Random forest predictions', edgecolors='black')
plt.grid(True)
plt.legend(loc="upper right")
plt.xlabel("Distance from Magnet (m)")
plt.ylabel("Magnetic Field Intensity (A)")
plt.title(f"Magnetic Potential {scalar} Plot")
plt.savefig(f"Documents\MagnetResults\{scalar}Plot.png", bbox_inches='tight')

# Evaluation of Random Forest
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error as mae
final_mse = mean_squared_error(simdata, forestpredict)
final_mae = mae(simdata, forestpredict)
final_rmse = np.sqrt(final_mse)
print("Final RMSE: ", final_rmse)
print("Final MAE: ", final_mae)

runMLtimers = True
if runMLtimers: 
    MLtimers(trainX, trainY)