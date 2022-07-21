from Stressanalysisdatagrab import *
import matplotlib.pyplot as plt
import pandas as pd

# Getting Training and Testing Data from Constructed Files
trainpath = "C:\\Users\\sam.hyams\\Documents\\StressAnalysisTraining.csv"
trainX = pd.read_csv(trainpath, usecols=[0]) # getting force numbers only
dataframe = pd.read_csv(trainpath)
firstcol = dataframe.columns[0]              
trainY = dataframe.drop([firstcol], axis=1)  # getting the rest of the data 

testpath = "c:/Users/sam.hyams/Documents/StressAnalysisTesting2.csv"
force = 18000
xtest = pd.read_csv(testpath, usecols=[0])
dataframe = pd.read_csv(testpath)
ytest = dataframe.drop([firstcol], axis=1)
ytest = ytest.to_numpy()       # converting data frame to np array
ytest = np.squeeze(ytest)      # stripping a dimension off of np array (has 2 needs 1)

# Building the Random Forest (ExtraTrees - Random Forest that does a little more to prevent overfitting)
from sklearn.ensemble import ExtraTreesRegressor
forest = ExtraTreesRegressor(n_estimators=200, random_state=50)
forest.fit(trainX, trainY)
forestpredict = forest.predict(xtest)

# Preparing the Forest Data
forestpredict = np.squeeze(forestpredict) # strip a dimension from the array 
xlineforestpred = forestpredict[:100]
xylineforestpred = forestpredict[100:200]
ylineforestpred = forestpredict[200:]

# Plotting the simulation vs the predicted tree data for the lines 
plt.figure(figsize=[6.8,4.8],dpi=1200)
plt.plot(xpositionlist,xdatalist18000,color="darkblue",label='Simulation data')
plt.scatter(xpositionlist,xlineforestpred, color = "lightcoral",label='Random forest predictions', edgecolors='black')
plt.legend(loc="upper right")
plt.xlabel("Distance from Center (m)")
plt.ylabel("XX Force (Pa)")
plt.title(f"{force}Pa X Line Plot")
plt.savefig(f"Documents\StressAnalysisGraphs\{force}PaXLinePlot.png", bbox_inches='tight')

plt.figure(figsize=[6.8,4.8],dpi=1200)
plt.plot(xypositionlist,xydatalist18000,color="darkblue",label='Simulation data')
plt.scatter(xypositionlist,xylineforestpred, color = "lightcoral",label='Random forest predictions', edgecolors='black')
plt.legend(loc="upper left")
plt.xlabel("Distance from Center (m)")
plt.ylabel("XX Force (Pa)")
plt.title(f"{force}Pa XY Line Plot")
plt.savefig(f"Documents\StressAnalysisGraphs\{force}PaXYLinePlot.png", bbox_inches='tight')

plt.figure(figsize=[7,4.8],dpi=1200)
plt.plot(ypositionlist,ydatalist18000,color="darkblue",label='Simulation data')
plt.scatter(ypositionlist,ylineforestpred, color = "lightcoral",label='Random forest predictions', edgecolors='black')
plt.legend(loc="upper left")
plt.xlabel("Distance from Center (m)")
plt.ylabel("XX Force (Pa)")
plt.title(f"{force}Pa Y Line Plot")
plt.savefig(f"Documents\StressAnalysisGraphs\{force}PaYLinePlot.png", bbox_inches='tight')

# Evaluation of Random Forest
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error as mae
final_mse = mean_squared_error(ytest, forestpredict)
final_mae = mae(ytest, forestpredict)
final_rmse = np.sqrt(final_mse)
print("Final RMSE: ", final_rmse)
print("Final MAE: ", final_mae)

# Switch to use the MLtimers function
runMLtimers = True
if runMLtimers: 
    MLtimers(trainX, trainY)