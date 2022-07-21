import os
import csv 
import numpy as np

cwdir = os.getcwd()
filepath = "Documents\sam_stressanalysis\sam_stressproblem"
startpath = os.path.join(cwdir, filepath)
force = 0
StressBigData = os.path.join(cwdir, "Documents/StressAnalysisBigData.csv")

# Gets all the Data for the Xline, XYline, and Yline and puts it into a file
GenFile = False #switch for functionality
if GenFile:
    while force <= 20000: 
        forcepath = f"plateHole.{force}\postProcessing"
        holepath = os.path.join(startpath, forcepath)
        xlpath = os.path.join(holepath, "xline/200/line_sigmaxx.xy")
        xylpath = os.path.join(holepath, "xyline/200/line_sigmaxx.xy")
        ylpath = os.path.join(holepath, "yline/200/line_sigmaxx.xy")
        file = open(StressBigData, 'a+')
        xdata = np.genfromtxt(xlpath, usecols=(1))
        xydata = np.genfromtxt(xylpath, usecols=(1))
        ydata = np.genfromtxt(ylpath, usecols=(1))
        xdatalist = xdata.tolist()
        xydatalist = xydata.tolist()
        ydatalist = ydata.tolist()
        final_row = [force] + xdatalist + xydatalist + ydatalist
        with file:
            write = csv.writer(file)
            write.writerow(final_row)
        force = force + 2000

def GenData(force):
    forcepath = f"plateHole.{force}\postProcessing"
    holepath = os.path.join(startpath, forcepath)
    xlpath = os.path.join(holepath, "xline/200/line_sigmaxx.xy")
    xylpath = os.path.join(holepath, "xyline/200/line_sigmaxx.xy")
    ylpath = os.path.join(holepath, "yline/200/line_sigmaxx.xy")
    xdata = np.genfromtxt(xlpath, usecols=(1))
    xydata = np.genfromtxt(xylpath, usecols=(1))
    ydata = np.genfromtxt(ylpath, usecols=(1))
    xdatalist = xdata.tolist()
    xydatalist = xydata.tolist()
    ydatalist = ydata.tolist()
    return xdatalist, xydatalist, ydatalist

def GenPosition(force):
    forcepath = f"plateHole.{force}\postProcessing"
    holepath = os.path.join(startpath, forcepath)
    xlpath = os.path.join(holepath, "xline/200/line_sigmaxx.xy")
    xylpath = os.path.join(holepath, "xyline/200/line_sigmaxx.xy")
    ylpath = os.path.join(holepath, "yline/200/line_sigmaxx.xy")
    xposition = np.genfromtxt(xlpath, usecols=(0))
    xyposition = np.genfromtxt(xylpath, usecols=(0))
    yposition = np.genfromtxt(ylpath, usecols=(0))
    xpositionlist = xposition.tolist()
    xypositionlist = xyposition.tolist()
    ypositionlist = yposition.tolist()
    return xpositionlist, xypositionlist, ypositionlist


# Generating Sets of Data
xpositionlist, xypositionlist, ypositionlist = GenPosition(6000)
xdatalist0, xydatalist0, ydatalist0 = GenData(0)
xdatalist2000, xydatalist2000, ydatalist2000 = GenData(2000)
xdatalist4000, xydatalist4000, ydatalist4000 = GenData(4000)
xdatalist6000, xydatalist6000, ydatalist6000 = GenData(6000)
xdatalist8000, xydatalist8000, ydatalist8000 = GenData(8000)
xdatalist12000, xydatalist12000, ydatalist12000 = GenData(12000)
xdatalist16000, xydatalist16000, ydatalist16000 = GenData(16000)
xdatalist18000, xydatalist18000, ydatalist18000 = GenData(18000)
xdatalist20000, xydatalist20000, ydatalist20000 = GenData(20000)

################# Regression Timers and Accuracy ###################### 
import lazypredict
from lazypredict.Supervised import LazyRegressor
import seaborn as sns; sns.set()
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import RandomizedSearchCV
import random

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
